import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
    get_model_params,
    get_world_size,
    estimate_train_flops,
    wandb_define_flops_xaxis,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, perf_state=None, active_params_m: float = 0.0):
    start_time = time.time()
    last_step = start_step
    if perf_state is None:
        perf_state = {"tokens": 0, "flops": 0.0}
    ws = get_world_size()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        iter_start = time.time()
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # 统计 tokens / FLOPs（按全局 world_size 计）
        step_tokens = int(input_ids.numel()) * ws
        perf_state["tokens"] += step_tokens
        if active_params_m > 0:
            perf_state["flops"] += estimate_train_flops(step_tokens, active_params_m)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        grad_norm = None
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # clip_grad_norm_ 返回的是“裁剪前”的总范数，便于监控稳定性
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                iter_time = max(time.time() - iter_start, 1e-6)
                log_data = {
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                    "train/tokens": perf_state["tokens"],
                    "train/flops": perf_state["flops"],
                    "train/iter_time": iter_time,
                    "train/tokens_per_sec": float(step_tokens) / iter_time,
                }
                if grad_norm is not None:
                    log_data["train/grad_norm"] = grad_norm

                # GPU 显存（仅 CUDA）
                if torch.cuda.is_available() and ("cuda" in str(args.device)):
                    try:
                        log_data.update({
                            "gpu/mem_alloc_mb": float(torch.cuda.memory_allocated() / 1024 / 1024),
                            "gpu/mem_reserved_mb": float(torch.cuda.memory_reserved() / 1024 / 1024),
                            "gpu/max_mem_alloc_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
                        })
                    except Exception:
                        pass

                # MoE 路由统计（与 pretrain 同款）
                if lm_config.use_moe:
                    try:
                        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                        raw_model = getattr(raw_model, '_orig_mod', raw_model)
                        layers = getattr(getattr(raw_model, 'model', None), 'layers', None)
                        if layers is not None:
                            stats = []
                            for l in layers:
                                s = getattr(getattr(l, 'mlp', None), 'last_router_stats', None)
                                if s:
                                    stats.append(s)
                            if stats:
                                def _mean(key: str):
                                    vs = [float(x[key].detach().cpu()) for x in stats if key in x and x[key] is not None]
                                    return sum(vs) / len(vs) if vs else None
                                for k in ["load_entropy", "load_cv", "utilization", "max_load", "min_load"]:
                                    v = _mean(k)
                                    if v is not None:
                                        log_data[f"moe/{k}"] = v
                    except Exception:
                        pass

                wandb.log(log_data)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            # ckpt 命名与 wandb run name 对齐（与 pretrain 一致）
            ckp = f'{args.save_dir}/{run_tag}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=run_tag, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--resume_weight', default='', type=str, help="从哪个断点继续训练（用于加载 optimizer/step 等），为空则默认与 save_weight/run_tag 一致")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))

    # 统一 run/ckpt 命名（wandb name / out ckpt / checkpoints 同名）
    # - 如果用户手动传入了完整命名（例如 MiniMind-xxx-DS...-P...-A...），则直接使用
    # - 否则按当前 full_sft 训练参数生成 full_sft 的命名
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    moe_experts = lm_config.num_experts if lm_config.use_moe else 0
    moe_topk = lm_config.num_experts_per_tok if lm_config.use_moe else 0
    if args.save_weight != 'full_sft' and ('MiniMind-' in args.save_weight):
        base_tag = args.save_weight
        run_tag = base_tag
    else:
        base_tag = (
            # full_sft 使用自己的命名前缀，避免与 pretrain 混淆
            f"MiniMind-Full-SFT-"
            f"DS{dataset_name}-"
            f"L{args.num_hidden_layers}-H{args.hidden_size}-S{args.max_seq_len}"
            f"-MoE{moe_experts}K{moe_topk}-BS{args.batch_size}-GA{args.accumulation_steps}"
            f"-LR{args.learning_rate}-Ep{args.epochs}"
        )
        run_tag = base_tag

    # 先不加载断点；等模型初始化后再决定最终 run_tag（需要参数量 P/A）
    ckp_data = None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # 若不是用户显式传入完整 run_tag，则补齐参数量（P/A）
    total_params_m, active_params_m = get_model_params(model, lm_config, log=False)
    if run_tag == base_tag and not (args.save_weight != 'full_sft' and ('MiniMind-' in args.save_weight)):
        fmt = lambda x: (f"{x:.1f}".replace('.', 'p'))
        run_tag = f"{base_tag}-P{fmt(total_params_m)}M-A{fmt(active_params_m)}M"

    # ========== 5. 检查/加载断点（与 pretrain 同款回退策略） ==========
    resume_weight = args.resume_weight.strip() if isinstance(args.resume_weight, str) else ''
    if args.from_resume == 1:
        # 允许“从 A 的断点恢复，但保存为 B 的 run_tag”（典型：用 pretrain 的 resume 继续跑 sft）
        if resume_weight:
            ckp_data = lm_checkpoint(lm_config, weight=resume_weight, save_dir='../checkpoints')
        else:
            ckp_data = lm_checkpoint(lm_config, weight=run_tag, save_dir='../checkpoints')
            if ckp_data is None:
                ckp_data = lm_checkpoint(lm_config, weight=base_tag, save_dir='../checkpoints')
            if ckp_data is None:
                ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')

    # ========== 6. 配wandb（run name 与 ckpt 对齐，和 pretrain 一致） ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 仅当“恢复的断点就是当前 run_tag”时，才尝试续接同一个 wandb run
        # 否则（例如从 pretrain resume 接着跑 sft，保存为新的 run_tag）默认新开 run，避免串到 pretrain 的 run
        can_resume_wandb = (not resume_weight) or (resume_weight == run_tag)
        wandb_id = ckp_data.get('wandb_id') if (ckp_data and can_resume_wandb) else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=run_tag, id=wandb_id, resume=resume)

        # 让 loss 曲线支持 FLOPs 横坐标（与 pretrain 同款；不影响默认 step 横坐标）
        wandb_define_flops_xaxis(wandb, y_keys=["loss", "logits_loss", "aux_loss"], x_key="train/flops")
        # 额外记录到 wandb config（便于筛选/对比）
        if hasattr(wandb, 'config'):
            try:
                wandb.config.update({
                    "data/name": dataset_name,
                    "data/path": args.data_path,
                    "model/num_hidden_layers": args.num_hidden_layers,
                    "model/hidden_size": args.hidden_size,
                    "data/max_seq_len": args.max_seq_len,
                    "moe/use_moe": bool(args.use_moe),
                    "moe/num_experts": moe_experts,
                    "moe/num_experts_per_tok": moe_topk,
                    "train/batch_size": args.batch_size,
                    "train/grad_accumulation_steps": args.accumulation_steps,
                }, allow_val_change=True)
            except Exception:
                pass

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    perf_state = {"tokens": 0, "flops": 0.0}
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, perf_state=perf_state, active_params_m=active_params_m)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, perf_state=perf_state, active_params_m=active_params_m)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
