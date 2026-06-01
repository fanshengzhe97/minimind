import time
import argparse
import random
import warnings
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')


def _strip_ckpt_suffix(name: str) -> str:
    """把 xxx.pth / xxx_resume.pth 统一成 tag（不带后缀）。"""
    base = os.path.basename(name)
    if base.endswith('.pth'):
        base = base[:-4]
    if base.endswith('_resume'):
        base = base[:-7]
    return base


def parse_ckpt_tag(tag: str) -> dict:
    """从 ckpt 命名里解析模型结构参数。

    支持形如：
    - MiniMind-Pretrain-DSxxx-L8-H768-S340-MoE4K1-BS32-GA8-LR...-Ep...-P...-A...
    - MiniMind-Full-SFT-DSxxx-L8-H768-S768-MoE0K0-...
    """
    out = {}
    m = re.search(r"-L(?P<L>\d+)-H(?P<H>\d+)(?:-S(?P<S>\d+))?", tag)
    if m:
        out["num_hidden_layers"] = int(m.group("L"))
        out["hidden_size"] = int(m.group("H"))
        if m.group("S") is not None:
            out["max_seq_len"] = int(m.group("S"))

    m = re.search(r"-MoE(?P<E>\d+)K(?P<K>\d+)", tag)
    if m:
        e = int(m.group("E"))
        k = int(m.group("K"))
        out["use_moe"] = 1 if e > 0 else 0
        out["num_experts"] = e if e > 0 else None
        out["num_experts_per_tok"] = k if e > 0 else None

    return out


def resolve_ckpt_path(args) -> str:
    """兼容传 weight/tag 或直接传 .pth 路径。"""
    w = args.weight
    if isinstance(w, str) and (w.endswith('.pth') or '/' in w or w.startswith('.')):
        return w
    return os.path.join(args.save_dir, f"{w}.pth")


def infer_args_from_ckpt_name(args):
    """从 ckpt 文件名推断一些关键配置（若命名包含则覆盖 args）。"""
    ckpt_path = resolve_ckpt_path(args)
    tag = _strip_ckpt_suffix(ckpt_path)
    args.ckpt_tag = tag
    parsed = parse_ckpt_tag(tag)

    # 只要能从命名里解析出来，就覆盖（避免手工传参出错）
    if "hidden_size" in parsed:
        args.hidden_size = parsed["hidden_size"]
    if "num_hidden_layers" in parsed:
        args.num_hidden_layers = parsed["num_hidden_layers"]
    if "use_moe" in parsed:
        args.use_moe = parsed["use_moe"]
    if parsed.get("num_experts") is not None:
        args.num_experts = parsed["num_experts"]
    if parsed.get("num_experts_per_tok") is not None:
        args.num_experts_per_tok = parsed["num_experts_per_tok"]

    return ckpt_path

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        cfg_kwargs = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "use_moe": bool(args.use_moe),
            "inference_rope_scaling": args.inference_rope_scaling,
        }
        # MoE 的专家数/TopK 若能从 ckpt 命名解析出来，则一并带上
        if bool(args.use_moe):
            cfg_kwargs["num_experts"] = getattr(args, "num_experts", 4)
            cfg_kwargs["num_experts_per_tok"] = getattr(args, "num_experts_per_tok", 1)
        model = MiniMindForCausalLM(MiniMindConfig(**cfg_kwargs))

        ckp = getattr(args, "ckpt_path", None) or resolve_ckpt_path(args)
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重tag或路径：支持传 run_tag（不带.pth）或直接传 /path/to/xxx.pth")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--num_experts', default=4, type=int, help="MoE专家数（可从ckpt命名中自动解析覆盖）")
    parser.add_argument('--num_experts_per_tok', default=1, type=int, help="MoE TopK（可从ckpt命名中自动解析覆盖）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    # 适配新命名：从 ckpt tag 里自动推断 hidden_size/layers/moe 等
    args.ckpt_path = infer_args_from_ckpt_name(args)
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        tag_lower = getattr(args, 'ckpt_tag', args.weight).lower() if isinstance(getattr(args, 'ckpt_tag', args.weight), str) else str(args.weight).lower()
        if 'pretrain' in tag_lower:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
