import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import re
import gc
import json
import math
import random
import signal
import argparse
import warnings
from collections import Counter
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
    LMForRewardModel,
    get_model_params,
)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')


def _parse_sft_seq_len_from_weight_name(weight_name: str):
    """从 sft 阶段 weight 名称里解析长度信息。

    约定：权重命名里包含 "-S{len}"（例如 MiniMind-Full-SFT-...-S768-...）。
    agent RL 权重可能是 "-S{seq}+{gen}"，这里也兼容，取 seq 部分。
    """
    if not weight_name or not isinstance(weight_name, str):
        return None
    m = re.search(r"-S(\d+)(?:\+(\d+))?", weight_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# 与 trainer/train_full_sft.py 的默认 max_seq_len 保持一致：默认长度不进 tag
SFT_DEFAULT_SEQ_LEN = 768

# ================================ 工具与 Reward = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义 ========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行 ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
CHECK_ARGS = {
    # 注意：模型可能输出非法 arguments（非 dict / JSON 解析失败）。这里统一做类型保护，
    # 让 reward 侧把它计为 invalid，而不是直接 crash。
    "calculate_math": lambda a: isinstance(a, dict) and bool(a.get("expression")),
    "unit_converter": lambda a: isinstance(a, dict) and a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: isinstance(a, dict) and bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: isinstance(a, dict) and bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: isinstance(a, dict) and bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except: pass
    return calls

def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except:
        return None
    finally:
        try: signal.alarm(0)
        except: pass

# ======== 多轮 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        unfinished = turn == max_turns - 1
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})

        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== Reward 计算 ========
def validate_gt_in_text(text, gt_list):
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    """Agent RL reward.

    Returns:
        rewards: 原始 reward（用于训练 loss）。
        rewards_wo_len: 去掉显式“长度奖励”（response 长度分 + think 长度分）的 reward（仅用于监控/打点）。
    """
    rewards = torch.zeros(len(completions), device=device)
    rewards_wo_len = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, len_reward, answer = 0.0, 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))  # 解析tool调用
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
        # -------- 无工具调用：格式+reward奖励 --------
        if not tool_calls:
            # 长度分（显式 length-based；wo_len 不包含）
            s = 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
            reward += s
            len_reward += s
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                # 思考长度分（显式 length-based；wo_len 不包含）
                s = 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                reward += s
                len_reward += s
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
                answer = answer.strip()
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score  # RM分
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
            rewards_wo_len[idx] = max(min(reward - len_reward, 3.0), -3.0)
        # -------- 有工具调用：执行结果奖励 --------
        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                # 模型可能直接给出数字/列表等，避免 .get AttributeError
                if not isinstance(raw, dict):
                    raw = {}
                check = CHECK_ARGS.get(name)
                try:
                    ok = bool(name in valid_names and check and check(raw))
                except Exception:
                    ok = False
                valid_call_count += int(ok)
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap  # tool对齐分
            
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # GT分
            if unfinished: reward -= 0.5  # 未完成扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)  # 总分Clip
            rewards_wo_len[idx] = rewards[idx]
    return rewards, rewards_wo_len


def _safe_quantile(x: torch.Tensor, q: float, default: float = 0.0) -> float:
    """torch.quantile wrapper for empty tensors."""
    if x is None:
        return default
    x = x.detach()
    if x.numel() == 0:
        return default
    return torch.quantile(x.float(), q).item()


def _masked_stats(x: torch.Tensor, mask: torch.Tensor):
    """Return (mean, std, p95, max) over x[mask]."""
    if x is None or mask is None:
        return 0.0, 0.0, 0.0, 0.0
    m = mask.bool()
    if m.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    v = x.detach()[m]
    if v.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0
    v = v.float()
    return v.mean().item(), v.std(unbiased=False).item(), _safe_quantile(v, 0.95, 0.0), v.max().item()


def _compute_tool_health_metrics(turn_outputs_batch, tools_batch, gt_batch, num_gen: int, unfinished_batch=None):
    """Compute tool-call related health metrics for agent RL."""
    total_samples = len(turn_outputs_batch) if turn_outputs_batch is not None else 0
    total_calls = 0
    tool_call_samples = 0
    invalid_name_calls = 0
    arg_invalid_calls = 0
    valid_calls = 0
    tool_gap_sum = 0.0
    unfinished_cnt = 0
    gt_hit_sum = 0.0
    gt_hit_n = 0

    name_counter = Counter()

    if total_samples == 0:
        return {
            "tool_call_rate": 0.0,
            "invalid_tool_call_rate": 0.0,
            "arg_invalid_call_rate": 0.0,
            "valid_call_rate": 0.0,
            "tool_gap_mean": 0.0,
            "unfinished_rate": 0.0,
            "gt_hit_rate": 0.0,
            "top_tool_counts": {},
        }

    for idx, turn_outputs in enumerate(turn_outputs_batch):
        sample_idx = idx // num_gen if num_gen > 0 else 0
        tools = tools_batch[sample_idx] if tools_batch is not None else None
        gt = gt_batch[sample_idx] if gt_batch is not None else None
        valid_names = {t["function"]["name"] for t in tools} if tools else set()

        unfinished = bool(unfinished_batch[idx]) if unfinished_batch is not None else False
        unfinished_cnt += int(unfinished)

        # 与 calculate_rewards 保持一致：每 turn 取 </think> 之后的部分
        turn_answers = [
            (t.split("</think>", 1)[-1].strip() if "</think>" in t else t.strip())
            for t in (turn_outputs or [])
        ]
        answer = turn_answers[-1] if turn_answers else ""

        tool_calls = []
        for t in turn_answers:
            tool_calls.extend(parse_tool_calls(t))

        if tool_calls:
            tool_call_samples += 1

        valid_call_count = 0
        for call in tool_calls:
            name = call.get("name", "")
            raw = call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}

            total_calls += 1
            name_counter[name] += 1

            if name not in valid_names:
                invalid_name_calls += 1
                continue

            check = CHECK_ARGS.get(name)
            if check is not None and check(raw):
                valid_call_count += 1
                valid_calls += 1
            else:
                arg_invalid_calls += 1

        # tool_gap 定义与 reward 中一致
        gt_len = len(gt) if gt else 0
        tool_gap = abs(valid_call_count - gt_len) + max(0, len(tool_calls) - valid_call_count)
        tool_gap_sum += float(tool_gap)

        # gt_hit_rate：看最终答案里命中多少 gt
        final_text = "" if unfinished else (answer.split("</tool_call>")[-1] if "</tool_call>" in answer else answer)
        if gt:
            verified = validate_gt_in_text(final_text, gt)
            gt_hit_sum += (len(verified) / max(len(gt), 1))
            gt_hit_n += 1

    tool_call_rate = tool_call_samples / max(total_samples, 1)
    invalid_tool_call_rate = invalid_name_calls / max(total_calls, 1)
    arg_invalid_call_rate = arg_invalid_calls / max(total_calls, 1)
    valid_call_rate = valid_calls / max(total_calls, 1)
    tool_gap_mean = tool_gap_sum / max(total_samples, 1)
    unfinished_rate = unfinished_cnt / max(total_samples, 1)
    gt_hit_rate = gt_hit_sum / max(gt_hit_n, 1)

    # 注意：swanlab 对 scalar chart 值类型要求较严格，string 作为 scalar 会报错。
    # 因此这里不返回 top_tool_name 字段，只返回可数值化的 top_tool_counts，供外层展开成动态 key。
    top3 = name_counter.most_common(3)
    top_tool_counts = {name: int(cnt) for name, cnt in top3 if name}

    return {
        "tool_call_rate": tool_call_rate,
        "invalid_tool_call_rate": invalid_tool_call_rate,
        "arg_invalid_call_rate": arg_invalid_call_rate,
        "valid_call_rate": valid_call_rate,
        "tool_gap_mean": tool_gap_mean,
        "unfinished_rate": unfinished_rate,
        "gt_hit_rate": gt_hit_rate,
        "top_tool_counts": top_tool_counts,
    }

# ================================ 工具与 Reward = End ================================
def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']
        last_step = step

        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        rewards, rewards_wo_len = calculate_rewards(
            prompts,
            completions,
            gt_batch,
            tools_batch,
            args.num_generations,
            reward_model,
            device=args.device,
            turn_outputs_batch=turn_outputs_batch,
            unfinished_batch=unfinished_batch,
        )

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        completion_mask = full_response_masks[:, 1:]
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            ar_wo_len = rewards_wo_len.mean().item()
            al = token_counts.float().mean().item()
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                # ===== reward/len 分布 =====
                r = rewards.detach()
                r_std = r.std(unbiased=False).item()
                r_p50 = _safe_quantile(r, 0.50, ar)
                r_p90 = _safe_quantile(r, 0.90, ar)
                r_min = r.min().item() if r.numel() else ar
                r_max = r.max().item() if r.numel() else ar

                lens = token_counts.detach().float()
                len_p50 = _safe_quantile(lens, 0.50, al)
                len_p90 = _safe_quantile(lens, 0.90, al)
                len_max = lens.max().item() if lens.numel() else al
                eos_rate = has_eos.float().mean().item() if has_eos is not None else 0.0

                # ===== KL/ratio/clip 状态 =====
                kl_div_tokens = (ref_per_token_logps - per_token_logps)
                kl_div_mean, kl_div_std, kl_div_p95, kl_div_max = _masked_stats(kl_div_tokens, completion_mask)
                ratio_mean, ratio_std, ratio_p95, ratio_max = _masked_stats(ratio, completion_mask)
                per_token_kl_mean, _, per_token_kl_p95, per_token_kl_max = _masked_stats(per_token_kl, completion_mask)

                masked = completion_mask.bool()
                if masked.any():
                    if args.loss_type == "cispo":
                        clip_frac = (ratio.detach()[masked] > args.epsilon_high).float().mean().item()
                    else:
                        clip_frac = ((ratio.detach()[masked] < (1 - args.epsilon)) | (ratio.detach()[masked] > (1 + args.epsilon))).float().mean().item()
                else:
                    clip_frac = 0.0

                # ===== 工具调用健康度 =====
                tool_metrics = _compute_tool_health_metrics(
                    turn_outputs_batch=turn_outputs_batch,
                    tools_batch=tools_batch,
                    gt_batch=gt_batch,
                    num_gen=args.num_generations,
                    unfinished_batch=unfinished_batch,
                )

                # ===== 数值健康：grad_norm / aux_loss / total_loss =====
                with torch.no_grad():
                    sq_sum = torch.tensor(0.0, device=args.device)
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        sq_sum += g.float().pow(2).sum()
                    grad_norm = torch.sqrt(sq_sum).item()

                total_loss = (policy_loss + aux_loss).detach().item() if isinstance(aux_loss, torch.Tensor) else pl
                aux_loss_v = aux_loss.detach().item() if isinstance(aux_loss, torch.Tensor) else 0.0

                wandb_payload = {
                    # 原有
                    "reward": ar,
                    "reward_wo_len": ar_wo_len,
                    "kl_ref": kl,
                    "group_reward_std": gs,
                    "advantages_std": ast,
                    "policy_loss": pl,
                    "avg_response_len": al,
                    "advantages_mean": am,
                    "learning_rate": lr,

                    # reward 分布
                    "reward/std": r_std,
                    "reward/p50": r_p50,
                    "reward/p90": r_p90,
                    "reward/min": r_min,
                    "reward/max": r_max,

                    # 长度与终止
                    "len/p50": len_p50,
                    "len/p90": len_p90,
                    "len/max": len_max,
                    "eos/rate": eos_rate,

                    # KL/ratio
                    "kl_div/mean": kl_div_mean,
                    "kl_div/std": kl_div_std,
                    "kl_div/p95": kl_div_p95,
                    "kl_div/max": kl_div_max,
                    "ratio/mean": ratio_mean,
                    "ratio/std": ratio_std,
                    "ratio/p95": ratio_p95,
                    "ratio/max": ratio_max,
                    "per_token_kl/mean": per_token_kl_mean,
                    "per_token_kl/p95": per_token_kl_p95,
                    "per_token_kl/max": per_token_kl_max,
                    "clip/frac": clip_frac,

                    # loss / grads
                    "loss/total": total_loss,
                    "loss/aux": aux_loss_v,
                    "grad/norm": grad_norm,
                }
                # 工具相关指标（打平到 wandb）
                for k, v in tool_metrics.items():
                    if k == "top_tool_counts":
                        continue
                    wandb_payload[f"tools/{k}"] = v

                # top 工具频次：展开成数值型动态 key，避免 string scalar 导致 swanlab 报错
                top_tool_counts = tool_metrics.get("top_tool_counts", {}) or {}
                for name, cnt in top_tool_counts.items():
                    # 例：tools/top/calculate_math = 12
                    wandb_payload[f"tools/top/{name}"] = float(cnt)

                # 显存（仅 cuda）
                if torch.cuda.is_available() and "cuda" in args.device:
                    wandb_payload["cuda/mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
                    wandb_payload["cuda/mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)

                wandb.log(wandb_payload)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            # ckpt 命名与 run_tag 对齐（与 pretrain/sft 一致）
            ckp = f'{args.save_dir}/{run_tag}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=run_tag,
                model=model,
                optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # DDP 对齐：保存仅主进程执行，但其它 rank 也必须等待，避免 rank 间步数/退出节奏不一致。
        # 否则可能出现：rank0 已经保存并进入退出/销毁 PG，其它 rank 仍在 rollout/forward，导致 torchrun 残留。
        if dist.is_initialized() and (step % args.save_interval == 0 or step == iters):
            dist.barrier()

        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument('--inference_rope_scaling', default=0, type=int, choices=[0, 1], help="rollout/训练时启用YaRN RoPE外推（需policy/ref一致；仅解决位置编码问题）")
    parser.add_argument('--yarn_target_len', default=5000, type=int, help="YaRN 外推目标长度（用于 factor 计算；同时也用于 RoPE buffer 预计算长度下界）")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    # ===== YaRN 配置（按：SFT 默认 768；Agentic RL 训练侧总长度上界 max_total_len） =====
    # - target_len：用于 RoPE buffer / 外推目标长度。至少覆盖 max_seq_len+max_gen_len，避免越界。
    # - orig_len：用 SFT 训练时的默认长度 768 作为“原始上下文长度”。
    cfg_kwargs = {}
    if bool(args.inference_rope_scaling):
        # target_len: YaRN 预期外推长度（用于 factor 计算）
        target_len = max(int(args.yarn_target_len), int(args.max_seq_len + args.max_gen_len))
        # RoPE buffer 预计算长度：固定给到 32768，避免 rollout/context 偶发变长导致越界
        rope_buf_len = 32768
        # YaRN 的 original_max_position_embeddings 更合理的来源：SFT 权重 tag 里的 -Sxxx
        # 解析不到就回退到默认 768（与 full_sft 默认一致）
        orig_len = _parse_sft_seq_len_from_weight_name(args.from_weight) or int(SFT_DEFAULT_SEQ_LEN)
        cfg_kwargs.update(
            {
                # 让 RoPE 预计算到 rope_buf_len，避免默认 32768 带来的额外显存/内存占用
                "max_position_embeddings": rope_buf_len,
                # 显式覆盖 rope_scaling（MiniMindConfig 支持 kwargs['rope_scaling'] 覆盖）
                "rope_scaling": {
                    "type": "yarn",
                    "factor": float(target_len) / float(orig_len),
                    "original_max_position_embeddings": orig_len,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "attention_factor": 1.0,
                },
            }
        )

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=bool(args.inference_rope_scaling),
        **cfg_kwargs,
    )

    # ========== run/ckpt 命名（与 pretrain/sft 类似，避免混淆） ==========
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    moe_experts = lm_config.num_experts if lm_config.use_moe else 0
    moe_topk = lm_config.num_experts_per_tok if lm_config.use_moe else 0
    if args.save_weight != 'agent' and ('MiniMind-' in args.save_weight):
        base_tag = args.save_weight
        run_tag = base_tag
    else:
        sft_seq_len = _parse_sft_seq_len_from_weight_name(args.from_weight)
        # 仅当：能解析到，且不是 SFT 默认长度时才加（默认长度不加，避免噪声）
        sft_len_tag = (
            f"-SFTS{sft_seq_len}"
            if (sft_seq_len and int(sft_seq_len) != int(SFT_DEFAULT_SEQ_LEN))
            else ""
        )
        base_tag = (
            f"MiniMind-Agent-RL-"
            f"DS{dataset_name}-"
            f"L{args.num_hidden_layers}-H{args.hidden_size}-S{args.max_seq_len}+{args.max_gen_len}"
            f"{sft_len_tag}"
            f"-MoE{moe_experts}K{moe_topk}-BS{args.batch_size}-GA{args.accumulation_steps}"
            f"-LR{args.learning_rate}-Ep{args.epochs}-G{args.num_generations}"
            f"-B{args.beta}-T{args.loss_type}"
        )
        run_tag = base_tag

    # ========== 检查/加载断点（仅当 from_resume==1；兼容新/旧命名） ==========
    ckp_data = None
    if args.from_resume == 1:
        # 优先按当前 run_tag 恢复（新命名）；找不到则回退到 base_tag / args.save_weight（兼容旧用法）
        ckp_data = lm_checkpoint(lm_config, weight=run_tag, save_dir='../checkpoints')
        if ckp_data is None:
            ckp_data = lm_checkpoint(lm_config, weight=base_tag, save_dir='../checkpoints')
        if ckp_data is None:
            ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 补齐参数量到 run_tag
    total_params_m, active_params_m = get_model_params(model, lm_config, log=False)
    if run_tag == base_tag and not (args.save_weight != 'agent' and ('MiniMind-' in args.save_weight)):
        fmt = lambda x: (f"{x:.1f}".replace('.', 'p'))
        run_tag = f"{base_tag}-P{fmt(total_params_m)}M-A{fmt(active_params_m)}M"

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=run_tag, id=wandb_id, resume=resume)

    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

        # epoch 级别对齐：确保所有 rank 都结束 epoch，再进入下一轮/销毁进程组
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        # 再次对齐，避免有的 rank 先 destroy 导致其它 rank 后续 NCCL/barrier 异常
        dist.barrier()
        dist.destroy_process_group()
