import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
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
    """从 ckpt 命名里解析模型结构参数（与 eval_llm.py 对齐）。"""
    out = {}
    m = re.search(r"-L(?P<L>\d+)-H(?P<H>\d+)(?:-S(?P<S>\d+))?", tag)
    if m:
        out["num_hidden_layers"] = int(m.group("L"))
        out["hidden_size"] = int(m.group("H"))

    m = re.search(r"-MoE(?P<E>\d+)K(?P<K>\d+)", tag)
    if m:
        e = int(m.group("E"))
        k = int(m.group("K"))
        out["use_moe"] = 1 if e > 0 else 0
        if e > 0:
            out["num_experts"] = e
            out["num_experts_per_tok"] = k
    return out


def resolve_ckpt_path(args) -> str:
    """兼容 ckpt 命名：

    1) 直接传 /path/to/xxx.pth
    2) 新命名：save_dir/<run_tag>.pth
    3) 旧命名：save_dir/<weight>_<hidden_size>[_moe].pth
    """
    w = args.weight
    if isinstance(w, str) and (w.endswith('.pth') or '/' in w or w.startswith('.')):
        return w

    # 优先新命名：<tag>.pth
    cand_new = os.path.join(args.save_dir, f"{w}.pth")
    if os.path.exists(cand_new):
        return cand_new

    # 兼容旧命名：<weight>_<hidden_size>[_moe].pth
    moe_suffix = '_moe' if bool(getattr(args, 'use_moe', 0)) else ''
    cand_old = os.path.join(args.save_dir, f"{w}_{getattr(args, 'hidden_size', 768)}{moe_suffix}.pth")
    if os.path.exists(cand_old):
        return cand_old

    # 找不到就返回新命名候选，让后续报错信息更直观
    return cand_new


def infer_args_from_ckpt_name(args):
    """从 ckpt 文件名推断一些关键配置（若命名包含则覆盖 args）。"""
    ckpt_path = resolve_ckpt_path(args)
    tag = _strip_ckpt_suffix(ckpt_path)
    args.ckpt_tag = tag
    parsed = parse_ckpt_tag(tag)

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

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        cfg_kwargs = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "use_moe": bool(args.use_moe),
        }
        if bool(args.use_moe):
            cfg_kwargs["num_experts"] = getattr(args, "num_experts", 4)
            cfg_kwargs["num_experts_per_tok"] = getattr(args, "num_experts_per_tok", 1)
        model = MiniMindForCausalLM(MiniMindConfig(**cfg_kwargs))

        ckp = getattr(args, "ckpt_path", None) or resolve_ckpt_path(args)
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            obj = json.loads(m.strip())
            # 兼容输出为 list 的情况
            if isinstance(obj, list):
                calls.extend([x for x in obj if isinstance(x, dict)])
            elif isinstance(obj, dict):
                calls.append(obj)
        except Exception:
            pass
    return calls


def normalize_local_tool_call(call: dict, idx: int = 0) -> dict:
    """把本地模型输出的 tool_call 统一成脚本内部结构。

    统一为：{"id": "call_x", "name": "xxx", "arguments": <dict_or_json_str>}

    兼容常见变体：
    - {"name": "xxx", "arguments": {...}}
    - {"name": "xxx", "args": {...}}
    - {"name": "xxx", "arg": {...}}
    """
    if not isinstance(call, dict):
        return {"id": f"call_{idx}", "name": "", "arguments": {}}
    name = call.get("name", "")
    raw_args = call.get("arguments", None)
    if raw_args is None:
        raw_args = call.get("args", None)
    if raw_args is None:
        raw_args = call.get("arg", None)
    if raw_args is None:
        raw_args = {}
    return {"id": f"call_{idx}", "name": name, "arguments": raw_args}



def parse_tool_call_from_text(content):
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        if isinstance(call, dict):
            raw_args = call.get("arguments", None)
            if raw_args is None:
                raw_args = call.get("args", None)
            if raw_args is None:
                raw_args = call.get("arg", {})
        else:
            raw_args = arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = [normalize_local_tool_call(tc, i) for i, tc in enumerate(parse_tool_calls(content))]
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc.get("arguments", {})
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall评测")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端（local=本地模型，api=OpenAI兼容接口）")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='../out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重tag或路径：支持传 run_tag（不带.pth）或直接传 /path/to/xxx.pth")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--num_experts', default=4, type=int, help="MoE专家数（可从ckpt命名中自动解析覆盖）")
    parser.add_argument('--num_experts_per_tok', default=1, type=int, help="MoE TopK（可从ckpt命名中自动解析覆盖）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--show_speed', default=0, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI兼容接口的base_url")
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI兼容接口的api_key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API请求时使用的模型名称")
    parser.add_argument('--stream', default=1, type=int, help="API模式下是否流式输出（0=否，1=是）")
    args = parser.parse_args()

    # 适配新 ckpt 命名：从 ckpt tag 自动推断 hidden_size/layers/moe 等
    args.ckpt_path = infer_args_from_ckpt_name(args)

    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    if input_mode == 0:
        cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES]
    else:
        # 手动模式：不做工具挑选，直接与自动测试对齐——固定使用自动测试第 1 条用例的工具集合。
        # 这样你手动输入同类问题（例如算术）时，行为与自动测试一致。
        base_tool_names = TEST_CASES[0]["tools"] if TEST_CASES else [t["function"]["name"] for t in TOOLS]
        base_tools = get_tools(base_tool_names) if TEST_CASES else TOOLS

        def _next_manual_case():
            prompt = input('💬: ')
            if not prompt:
                return {"prompt": "", "tools": [], "tool_names": []}
            return {"prompt": prompt, "tools": base_tools, "tool_names": base_tool_names}

        cases = iter(_next_manual_case, {"prompt": "", "tools": [], "tool_names": []})
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        # 对齐展示：两种模式都打印本轮工具列表
        print(f'📦 可用工具: {case["tool_names"]}\n')
        if input_mode == 0:
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
