# MiniMind Datasets

本目录包含 MiniMind 训练/对齐所用的示例数据集（均为 `.jsonl`）。

> 说明
> - 统计口径使用仓库自带 tokenizer：`AutoTokenizer.from_pretrained(model/)`。
> - 预训练数据（`pretrain_t2t` / `seq_monkey`）按 `PretrainDataset` 的口径计 token：
>   `raw_len = tokenize(text, add_special_tokens=False) + 2(bos/eos)`，训练时再按 `max_seq_len` 截断。
> - SFT 数据（`sft_t2t`）按 `apply_chat_template` 拼接后计 token，并按 `max_seq_len` 截断。
> - Agent RL 数据（`agent_rl`）为在线 rollout 训练；此处统计的是「prompt（含 generation prompt）」的 token 长度。
> - 大文件无法全量逐条 tokenize：token 总量与分布为抽样估算（agent_rl 为全量精确统计）。

## 文件一览

- `pretrain_t2t.jsonl`：预训练文本（字段：`{"text": "..."}`）
- `seq_monkey.jsonl`：预训练长文本（字段：`{"text": "..."}`，长样本占比更高）
- `sft_t2t.jsonl`：监督微调对话（字段：`{"conversations": [...]}`）
- `agent_rl.jsonl`：Agent RL 多轮工具调用/对话数据（字段：`{"conversations": [...], "gt": [...]}`）

## Token 规模与长度分布（概览）

下表中的：
- **raw tokens**：按各自数据构造方式，未做长度 cap 前的 token 数（含必要的 BOS/EOS 或 chat template 特殊 token）。
- **有效 tokens（cap 后）**：按训练默认 cap 截断后的 token 数（不含 padding）。
- **padding 后 tokens**：训练中张量 `input_ids.numel()` 的量级（包含 padding，近似为 `N * cap_len`）。

| 数据集 | 行数 N | cap_len | 估算 raw tokens 总量 | 估算 有效 tokens 总量（cap 后） | padding 后 tokens（N*cap_len） |
|---|---:|---:|---:|---:|---:|
| pretrain_t2t | 8,468,827 | 340 | ~1.884B | ~1.544B | ~2.879B |
| seq_monkey | 13,000,000 | 340 | ~4.267B | ~2.674B | ~4.420B |
| sft_t2t | 5,109,432 | 768 | ~3.508B | ~2.637B | ~3.924B |
| agent_rl（prompt） | 39,988 | 2500 | 23.93M | 23.93M | - |

### pretrain_t2t（cap=340）

- 平均 raw_len：~222.50；平均 cap 后有效 len：~182.33
- 截断率（raw_len > 340）：~13.78%
- 分位数（raw / cap 后）：
  - p50：166 / 166
  - p90：373 / 340
  - p95：488 / 340
  - p99：1402 / 340

### seq_monkey（cap=340）

- 平均 raw_len：~328.23；平均 cap 后有效 len：~205.67
- 截断率：~34.50%
- 分位数（raw / cap 后）：
  - p50：220 / 220
  - p90：731 / 340
  - p95：996 / 340
  - p99：1843 / 340

### sft_t2t（cap=768）

> 备注：训练中会以较高概率移除空的 `<think>\n\n</think>\n\n` 块；这里的统计按“移除空 think 块”的口径，更贴近大多数训练步。

- 平均 raw_len：~686.50；平均 cap 后有效 len：~516.13
- 截断率：~26.33%
- 分位数（raw / cap 后）：
  - p50：523 / 517
  - p90：1301 / 768
  - p95：1604 / 768
  - p99：3106 / 768

### agent_rl（prompt，cap=2500，全量精确统计）

- 平均 prompt_len：~598.40
- 截断率（prompt_len > 2500）：~0.0025%（几乎没有）
- 分位数：
  - p50：535
  - p90：1172
  - p95：1335
  - p99：1606
- 含 tools 的 prompt 占比：~50.02%
