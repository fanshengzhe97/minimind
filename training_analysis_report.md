# MiniMind 训练阶段分析报告

> 生成日期：2026-05-30

---

## 总览

MiniMind 项目目前包含 **8 个训练阶段**（另有一个可选的 Tokenizer 训练脚本，仅为学习参考）。下表快速汇总各阶段：

| 阶段 | 脚本 | 训练方法 | 数据格式 | 核心损失 | 输入-标签对齐方式 |
|------|------|----------|----------|----------|-------------------|
| ① 预训练 | `trainer/train_pretrain.py` | 自回归语言建模 | 纯文本 JSONL | 所有 token 的 CE | 模型内部 shift：`logits[:, :-1]` vs `labels[:, 1:]` |
| ② 全量 SFT | `trainer/train_full_sft.py` | 有监督微调 | 对话 JSONL | 仅回答部分的 CE | 模型内部 shift：`logits[:, :-1]` vs `labels[:, 1:]` |
| ③ LoRA SFT | `trainer/train_lora.py` | 参数高效微调 | 对话 JSONL | 仅回答部分的 CE（仅更新 LoRA） | 模型内部 shift：`logits[:, :-1]` vs `labels[:, 1:]` |
| ④ 知识蒸馏 | `trainer/train_distillation.py` | 学生模仿教师 | 对话 JSONL | CE + KL 散度 | 代码手动 shift：`logits[..., :-1, :]` vs `labels[..., 1:]` |
| ⑤ DPO | `trainer/train_dpo.py` | 直接偏好优化 | chosen/rejected 对 | DPO loss | **数据层提前 shift**：`x = input_ids[:-1]`，`y = input_ids[1:]` |
| ⑥ GRPO | `trainer/train_grpo.py` | 组相对策略优化 | 对话 JSONL | 策略梯度 + KL | 代码手动 shift：`logits[:, :-1]` vs `outputs[:, 1:]` |
| ⑦ PPO | `trainer/train_ppo.py` | 近端策略优化 | 对话 JSONL | PPO Actor + Critic loss | 代码手动 shift：`logits[:, :-1]` vs `labels[:, 1:]` |
| ⑧ Agent RL | `trainer/train_agent.py` | Agent 强化学习 | 含工具的对话 JSONL | 策略梯度 + KL | 代码手动 shift：`logits[:, :-1]` vs `input_ids[:, 1:]` |

---

## 数据样例：从原始 JSONL 到模型输入

为了更直观地理解每个阶段的数据变换过程，下面用具体的数值例子来展示。

### 预训练样例

**原始数据**（`pretrain_t2t_mini.jsonl` 中的一行）：
```json
{"text": "人工智能是计算机科学的一个分支"}
```

**经过 `PretrainDataset.__getitem__` 处理后，送入模型的 input_ids 和 labels**（假设 `max_length=8`，实际会大得多）：

```
Tokenizer 词表: 6400, BOS=1, EOS=2, PAD=0

编码后 tokens:      [1, 45, 123, 67, 89, 234, 56, 78, 12, 2]
加 BOS/EOS:         [1, 1, 45, 123, 67, 89, 234, 56, 78, 12, 2]
padding(→8):        [1, 1, 45, 123, 67, 89, 234, 2, 0, 0]
                     ↑BOS                    ↑EOS  ↑PAD

input_ids: [1, 1, 45, 123, 67, 89, 234, 2, 0, 0]
labels:    [1, 1, 45, 123, 67, 89, 234, 2, -100, -100]
                                              ↑EOS   ↑PAD被忽略
```

**模型收到的信号**：每个位置（除最后一个）预测下一个 token，所有非 padding 位置都参与 loss。

**输入-标签对齐**：模型内部自动 shift
```
模型输入:    [1, 1, 45, 123, 67, 89, 234, 2, 0, 0]      ← input_ids (完整)
             │
模型内部:    logits[:, :-1]  →  [BOS, t1, t2, t3, t4, t5, t6, EOS, PAD]
             labels[:, 1:]   →  [t1,  t2, t3, t4, t5, t6, EOS, PAD, PAD]
                                ↑ 参与 loss          ↑ 参与 loss  ↑ 被 -100 忽略
```


---

### SFT 样例

**原始数据**（`sft_t2t_mini.jsonl` 中的一行）：
```json
{
  "conversations": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是让计算机从数据中学习的技术。"}
  ]
}
```

**经 `apply_chat_template` 渲染后的文本**：
```
<|im_start|>user\n什么是机器学习？<|im_end|>\n<|im_start|>assistant\n机器学习是让计算机从数据中学习的技术。<|im_end|>\n
```

**Tokenizer 编码后**（简化示意，实际 token ID 因词表而异）：
```
token序列:
  <|im_start|>  user  \n  什么是机器学习？  <|im_end|>  \n  <|im_start|>  assistant  \n  机器学习是...  <|im_end|>  \n
```

**SFTDataset.generate_labels 处理后**：
```
input_ids: [tok_im_start, tok_user, tok_n, tok_question..., tok_im_end, tok_n,
            tok_im_start, tok_assistant, tok_n, tok_answer...,   tok_im_end, tok_n,  0, 0, ...]
labels:    [-100,        -100,    -100,  -100,           -100,      -100,
            -100,        -100,      tok_answer...,        tok_im_end, tok_n, -100, -100, ...]
                                                                       ↑
                                                             仅这部分参与 loss
```

> 与预训练的关键区别：user 输入部分的预测结果被 `-100` 忽略，模型只学习如何正确生成 assistant 的回答。

**输入-标签对齐**：与预训练相同，模型内部自动 shift
```
input_ids: [im_start, user, \n, 问题..., im_end, \n, im_start, assistant, \n, 回答..., im_end, \n, PAD, PAD]
labels:    [-100,    -100, -100, -100, -100, -100, -100,  -100,     -100, 回答..., im_end, \n, -100, -100]
             ↑ user 部分全部是 -100，不参与 loss                ↑ 只有回答部分保留原始 ID，参与 loss

模型内部: logits[:, :-1] 预测所有位置的下一个 token
         labels[:, 1:]    只有回答部分有真实 ID，其余都是 -100 被忽略
```


---

### DPO 样例

**原始数据**（`dpo.jsonl` 中的一行）：
```json
{
  "chosen": [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "1+1=2"}
  ],
  "rejected": [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "1+1=3"}
  ]
}
```

**DPODataset 的数据处理与 SFT 类似，同样经过 chat template 渲染**（代码中确有这部分逻辑）：

```python
# DPODataset.__getitem__()
chosen_prompt = self.tokenizer.apply_chat_template(
    chosen, tokenize=False, add_generation_prompt=False
)  # → 渲染为纯文本字符串
chosen_prompt = post_processing_chat(chosen_prompt)   # 80%概率移除空think
chosen_encoding = self.tokenizer(chosen_prompt, ...)  # 编码为 input_ids
```

**渲染后的文本**（与 SFT 完全一致的格式）：
```
<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
1+1=2<|im_end|>
```

**DPODataset 处理后**（简化示意）：
```
chosen_input_ids:  [tok_user, tok_question..., tok_im_end, tok_assistant, tok_1+1=2, tok_im_end, 0, 0, ...]
chosen_loss_mask:  [0,       0, ...,      0,         0,            1,          1,        0, 0, ...]
                    ↑user部分=0                     ↑assistant回答=1             ↑pad=0

x_chosen = chosen_input_ids[:-1]   → 模型输入
y_chosen = chosen_input_ids[1:]    → 标签（用于 gather 提取 log prob）
mask_chosen = chosen_loss_mask[1:] → 仅 mask=1 的位置参与 DPO loss

  chosen 和 rejected 各占 batch 的一半，拼接后一起前向：
  前一半 (B/2): chosen 的 x/y/mask
  后一半 (B/2): rejected 的 x/y/mask
```

模型对 chosen 和 rejected 的 log_probs 只在 mask=1（assistant 回答）的位置求和，然后计算 DPO 损失让 chosen 的概率 > rejected 的概率。

**输入-标签对齐**：⚠️ **DPO 是唯一在数据层就 shift 的阶段**
```
数据层提前拆开:
  chosen_input_ids(完整): [tok_user, tok_question, tok_im_end, tok_assistant, tok_answer, tok_im_end, PAD, PAD]

  x_chosen = [:-1]       → 传给模型: [tok_user, tok_question, tok_im_end, tok_assistant, tok_answer, tok_im_end, PAD]
  y_chosen = [1:]        → 用于 gather: [tok_question, tok_im_end, tok_assistant, tok_answer, tok_im_end, PAD, PAD]
  mask_chosen = mask[1:] → 用于过滤:   [0,          0,           0,            1,           1,        0,    0]
                                           ↑ 模型 logits 输出 shape = (B, L-1, V)，和 y 天然对齐！
```


---

### GRPO/PPO/Agent RL 样例

**原始数据**（`rlaif.jsonl` 中的一行）：
```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有用的AI。"},
    {"role": "user", "content": "请用一句话描述春天"}
  ]
}
```

**RLAIFDataset 处理后——只保留 prompt 部分**：
```
渲染后文本: "<|im_start|>system\n你是一个有用的AI。<|im_end|>\n
             <|im_start|>user\n请用一句话描述春天<|im_end|>\n
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
             ↑ 这里没有回答！模型需要在训练中自行生成
```

**训练过程中的 Rollout 产出**（不是预先准备的）：
```
prompt token IDs:  [BOS, sys_tokens..., user_tokens..., assistant_header...]
                                                         ↑ 模型从此处开始生成
生成 completion:    "春天是万物复苏的季节。"  → token IDs: [102, 345, 678, ..., 2(EOS)]
completion_mask:    [1, 1, 1, ..., 0]  → 标记哪些位置是模型生成的
                                                    ↑ EOS 之后设为 0
```

**Loss 只算 completion 部分**，prompt 部分始终不参与 loss。

**输入-标签对齐**：手动 shift + `logp_pos` 定位 completion

```
传给模型: outputs = [p0, p1, p2, | g0, g1, g2, g3, g4, EOS, PAD, PAD]
                     ↑prompt     ↑completion (= prompt_lens)

代码手动:
  logits[:, :-1, :]  →  [p0, p1, p2, | g0, g1, g2, g3, g4, EOS, PAD]
  outputs[:, 1:]      →  [p1, p2, g0, | g1, g2, g3, g4, EOS, PAD, PAD]
                                           ↑
                                   logp_pos 只定位到这里
                                   (prompt_lens-1 + [0..R-1])

  然后 per_token_logps = log_probs.gather(1, logp_pos)
  → 只取 completion tokens [g1, g2, g3, g4, EOS] 的 log_prob
  → 这些用于策略梯度 loss 和 KL 散度计算
```

> `outputs[:, 1:]` 就是每个位置"应该预测的下一个 token"的 ground truth。把它传给 `gather()`，从模型预测的分布 `logits[:, :-1]` 中取出真实 token 的 log probability。之后再通过 `logp_pos` 只保留 completion 部分的索引，丢弃 prompt 部分。


---

## 阶段 ①：预训练（Pretrain）

**脚本**：`trainer/train_pretrain.py`

### 原始数据格式

JSONL 文件，每行一个 JSON 对象，包含 `{"text": "..."}` 字段。例如 `pretrain_t2t_mini.jsonl`、`seq_monkey.jsonl`。

```json
{"text": "机器学习是人工智能的一个分支..."}
```

### 数据处理流程

数据加载类：`PretrainDataset`（定义在 `dataset/lm_dataset.py`）

```
原始 JSONL           tokenizer 编码        加 BOS/EOS              padding 到固定长度
─────►  {"text":...}   ─────►  [t1,t2,...]   ─────►  [BOS,t1,t2,...,EOS]   ─────►  input_ids
                                                      labels = input_ids.clone()
                                                      labels[PAD 位置] = -100
```

关键代码逻辑：

```python
sample = self.samples[index]                     # 读取 {"text": "..."}
tokens = tokenizer(str(sample['text']),
                   add_special_tokens=False).input_ids
tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
input_ids = tokens + [pad_token] * (max_length - len(tokens))
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
```

### 损失计算

```
model(input_ids, labels=labels)
  └── MiniMindForCausalLM.forward()
       └── F.cross_entropy(logits[:, :-1, :],
                            labels[:, 1:],
                            ignore_index=-100)
```

| 项目 | 说明 |
|------|------|
| **算 loss 的部分** | 所有非 padding 的 token（BOS、正文文本、EOS） |
| **不算 loss 的部分** | padding token（被 `ignore_index=-100` 忽略） |
| **额外损失** | 若启用 MoE，加上路由器负载均衡损失 `aux_loss` |

**总损失**：`loss = logits_loss + aux_loss`

> 本质是"预测下一个 token"的自回归语言建模任务，序列中每个位置（除最后一个）都预测下一个位置的 token。

#### 📌 输入-标签对齐方式

```
传给模型: model(input_ids=full_seq, labels=full_seq)
                 │                      │
模型内部:        logits[:, :-1, :]       labels[:, 1:]
                 (预测位置 0..n-2)        (目标位置 1..n-1)
```

**关键点**：模型在 `forward()` 内部自动做了 shift。传给模型的是**完整序列**，模型拿出 `logits[:, :-1]` 和 `labels[:, 1:]` 计算 CE loss。

```python
# MiniMindForCausalLM.forward()
x = logits[..., :-1, :].contiguous()     # (B, L-1, V)
y = labels[..., 1:].contiguous()          # (B, L-1)
loss = F.cross_entropy(x.view(-1, V), y.view(-1), ignore_index=-100)
```

| 位置 | `logits` 预测 | `labels` 目标 |
|------|---------------|---------------|
| 0 | token₁ | token₁（BOS） |
| 1 | token₂ | ... |
| ... | ... | ... |
| L-2 | token_{L-1}（EOS） | token_{L-1}（EOS） |
| L-1（最后一个） | token_L（PAD） | **被丢弃** |

> 直观理解：序列 `[BOS, 人, 工, 智, 能, EOS, PAD]`，模型用 `[BOS, 人, 工, 智, 能, EOS]` 预测 `[人, 工, 智, 能, EOS, PAD]`，但 PAD 被 `ignore_index=-100` 忽略。

---

## 阶段 ②：全量 SFT（Supervised Fine-Tuning）

**脚本**：`trainer/train_full_sft.py`

### 原始数据格式

JSONL 文件，每行包含 `{"conversations": [...]}`，conversations 是多轮对话列表。

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个知识丰富的AI，尽力为用户提供准确的信息。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进..."}
  ]
}
```

### 数据处理流程

数据加载类：`SFTDataset`（定义在 `dataset/lm_dataset.py`）

```
原始 JSONL
─────►  conversations
        │
        ├── pre_processing_chat()                        ← 概率性处理①
        │   └── 以 20% 概率随机添加一个 system prompt（详见后文解释）
        │       工具数据（含 tools 字段）不做任何处理
        │
        ├── create_chat_prompt()
        │   └── tokenizer.apply_chat_template(conversations)
        │       渲染为带特殊标记的文本字符串：
        │       "<|im_start|>system\n...<|im_end|>\n
        │        <|im_start|>user\n...<|im_end|>\n
        │        <|im_start|>assistant\n...<|im_end|>\n"
        │
        ├── post_processing_chat()                       ← 概率性处理②
        │   └── 以 80% 概率移除空的 <think>\n\n</think>\n\n 标签（详见后文解释）
        │
        ├── tokenizer(prompt)
        │   └── 编码为 input_ids，padding 到 max_length
        │
        └── generate_labels(input_ids)    ← 关键：标记哪些 token 算 loss
            └── 搜索 "<|im_start|>assistant\n" 的位置
                仅该位置之后的 token 保留原始 ID（参与 loss 计算）
                其余 token 全部设为 -100（被忽略）
```

`generate_labels` 的核心逻辑：

```python
labels = [-100] * len(input_ids)
while i < len(input_ids):
    if input_ids[i:i + len(bos_id)] == bos_id:  # 找到 "<|im_start|>assistant\n"
        start = i + len(bos_id)
        end = start
        while end < len(input_ids):
            if input_ids[end:end + len(eos_id)] == eos_id:  # 找到 "<|im_end|\n>"
                break
            end += 1
        for j in range(start, min(end + len(eos_id), max_length)):
            labels[j] = input_ids[j]  # 仅 assistant 回答参与 loss
        i = end + len(eos_id)
    else:
        i += 1
```

### 损失计算

```
model(input_ids, labels=labels)
  └── F.cross_entropy(logits[:, :-1, :],
                       labels[:, 1:],
                       ignore_index=-100)
```

| 项目 | 说明 |
|------|------|
| **算 loss 的部分** | 仅 `assistant` 角色的回复内容 token（含回答中的 `<think>` 等） |
| **不算 loss 的部分** | `system` 和 `user` 的输入、特殊标记（`<|im_start|>`、`<|im_end|>`）、padding token |
| **额外损失** | MoE aux_loss |

> 这是有监督微调的标准做法：让模型学会生成正确的回答，同时不惩罚它对用户输入部分的预测。

#### 📌 输入-标签对齐方式

与预训练**完全相同**——传给模型完整序列，模型内部自动 shift。

```
传给模型: model(input_ids=full_seq, labels=full_seq)
                         │                      │
模型内部:                logits[:, :-1, :]       labels[:, 1:]
```

```python
# MiniMindForCausalLM.forward() — 与预训练同一段代码
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, V), y.view(-1), ignore_index=-100)
```

**与预训练的唯一区别**：`labels` 中除了 padding 位置的 `-100` 外，user/system 输入部分也被设为 `-100`，因此模型不会学习预测 user 输入的内容。

---

## 阶段 ③：LoRA SFT

**脚本**：`trainer/train_lora.py`

### 原始数据格式

与阶段②完全相同，默认数据 `lora_medical.jsonl`。

### 数据处理流程

与阶段②完全一致，使用同一个 `SFTDataset` 类。

### LoRA 参数的具体位置

`apply_lora()` 函数遍历模型的**所有模块**，对满足以下条件的 nn.Linear 层添加 LoRA：

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 给这个 Linear 层挂上一个 LoRA 分支
            setattr(module, "lora", lora)
```

**条件**：`nn.Linear` 且 `in_features == out_features`（方阵）。

以一个 8 层、hidden_size=768 的 MiniMind 为例，以下 Linear 层会被添加 LoRA：

| 模块路径 | 所在位置 | shape | 是否方阵 | 加 LoRA？ |
|----------|----------|-------|---------|----------|
| `model.embed_tokens` | Embedding（不是 Linear） | — | — | ❌ |
| `model.layers.0.self_attn.q_proj` | Attention Q 投影 | 768→768 | ✅ | ✅ |
| `model.layers.0.self_attn.k_proj` | Attention K 投影 | 768→384 | ❌ | ❌ |
| `model.layers.0.self_attn.v_proj` | Attention V 投影 | 768→384 | ❌ | ❌ |
| `model.layers.0.self_attn.o_proj` | Attention O 投影 | 768→768 | ✅ | ✅ |
| `model.layers.0.mlp.gate_proj` | FFN 门控投影 | 768→1536 | ❌ | ❌ |
| `model.layers.0.mlp.up_proj` | FFN 上投影 | 768→1536 | ❌ | ❌ |
| `model.layers.0.mlp.down_proj` | FFN 下投影 | 1536→768 | ❌ | ❌ |
| `model.layers.1...` | 同上（每层都一样） | ... | ... | ... |
| `model.norm` | RMSNorm（不是 Linear） | — | — | ❌ |
| `lm_head` | 语言模型头 | 768→6400 | ❌ | ❌ |

**实际被加 LoRA 的层**：每个 Transformer 层的 `self_attn.q_proj` 和 `self_attn.o_proj`（仅这 2 个是方阵）。

对于 8 层模型：`8 层 × 2 个方阵 Linear = 16 个 LoRA 分支`，每个分支的参数量为 `rank × (in + out) = 16 × (768 + 768) = 24,576`，共 `16 × 24,576 ≈ 0.39M` 参数，约占总参数量（~42M）的 **0.93%**。

### 学习率变化

| 参数 | 全量 SFT | LoRA SFT |
|------|----------|----------|
| 默认学习率 | `1e-5` | **`1e-4`**（比全量 SFT 大 10 倍） |
| Epochs | 2 | **10**（比全量 SFT 多 5 倍） |
| Batch size | 16 | 32 |
| 优化器 | `AdamW(model.parameters())` | `AdamW(lora_params)` |

LoRA 学习率比全量 SFT 大 10 倍（`1e-4` vs `1e-5`），原因是：
- LoRA 只更新极少量参数（~1%），梯度信号较弱，需要更大的学习率才能有效更新
- LoRA 的 A 矩阵初始化为高斯分布（`std=0.02`），B 矩阵初始化为 0，初始时 LoRA 分支的输出为 0，**不影响原始模型输出**。大学习率让 LoRA 参数快速从零开始学习有效方向

### 核心区别：参数更新范围

| 项目 | 全量 SFT | LoRA SFT |
|------|----------|----------|
| 更新参数 | 全部模型参数 | 仅 LoRA 低秩矩阵（A、B） |
| 冻结参数 | 无 | 主干模型全部参数 |
| 梯度裁剪对象 | `model.parameters()` | `lora_params` |
| 优化器管理 | 全部参数 | 仅 `lora_params` |
| 参数量占比 | 100% | 通常 ~1%（rank=16 时） |

### 损失计算

与阶段②**完全相同**（Assistant 回答的 CE + aux_loss），只是梯度只回传到 LoRA 参数。

```
loss = res.loss + res.aux_loss    ← 同全量 SFT
loss.backward()                    ← 梯度仅更新 requires_grad=True 的 LoRA 参数
```

#### 📌 输入-标签对齐方式

与 SFT **完全相同**，模型内部 shift。`input_ids` 存完整序列，`labels` 存完整序列（非 assistant 回答部分为 `-100`）。

---

## 阶段 ④：知识蒸馏（Distillation）

**脚本**：`trainer/train_distillation.py`

### 原始数据格式

与 SFT 相同（使用 `SFTDataset`），默认数据 `sft_t2t_mini.jsonl`。

### 数据处理流程

与阶段②完全一致。

### 模型架构

```
学生模型（student）← 训练，更新参数
教师模型（teacher）← frozen，evaluation 模式
```

两模型可以有不同的 hidden_size、num_layers、是否 MoE。

### 损失计算

蒸馏使用了**两条不同的监督信号**，加权求和：

```
total_loss = alpha * ce_loss + (1-alpha) * distill_loss + aux_loss
```

> ⚠️ **两条 loss 的分工不同，不要混淆：**

| Loss | 输入 | 目标 | 本质 |
|------|------|------|------|
| **① CE Loss** | 学生 logits | 数据集的**真实标注 token ID**（hard label / one-hot） | 标准 SFT：学会预测正确下一个词 |
| **② Distill Loss** | 学生 log_probs | 教师模型的 **softmax 概率分布**（soft target，是向量不是 one-hot） | 知识蒸馏：模仿教师在整个词表上的分布 |

- **CE Loss**：学生 vs **数据集的标准答案**（hard label）。例如数据中标注的回答是 `"机器学习是..."`，CE Loss 就让学生去拟合这个真实文本。这是**普通的监督学习**。
- **Distill Loss**：学生 vs **教师的输出分布**（soft target）。教师的 softmax 经过温度软化后给出整个词表的概率（如"狗0.6, 狼0.3, 猫0.05..."），让学生去匹配这个分布，从而学到教师对词表相似性的**暗知识**。

当 `alpha=0.5`（默认）时两者等权；`alpha=1.0` 退化为纯 SFT；`alpha=0.0` 为纯蒸馏。

#### ① Ground-Truth CE Loss

与阶段②类似但也有细微差异，蒸馏的 CE 损失计算如下：

```python
loss_mask = (labels[..., 1:] != -100).float()        # (B, L-1), 1=assistant回答
shift_labels = labels[..., 1:].contiguous()           # (B, L-1)

ce_loss = F.cross_entropy(
    student_logits.view(-1, vocab_size),              # (B*(L-1), V)
    shift_labels.view(-1),                            # (B*(L-1))
    ignore_index=-100,                                # label=-100 的位置 loss 输出为 0
    reduction='none'                                  # 返回逐 token 的 loss
)                                                     # → (B*(L-1))

loss_mask_flat = loss_mask.view(-1)                   # → (B*(L-1))

ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
```

**关键细节分析：**

| 步骤 | 作用 |
|------|------|
| `labels[..., 1:]` | 手动 shift，得到 `(B, L-1)` 的目标标签 |
| `loss_mask = (labels[..., 1:] != -100).float()` | 创建 0/1 mask，`1`=assistant 回答 token，`0`=user/system/padding |
| `F.cross_entropy(ignore_index=-100, reduction='none')` | 计算逐 token CE，label=-100 的位置 loss 输出为 **0** |
| `ce_loss * loss_mask_flat` | 再乘以 loss_mask，**双重保险**确保非回答位置被清零 |
| `sum / mask.sum()` | 仅在 assistant 回答 token 上求平均 |

**为什么有了 `ignore_index=-100` 还要乘 `loss_mask`？**

这两者并不完全等价：

- `loss_mask = (labels[..., 1:] != -100).float()` mask 的是 `shift_labels`（即 `labels[:, 1:]`）
- 但 `F.cross_entropy` 的 `ignore_index=-100` 检查的是 `shift_labels.view(-1)` 中值为 `-100` 的位置

由于 SFTDataset 的 `generate_labels` 已经把非 assistant 部分全部设为 `-100`，所以在这里 `loss_mask_flat` 为 0 的位置恰好也是 `shift_labels` 中为 `-100` 的位置。**在蒸馏代码中这个乘法是冗余的**（`cross_entropy` 已经输出了 0），但它让代码意图更明确：**明确地只对 assistant 回答计算 CE loss，并将其作为分母计算平均值**，而不是依赖 `cross_entropy` 内部的 `ignore_index` 计数。

**与 SFT（阶段②）的对比：**

| 方面 | 全量 SFT | 蒸馏中的 CE |
|------|----------|-------------|
| 输入 | `model(input_ids, labels=labels)` | `model(input_ids)` 不传 labels |
| shift 方式 | 模型内部自动 shift | 代码手动 `student_logits[..., :-1]` vs `labels[..., 1:]` |
| 忽略位置 | `ignore_index=-100` | `ignore_index=-100` + 额外 `* loss_mask` |
| 取平均方式 | `cross_entropy` 内部自动平均（reduction='mean' 隐式） | 手动 `sum / mask.sum()` |
| 附带 loss | 无（仅 MoE 的 aux_loss） | 无（仅 MoE 的 aux_loss） |

#### ② Distillation Loss（KL 散度）

```python
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
distill_loss = temperature² * KL(teacher_probs || student_log_probs)
```

- 只在 `loss_mask == 1` 的位置（assistant 回答）计算
- 教师 logits 裁剪到学生 vocab 大小（如果不同）
- `temperature` 默认 1.5，控制软标签的平滑程度

---

##### 🤔 为什么要乘 `temperature²`？

这是知识蒸馏论文（Hinton et al., 2015）中的标准做法，原因是**补偿温度缩放对梯度大小的影响**。

**数学推导（直觉版）：**

当 temperature $T$ 较大时，对 softmax 做一阶泰勒展开可以证明：

```
KL(p_teacher || p_student) ≈ (1 / 2T²) * Σ(z_s - z_t)²
```

其中 $z_s$、$z_t$ 分别是学生和教师的 logit。代入后：

```
T² * KL ≈ (1/2) * Σ(z_s - z_t)²
```

**关键结论**：
- **不乘 T²**：loss 随 $T$ 增大而**衰减**，$T$ 越大梯度越小，学生几乎学不到教师的分布信息
- **乘 T² 后**：loss 近似等于 logits 之间的 MSE，**梯度大小与 $T$ 无关**，无论温度高低都能有效学习

**直观理解**：

| 温度 $T$ | softmax 分布 | 不乘 T² 的 KL | 乘 T² 后的 KL |
|----------|-------------|--------------|--------------|
| $T=1$ | 尖锐（接近 one-hot） | 正常大小 | 正常大小 |
| $T=5$ | 平滑（分布更均匀） | 很小（梯度消失）⚠️ | 正常大小 ✅ |
| $T=10$ | 几乎均匀分布 | 接近 0 ❌ | 正常大小 ✅ |

> 温度越高，softmax 分布越平滑，每个 token 的概率差异越小。如果没有 T² 补偿，高温度下的 KL 散度值会非常小，梯度几乎为 0，学生学不到任何东西。乘 T² 后，**无论温度高低，梯度大小保持稳定**，这样我们就可以自由地用高温度来获得更平滑的 soft target（包含更多暗知识），而不必担心梯度消失。

| 项目 | 说明 |
|------|------|
| **算 loss 的部分** | 仅 assistant 回答 token（CE + KL 都只在这里算） |
| **不算 loss 的部分** | user 输入、system prompt、特殊标记、padding |
| **额外损失** | 学生模型的 MoE aux_loss |

#### 📌 输入-标签对齐方式

**不同于 Pretrain/SFT 的模型内部 shift**，蒸馏代码在训练循环中**手动 shift**：

```python
res = model(input_ids)                              # 传完整序列
student_logits = res.logits[..., :-1, :].contiguous()  # 手动[:-1]
shift_labels = labels[..., 1:].contiguous()            # 手动[1:]

ce_loss = F.cross_entropy(
    student_logits.view(-1, V),    # (B*(L-1), V)
    shift_labels.view(-1),          # (B*(L-1))
    ignore_index=-100,
    reduction='none'
)
```

```
传给模型: model(input_ids=full_seq)     ← 注意：不传 labels
                 │
代码手动:        logits[..., :-1, :]     vs     labels[..., 1:]
                 (预测位置 0..n-2)              (目标位置 1..n-1)
```

虽然效果等价于模型内部 shift，但这里明确在代码层做了对齐，并使用 `loss_mask`（基于 `labels[..., 1:] != -100`）过滤有效位置。

---

## 阶段 ⑤：DPO（Direct Preference Optimization）

**脚本**：`trainer/train_dpo.py`

### 原始数据格式

JSONL 文件，每行包含 chosen（偏好）和 rejected（非偏好）两套完整对话。

```json
{
  "chosen": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是..."}
  ],
  "rejected": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "我不知道..."}
  ]
}
```

### 数据处理流程

数据加载类：`DPODataset`（定义在 `dataset/lm_dataset.py`）

```
原始 JSONL
─────► chosen / rejected 两套对话
        │
        ├── apply_chat_template() 渲染为文本
        ├── post_processing_chat() 移除空 think
        ├── tokenizer 编码 + padding
        └── generate_loss_mask() → 标记 assistant 回答位置为 1
            输出: x = input_ids[:-1]
                  y = input_ids[1:]
                  mask = loss_mask[1:]  (1=assistant回答, 0=其他)
```

### 损失计算

**不使用模型返回的 loss**（不传 `labels` 参数给模型）。

```
策略模型 (policy model)     ← 训练，更新参数
参考模型 (reference model)  ← frozen

1. ref_model(x) → ref_logits (no_grad)
2. model(x) → logits

3. logits_to_log_probs():
   log_probs = F.log_softmax(logits, dim=2)
   per_token_log_probs = gather(log_probs, labels)

4. dpo_loss():
   ref_log_probs = (ref_log_probs * mask).sum(dim=1)    # 序列级 logprob
   policy_log_probs = (policy_log_probs * mask).sum(dim=1)

   # 前一半 batch 是 chosen，后一半是 rejected
   pi_logratios = chosen_policy_logprob - reject_policy_logprob
   ref_logratios = chosen_ref_logprob - reject_ref_logprob
   logits = pi_logratios - ref_logratios
   loss = -logsigmoid(beta * logits)  → mean()
```

| 项目 | 说明 |
|------|------|
| **算 loss 的部分** | chosen/rejected 中的 assistant 回答 token（通过 mask 控制） |
| **不算 loss 的部分** | user 输入、system prompt、特殊标记、padding |
| **额外损失** | MoE aux_loss |

#### 📌 输入-标签对齐方式

**DPO 是所有阶段中唯一在数据层就做好 shift 的阶段！** 其他阶段传给模型的是完整序列，而 DPO 在 `DPODataset.__getitem__()` 中**显式拆开**：

```python
x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)   # 输入：[:-1]
y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)    # 标签：[1:]
mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long) # mask：[1:]
```

在训练循环中：

```python
ref_outputs = ref_model(x)       # x 已经是 input_ids[:-1]
ref_logits = ref_outputs.logits   # shape: (B, L-1, V) ← 已经对齐！

ref_log_probs = logits_to_log_probs(ref_logits, y)  # y 是 input_ids[1:]
# 等价于: gather(F.log_softmax(logits), dim=2, index=y.unsqueeze(2))
```

```
数据层提前 shift:
  chosen_input_ids: [t0, t1, t2, ..., t_{L-1}]
                          │
  x_chosen (输入):       [t0, t1, ..., t_{L-2}]      ← 传给模型
  y_chosen (标签):       [t1, t2, ..., t_{L-1}]      ← 用于 gather
  mask_chosen:           [m1, m2, ..., m_{L-1}]      ← 用于过滤

模型 logits 输出 shape: (B, L-1, V) ← 和 y 天然对齐，不需要额外 shift！
```

> **为什么 DPO 要在数据层 shift？** 因为 DPO 不使用模型的 `labels` 参数和返回的 `loss`，而是**手动计算每个 token 的 log probability**。为了代码简洁，直接在数据准备阶段将输入和标签对齐好，训练时直接 `gather(log_probs, y)` 即可，无需再次 shift。

---

## 阶段 ⑥：GRPO（Group Relative Policy Optimization）

**脚本**：`trainer/train_grpo.py`

### 原始数据格式

JSONL 文件，每行包含对话列表。例如 `rlaif.jsonl`。

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有用的AI。"},
    {"role": "user", "content": "请写一首关于春天的诗"}
  ]
}
```

### 数据处理流程

数据加载类：`RLAIFDataset`（定义在 `dataset/lm_dataset.py`）

```
原始 JSONL
─────► conversations
        │
        ├── pre_processing_chat()      → 随机添加 system prompt
        │
        └── create_chat_prompt()                              ← 概率性处理③
            └── apply_chat_template(conversations[:-1],
                                    add_generation_prompt=True,
                                    open_thinking=use_thinking)
                → 只取 prompt 部分（不含最后一个 assistant 回答）
                → 按 thinking_ratio 概率开启 <think> 标签（详见后文解释）
                → 输出纯文本 prompt（模型将在训练中自行生成回答）
```

### 损失计算——强化学习范式

```
┌──────────────────────────────────────────────────┐
│ 1. Rollout：对每个 prompt 生成 num_generations 个回答│
│ 2. Reward：规则分 + RM 分                         │
│ 3. Advantage：组内标准化                           │
│ 4. Policy Loss + KL penalty                      │
│ 5. aux_loss                                      │
└──────────────────────────────────────────────────┘
```

#### 详细流程

**① Rollout 生成**：
- 每个 prompt 通过 `rollout_engine` 生成 `num_generations`（默认 6）个回答
- 记录 `per_token_logps`（旧策略的 log probability）
- 记录 `completion_mask`（标记哪些 token 是模型生成的）

---

#### 🏗️ 训推分离架构：Rollout Engine

GRPO、PPO、Agent RL 三个强化学习阶段共享同一套 **Rollout Engine 架构**，将"推理生成（rollout）"和"训练更新（training）"解耦。这套设计位于 `trainer/rollout_engine.py`。

##### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│               RolloutEngine（抽象基类）                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  rollout()        ← 生成 completion（推理）       │   │
│  │  update_policy()  ← 同步最新的策略权重            │   │
│  └──────────────────────────────────────────────────┘   │
│                      △                                   │
│                      │ 继承                               │
├──────────────────────────┬───────────────────────────────┤
│  TorchRolloutEngine      │  SGLangRolloutEngine          │
│  （同进程推理）           │  （独立进程推理）               │
│                          │                               │
│  模型和训练在同一个 GPU    │  模型部署在独立的 SGLang 服务   │
│  用 model.generate()     │  通过 HTTP API 调用推理        │
│  不需要额外部署            │  update_policy 同步权重        │
└──────────────────────────┴───────────────────────────────┘
         │                              │
         │ ─rollout()──► 同 GPU 推理     │ ─rollout()──► HTTP POST /generate
         │                              │
         │ ◄── RolloutResult ────────── │ ◄── RolloutResult ──
```

##### TorchRolloutEngine（同进程推理）

```python
class TorchRolloutEngine(RolloutEngine):
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature):
        model = unwrap(self.policy_model)     # 获取策略模型
        with torch.no_grad():                 # 不追踪梯度
            output_ids = model.generate(      # 调用 model.generate() 生成
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                do_sample=True,
                temperature=temperature,
            )
            completion_ids = output_ids[:, prompt_len:]   # 截取 completion 部分
            per_token_logps = compute_per_token_logps(...) # 计算每个 token 的 log prob
        return RolloutResult(...)

    def update_policy(self, model):
        self.policy_model = model  # 直接指针赋值，零开销
```

**特点**：
- 模型和训练在同一进程、同一 GPU 上运行
- `rollout()` 带 `torch.no_grad()`，不追踪梯度，但**显存仍被模型参数占用**
- `update_policy()` 只是指针交换，**零开销**
- 适合模型较小的场景（MiniMind 默认使用此模式）

##### SGLangRolloutEngine（独立进程推理）

```python
class SGLangRolloutEngine(RolloutEngine):
    def rollout(self, prompt_ids, ...):
        # 将 prompt 通过 HTTP 发送到独立的 SGLang 服务器
        payload = {"input_ids": input_ids_list, "sampling_params": {...}, "return_logprob": True}
        resp = requests.post(f"{self.base_url}/generate", json=payload)
        results = resp.json()
        # 解析返回的 completion_ids 和 logprobs
        return RolloutResult(...)

    def update_policy(self, model):
        # Step 1: 将模型权重保存到共享磁盘
        model.save_pretrained(self.shared_ckpt_path)
        # Step 2: 通知 SGLang 服务器从磁盘加载新权重
        requests.post(f"{self.base_url}/update_weights_from_disk",
                      json={"model_path": self.shared_ckpt_path})
```

**特点**：
- 推理在独立的 SGLang 服务器上运行，**不占用训练 GPU 的显存和计算资源**
- rollout 通过 **HTTP API** 调用，训练和推理可以运行在不同的机器上
- `update_policy()` 需要**保存权重 → 磁盘同步 → HTTP 通知服务器加载**，开销较大，所以只在 `save_interval` 步时调用，而非每步都同步
- 适合大模型场景（如 >7B 参数），训练 GPU 显存不够同时放推理

##### 启动方式

使用 SGLang 引擎时需要先启动独立的推理服务：

```bash
# 终端 1：先启动 SGLang 推理服务
python -m sglang.launch_server \
    --model-path ./minimind-3 \
    --attention-backend triton \
    --host 0.0.0.0 --port 8998

# 终端 2：再启动训练（指定 --rollout_engine sglang）
torchrun --nproc_per_node=8 train_grpo.py \
    --rollout_engine sglang \
    --sglang_base_url http://localhost:8998 \
    --sglang_model_path ../model \
    --sglang_shared_path ./sglang_ckpt_grpo
```

##### Rollout 前后的训练流程

以 GRPO 为例，完整的训推交互流程：

```
                    ┌─────────────────────────────────────────┐
                    │            训练循环 (grpo_train_epoch)    │
                    │                                          │
  ┌─────┐          │  ① rollout_engine.rollout(prompt_ids)     │
  │数据  │─────────│──────────►  生成 completion（推理模式）    │
  │加载器│          │            返回 output_ids + log_probs    │
  └─────┘          │                                          │
                    │  ② model(output_ids) 前向传播（训练模式）  │
                    │     计算 per_token_logps、aux_loss       │
                    │                                          │
                    │  ③ ref_model(output_ids) 前向（no_grad）  │
                    │     计算 ref_per_token_logps              │
                    │                                          │
                    │  ④ 计算 reward → advantage → policy_loss │
                    │     loss.backward() → optimizer.step()   │
                    │                                          │
                    │  ⑤ 每 save_interval 步:                  │
                    │     rollout_engine.update_policy(model)  │
                    │     └─ Torch: 指针赋值                    │
                    │     └─ SGLang: 存盘→HTTP通知加载          │
                    └─────────────────────────────────────────┘
```

**关键设计原则**：

1. **生成（rollout）始终不带梯度**：`torch.no_grad()` 或 SGLang 的 HTTP 调用，确保推理不污染计算图
2. **前向传播始终带梯度**：`model(output_ids)` 重新计算 logits，追踪梯度用于反向传播
3. **旧策略 logprob 由 rollout 提供并 detach**：`old_per_token_logps` 来自 rollout 阶段，用 `.detach()` 切断梯度，作为 PPO 重要性采样的"旧策略"参考
4. **新策略 logprob 由前向传播重新计算**：`per_token_logps` 来自 `model(output_ids)`，有梯度，用于更新策略

---

#### 🤖 Reward Model（奖励模型）

GRPO、PPO、Agent RL 三个 RL 阶段都使用了一个**外部**的 Reward Model 对模型生成的回答进行语义评分。`LMForRewardModel` 封装类位于 `trainer/trainer_utils.py`。

##### 架构：外部预训练模型，非 MiniMind 训练

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        #                                            ↑ 使用 trust_remote_code 加载自定义模型
        self.model = self.model.to(device).eval()
```

**关键点**：
- RM **不是 MiniMind 自己训练的**，而是加载一个**外部的、预先训练好的** Reward Model
- 默认路径：`../../internlm2-1_8b-reward`（InternLM2 1.8B 参数量的奖励模型）
- 使用 `trust_remote_code=True`，说明该模型有自定义的前向逻辑（`get_score` 方法）
- 加载后 `eval()` + `requires_grad_(False)`，**全程不参与训练**，只作为评分器

##### 代码怎么知道该用 value_head 版本还是 causal LM 版本？

**通过 `AutoModel` vs `AutoModelForCausalLM` + 模型仓库的 `config.json` 区分。**

RM 加载代码用的是 **`AutoModel`**（不是 `AutoModelForCausalLM`）：

```python
# 加载 Reward Model — 用 AutoModel
self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# 对比：MiniMind 自己的模型初始化 — 直接实例化 MiniMindForCausalLM
model = MiniMindForCausalLM(lm_config)
```

**加载流程：**

```
AutoModel.from_pretrained("../../internlm2-1_8b-reward", trust_remote_code=True)
  │
  ├── ① 读取 config.json
  │      ┌──────────────────────────────────────┐
  │      │ {                                    │
  │      │   "architectures": ["InternLM2ForRewardModel"],  ← 自定义类名
  │      │   "model_type": "internlm2",          │
  │      │   "hidden_size": 2048,                │
  │      │   ...                                 │
  │      │ }                                     │
  │      └──────────────────────────────────────┘
  │
  ├── ② config.json 指定了 "InternLM2ForRewardModel"
  │     这不是 HuggingFace 标准类 → 需要 trust_remote_code
  │
  ├── ③ 从模型仓库下载并执行 modeling_internlm2.py（或类似文件）
  │     该文件定义了：
  │       class InternLM2ForRewardModel(PreTrainedModel):
  │           def __init__(self, config):
  │               super().__init__()
  │               self.transformer = InternLM2Model(config)  # 标准 backbone
  │               self.value_head = nn.Linear(config.hidden_size, 1)  # ← 评分头，不是 lm_head！
  │
  │           def get_score(self, tokenizer, messages):
  │               ...  # 评分逻辑
  │
  └── ④ 返回 InternLM2ForRewardModel 实例
        ├── .transformer → 标准 InternLM2 backbone（1.8B）
        ├── .value_head  → Linear(2048, 1) ✅ 输出标量
        └── .get_score() → 评分方法
```

**总结：区分机制完全靠模型目录下的 `config.json`**

| | MiniMind 自己的模型 | Reward Model |
|--|-------------------|-------------|
| **加载方式** | 直接 `MiniMindForCausalLM(config)` | `AutoModel.from_pretrained(path)` |
| **config.json** | `"architectures": ["MiniMindForCausalLM"]` | `"architectures": ["InternLM2ForRewardModel"]` |
| **模型代码** | 本地 `model_minimind.py` | 外部仓库 `modeling_internlm2.py`（通过 `trust_remote_code` 下载） |
| **输出头** | `lm_head: Linear(h, vocab_size)` | `value_head: Linear(h, 1)` |
| **输出** | 词表概率分布 | 标量分数 |

RM 的 `config.json` 里写的是 `"InternLM2ForRewardModel"`（含 value_head 的版本），而不是 `"InternLM2ForCausalLM"`（含 lm_head 的版本）。所以 **`AutoModel` 根据 config 自动加载了正确的架构**，全程不需要手动指定用哪个头。

##### RM 的内部架构：它不是输出概率分布的

##### RM 的内部架构：它不是输出概率分布的

**普通语言模型 vs Reward Model 的核心区别在于输出头：**

```
普通 LM（如 MiniMind）:
  Transformer  backbone  →  lm_head (Linear: hidden_size → vocab_size)
                              ↓
                          整个词表的概率分布（如 6400 维向量）
                              ↓
                          用 CrossEntropy 选择下一个 token

Reward Model（如 InternLM2-Reward）:
  Transformer  backbone  →  value_head (Linear: hidden_size → 1)
                              ↓
                          一个标量分数（如 2.35）
                              ↓
                          用 MSE / Ranking Loss 训练评分
```

**所以 RM 输出的不是概率分布，而是一个标量数值。** 具体来说：

```text
输入: "user: 什么是机器学习？\nassistant: 机器学习是..."

      ↓ InternLM2 Transformer backbone（1.8B 参数）

最后一层 hidden_states: [batch, seq_len, 2048]

      ↓ 取最后一个 token ([EOS]) 的 hidden_state: [2048]

      ↓ value_head = nn.Linear(2048, 1)

标量分数: 2.35
```

**为什么不用概率分布？** 因为奖励任务不是"预测下一个 token"，而是**评估"这个回答有多好"**。这是一个**回归任务**（regression），不是分类任务（classification）。所以用 `value_head`（单输出神经元）替代 `lm_head`（vocab_size 个输出神经元）。

##### value_head 是已训练好的吗？

**是的，value_head 已经是训练好的完整权重的一部分。** MiniMind 加载的是完整的 `InternLM2-1.8B-Reward` 模型，其 `model.safetensors` 权重文件中**同时包含了 backbone 和 value_head 的参数**：

```python
# 加载后的模型包含以下键（示意）
checkpoint_keys = {
    "transformer.layers.0.attention.q_proj.weight",    # backbone 参数
    "transformer.layers.0.attention.k_proj.weight",
    ...
    "value_head.weight",      # ← Linear(2048, 1) 的训练好的权重
    "value_head.bias",
}
```

这个模型已经通过数万条人工标注的偏好对（chosen/rejected）训练好了。value_head 学会了如何将最后一个 token 的 hidden state 映射到有意义的评分——**不需要 MiniMind 再做任何微调**。

MiniMind 的 RL 流程相当于雇了一个"评分员"（RM），它对好的回答给高分、差的回答给低分，MiniMind 用这些分数来指导策略模型的梯度更新。整个过程 RM 处于 `eval()` 模式，参数完全冻结。

##### RM 的训练过程（了解背景）

虽然 MiniMind 不自己训练 RM，但理解其训练方式有助于正确使用：

```
训练数据: (prompt, chosen_answer, rejected_answer)
               │
     ┌─────────┴──────────┐
     ▼                    ▼
  RM(chosen)=2.5      RM(rejected)=0.8
     │                    │
     └────────┬───────────┘
              ▼
      Bradley-Terry Loss:
      loss = -log(σ(score_chosen - score_rejected))
            = -log(σ(2.5 - 0.8))
            = -log(σ(1.7))  ← 越小越好
```

通过偏好对（chosen/rejected）训练后，RM 学会了对**更好的回答给更高的分数**。

##### 评分过程

```python
@torch.no_grad()
def get_score(self, messages, response):
    # ① 构造评分 prompt
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
    last_query = messages[-1]['content'] if messages else ""
    message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}"

    eval_messages = [
        {"role": "user", "content": message_context},   # 用户的问题（含上下文）
        {"role": "assistant", "content": response}       # 模型生成的回答
    ]

    # ② 调用 RM 的评分方法（trust_remote_code 加载的自定义方法）
    score = self.model.get_score(self.tokenizer, eval_messages)

    # ③ Clip 到 [-3, 3] 范围
    return max(min(score, 3.0), -3.0)
```

**步骤分解**：

1. **构造评分上下文**：将对话历史（不含最后一个 user 消息）拼接成文本，加上"以上是对话历史。我的新问题是："前缀，和最后的 user query 组合成完整的评分 prompt
2. **组装评估消息**：构造一个 `[user, assistant]` 的对话对，user 是评分上下文，assistant 是模型生成的待评分回答
3. **调用 RM 评分**：`self.model.get_score(tokenizer, eval_messages)` — 这是外部模型通过 `trust_remote_code` 提供的自定义方法，**具体实现取决于加载的 RM 模型**（如 InternLM2 的 reward 模型内部可能使用特定的 prompt 模板和 score head）
4. **Clip 输出**：原始 RM 输出被限制在 `[-3, 3]` 范围内，防止个别极端评分主导训练

##### `get_score` 内部做了什么？（推测）

虽然 `get_score` 是通过 `trust_remote_code` 加载的外部代码（不在 MiniMind 仓库中），但根据 Reward Model 的标准架构，其内部实现大概率是：

```python
# InternLM2-Reward 的 get_score 逻辑（示意）
def get_score(self, tokenizer, messages):
    # 1. 用 chat template 渲染消息
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    
    # 2. Transformer backbone 前向
    outputs = self.model(prompt)
    last_hidden = outputs.last_hidden_state      # [1, seq_len, hidden_size]
    
    # 3. 找到最后一个 token 的位置（排除 padding）
    last_token_idx = (prompt != pad_token_id).sum(dim=1) - 1
    
    # 4. 取最后一个 token 的 hidden state
    last_token_hidden = last_hidden[range(len(prompt)), last_token_idx]  # [1, hidden_size]
    
    # 5. 通过 value_head 回归出标量分数
    score = self.value_head(last_token_hidden).squeeze(-1)  # scalar
    #                      ↑ nn.Linear(hidden_size, 1)
    
    return score.item()
```

`trust_remote_code` 允许 Hugging Face 模型仓库自带 Python 代码（在 `modeling_internlm2_reward.py` 等文件中），这些代码在 `from_pretrained` 时被自动下载并执行。因此 MiniMind 只需要指定路径，不需要关心 RM 的具体实现细节。

##### 在训练中的使用

以 GRPO 为例，RM 评分被加到 rule-based 奖励上：

```python
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=device)
    for i, prompt in enumerate(prompts):
        # ... 计算 rule-based 奖励（长度、格式等）...
        rewards[i] += rule_based_score       # 规则分

        # 调用 RM 获取语义评分
        score = reward_model.get_score(messages, answer)
        rewards[i] += score                  # RM 语义分
    
    rewards = clip(rewards, -3.0, 3.0)       # 最终 clip
    return rewards
```

**Rule-based 奖励和 RM 奖励的分工**：

| 奖励来源 | 评估内容 | 优点 | 缺点 |
|---------|---------|------|------|
| Rule-based | 长度、格式、重复率 | **确定性强**，规则明确 | **肤浅**，不能评估语义质量 |
| Reward Model | 回答的**有用性、准确性、安全性** | **语义级评估**，接近人类判断 | 有偏，可能被对抗攻击 |

Rule-based 奖励提供**基础约束**（别太短、别重复、格式要对），RM 提供**语义导向**（回答要有用、准确）。两者结合比单独使用任何一个更鲁棒。

##### 为什么不自己训练 RM？

MiniMind 直接使用现成的 InternLM2 reward 模型，而不是自己训练一个，原因：
- 训练 RM 需要大量**人工标注的偏好数据**（chosen/rejected 对），成本高
- RM 通常需要和策略模型**规模相当或更大**才能提供有效信号（1.8B 的 RM 评估 42M 的 MiniMind 足够）
- 使用 `trust_remote_code` 可以灵活接入任何 Hugging Face 上的 reward 模型

**② Reward 计算**：

| 规则 | 分值 |
|------|------|
| 回答长度在 [20, 800] 字 | +0.5 |
| 回答长度超限 | -0.5 |
| thinking 内容长度在 [20, 300] 字 | +1.0 |
| thinking 长度超限 | -0.5 |
| 有且仅有一个 `</think>` | +0.25 |
| 多个 `</think>` | -0.25 |
| 重复惩罚（n-gram 重复） | -0~0.5 |
| Reward Model 评分 | [-3, +3] |

**重复惩罚算法详解**：

```python
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())     # 分词
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]  # 构建 n-gram
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
```

**算法步骤**：

1. **分词**：`re.findall(r"\w+|[^\w\s]", text.lower())` — 按连续单词/中文字符（`\w+`）和单个非空白符号（`[^\w\s]`）切分，不保留空格
2. **构建 n-gram**：默认 `n=3`（tri-gram），从分词结果中滑动窗口生成连续的 3-token 元组
3. **计算重复率**：`(总 gram 数 - 唯一 gram 数) / 总 gram 数`，即重复 gram 的比例
4. **映射到惩罚值**：`重复比例 × cap × 2`，再 `min(cap, 结果)` 截断

**例子**：

```
输入: "今天天气真好，今天天气真好"
分词: ["今天天气真好", "，", "今天天气真好"]
3-grams: [("今天天气真好", "，", "今天天气真好")]  ← 只有 1 个
唯一 grams: 1  ← 唯一
重复率: (1-1)/1 = 0  → penalty = 0  ✅ 无重复

输入: "a b c a b c"
分词: ["a", "b", "c", "a", "b", "c"]
3-grams: [("a","b","c"), ("b","c","a"), ("c","a","b"), ("a","b","c")]
唯一 grams: {("a","b","c"), ("b","c","a"), ("c","a","b")}  ← 3 个
重复率: (4-3)/4 = 0.25  → penalty = 0.25 × 0.5 × 2 = 0.25

输入: "a a a a a"
分词: ["a", "a", "a", "a", "a"]
3-grams: [("a","a","a"), ("a","a","a"), ("a","a","a")]
唯一 grams: {("a","a","a")}  ← 1 个
重复率: (3-1)/3 = 0.667  → penalty = 0.667 × 0.5 × 2 = 0.667
cap=0.5 → penalty = 0.5  ⛔ 被 cap 截断
```

**参数含义**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `n` | 3 | n-gram 大小，控制检测的重复粒度。`n=3` 检测连续 3 个 token 的重复模式 |
| `cap` | 0.5 | 惩罚上限（同时也是缩放因子），最大扣除 0.5 分 |

惩罚值范围：`[0, cap]`，即 `[0, 0.5]`。完全无重复为 0，完全重复（所有 n-gram 都一样）时达到 cap。

**③ Advantage 计算**：


```
group_mean = mean(rewards_per_group)       # 按 prompt 分组
group_std  = std(rewards_per_group)
advantages = (rewards - group_mean) / (group_std + 1e-4)
```

**④ 策略损失**：
```
completion_mask = 仅标记模型生成的回答 token

# GRPO 模式
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * adv, clipped_ratio * adv) + β * KL

# CiSPO 模式（默认）
clamped_ratio = clamp(ratio, max=ε_high).detach()
loss = -(clamped_ratio * adv * log_prob - β * KL)

policy_loss = (per_token_loss * completion_mask).sum()
              / completion_mask.sum()
```

**⑤ KL 惩罚**：
```
kl_div = ref_log_prob - policy_log_prob
per_token_kl = exp(kl_div) - kl_div - 1    # 近似公式
```

| 项目 | 说明 |
|------|------|
| **算 loss 的部分** | 仅模型生成的 completion token（由 `completion_mask` 标记） |
| **不算 loss 的部分** | prompt（user 输入、system prompt）、特殊标记、padding |
| **额外损失** | MoE aux_loss |

#### 📌 输入-标签对齐方式

GRPO 在训练循环中**手动 shift**，并通过 `logp_pos` 定位到 completion 位置：

```python
res = model_unwrapped(outputs, attention_mask=full_mask)  # outputs 是完整序列(prompt+completion)

# 手动 shift: logits[:, :-1] vs outputs[:, 1:]
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1) \
    .gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)     # (B, L-1)

# 然后用 logp_pos 只选出 completion 位置:
# logp_pos = prompt_lens.unsqueeze(1) - 1 + arange(completion_len)
per_token_logps = per_token_logps.gather(1, logp_pos)         # (B, R)
```

```
传给模型: model(outputs=full_seq)    ← 完整序列 (prompt + completion)
                 │
代码手动:        logits[:, :-1, :]    vs     outputs[:, 1:]
                 (所有预测位置)               (所有目标位置)
                         │
                 gather(1, logp_pos)
                         │
                 仅 completion 位置的 log_prob  ← 用于策略梯度
```

> **`outputs[:, 1:]` 是干啥用的？** 在自回归语言模型中，模型用位置 $i$ 的隐藏状态预测位置 $i+1$ 的 token。所以 `logits[:, :-1, :]` 是位置 $0$ 到 $L-2$ 的预测结果，而 `outputs[:, 1:]` 对应的就是这些位置**应该预测出的正确下一个 token**（ground truth）。`gather` 操作就是从预测的概率分布中，把"真实下一个 token"对应的 log probability 提取出来。
>
> ```
> 序列位置:      0         1         2         ...   L-2        L-1
> outputs:     [tok₀,    tok₁,     tok₂,     ..., tok_{L-2}, tok_{L-1}]
>                │         │         │                 │
> logits[:,:-1]: ├─预测→  tok₁?     tok₂?     ...     tok_{L-1}?
>                │         │         │                 │
>                ▼         ▼         ▼                 ▼
> outputs[:,1:]: [tok₁,   tok₂,     tok₃,     ..., tok_{L-1}]
>                 ↑        ↑         ↑                 ↑
>               这就是"下一个 token"的 ground truth
> ```
>
> 之后再用 `gather(1, logp_pos)` **进一步筛选**，只保留 completion 部分的 log_prob，丢弃 prompt 部分。

> 与 SFT/Pretrain 不同：GRPO 虽然也 shift，但 shift 后还要通过 `logp_pos` 进一步**只选出模型生成的 token**，prompt 部分的 token log_prob 虽然算出来了但被丢弃，不参与 loss。

---

## 阶段 ⑦：PPO（Proximal Policy Optimization）

**脚本**：`trainer/train_ppo.py`

### 原始数据格式

与 GRPO 相同（`rlaif.jsonl`），使用同一个 `RLAIFDataset`。

### 数据处理流程

与 GRPO 相同。

### 模型架构

与 GRPO 最大的区别：PPO 额外维护一个 **Critic 网络**

```
Actor 模型  → MiniMindForCausalLM（策略网络，和 GRPO 一样）
Critic 模型 → CriticModel（价值网络，继承 MiniMindForCausalLM，lm_head 替换为 value_head 输出标量价值）
              └── self.value_head = nn.Linear(hidden_size, 1)
Reference 模型 → frozen，用于 KL 惩罚
Reward 模型 → frozen，用于评分
```

### 损失计算

#### A. Rollout 阶段（无梯度）

```
1. 每个 prompt 生成 1 个回答（num_generations=1）
2. 计算 rewards（规则分 + RM 分，同 GRPO）
3. Critic 前向 → 得到每个 token 的 value 估计
4. GAE 计算 advantages 和 returns：
   delta = reward_t + γ * V(s_{t+1}) - V(s_t)
   advantage_t = delta + γ * λ * advantage_{t+1}
   return_t = advantage_t + V(s_t)
5. 标准化 advantages
```

#### B. PPO 更新阶段（多轮，默认 2 轮）

**① Actor Loss（策略损失）**：
```
ratio = exp(new_log_prob - old_log_prob)          # 重要性采样比
policy_loss = -min(ratio * adv,
                   clip(ratio, 1-ε, 1+ε) * adv)   # PPO 裁剪
              + kl_coef * KL_penalty               # KL 惩罚
```

**② Critic Loss（价值损失）**：
```
value_loss = 0.5 * max((V - returns)²,
                       (clip(V, old_V - δ, old_V + δ) - returns)²)
             # 带裁剪的 MSE，防止价值估计剧烈变化
```

**③ 总损失**：
```
loss = policy_loss + vf_coef * value_loss + aux_loss
```

**④ 提前停止**（基于 KL 散度）：
```
if approx_kl > early_stop_kl:  # 默认 0.25
    stop_ppo = True            # 停止当前 batch 的 PPO 更新
```

| 项目 | 说明 |
|------|------|
| **算 loss 的部分（Actor）** | 仅模型生成的 completion token |
| **算 loss 的部分（Critic）** | 仅 completion token 位置的价值估计 |
| **不算 loss 的部分** | prompt、特殊标记、padding |
| **额外损失** | MoE aux_loss |

---

#### ⚡ 几个关键问题的回答

##### ① 开不开 MoE 对 LR scheduler 有影响吗？

**没有直接影响。** LR scheduler 的参数在初始化时确定：

```python
total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, ...)
```

这个公式中**没有任何 MoE 相关的变量**。无论是否启用 MoE，`T_max` 和 `eta_min` 完全一致，LR 的余弦退火曲线一模一样。

不过，MoE 通过 `aux_loss` **间接影响 loss 的大小**：
```python
aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
```

有 MoE 时 loss 多了一项 `aux_loss`，但这只影响梯度的数值，不影响 LR scheduler 的步数规划。

##### ② 开不开 MoE 对 MoE 路由有影响吗？

这个问题有点"先有鸡还是先有蛋"——**如果不开 MoE（`use_moe=False`），根本就没有 MoE 路由，模型用的是普通的 `FeedForward` 而不是 `MOEFeedForward`**，所以谈不上"对路由有影响"。

如果开了 MoE，路由确实会受影响，因为**每次 PPO 的 mini-batch 前向传播都会计算路由并产生 `aux_loss`**，这个 loss 的梯度会更新路由器的 gating network（`gate` 层），从而影响后续的 token 路由分配：

```python
# 每次 Actor 前向时，MOEFeedForward 都会：
# 1. 计算 gate scores
# 2. 选 top-k 专家
# 3. 计算 aux_loss（负载均衡损失）
res = actor_unwrapped(input_ids=gen_out[inds], ...)  # ← 含路由 forward
aux_loss = res.aux_loss if lm_config.use_moe else ...
```

##### ③ PPO 做了几步 off-policy 更新？

PPO 是 off-policy 算法，同一批 rollout 数据会被重复使用多次。代码中由两个参数控制：

```python
args.ppo_update_iters   # 默认 2，同一批数据做几轮 PPO 更新
args.mini_batch_size    # 默认 2，每轮更新拆成多大的 mini-batch
```

具体的 off-policy 步数计算：

```python
mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
# 默认: batch_size=2, mini_batch_size=2 → mb_factor=1

total_optimizer_steps_per_rollout = args.ppo_update_iters * mb_factor
```

**默认情况（`batch_size=2, mini_batch_size=2, ppo_update_iters=2`）**：

```
Rollout 生成 2 个样本 ──→ PPO Epoch 1 (1 个 mini-batch, 1 步 update)
                       ──→ PPO Epoch 2 (1 个 mini-batch, 1 步 update)
                       ──→ 共 2 步 off-policy 更新
```

**如果 `batch_size=4, mini_batch_size=2, ppo_update_iters=3`**：

```
Rollout 生成 4 个样本 ──→ PPO Epoch 1 (拆成 2 个 mini-batch, 2 步 update)
                       ──→ PPO Epoch 2 (2 个 mini-batch, 2 步 update)
                       ──→ PPO Epoch 3 (2 个 mini-batch, 2 步 update)
                       ──→ 共 6 步 off-policy 更新
```

另外，PPO 还有基于 KL 散度的**早停机制**：
```python
if approx_kl_val > args.early_stop_kl:  # 默认 0.25
    stop_ppo = True  # 后续 PPO epoch 全部跳过
```

这意味着实际 off-policy 步数 ≤ `ppo_update_iters * mb_factor`，如果 KL 散度增长过快（策略变化太大），会提前停止以防止策略崩溃。

#### 📌 输入-标签对齐方式

PPO 与 GRPO 类似，在训练循环中**手动 shift**，并通过 `logp_pos` 定位到 completion 位置：

```python
labels = gen_out[:, 1:].clone()       # (B, L-1) — 手动 [1:]
logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx  # completion 位置索引

res = actor_unwrapped(input_ids=gen_out, attention_mask=full_mask)

# 手动 shift: logits[:, :-1] vs labels
mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1) \
    .gather(2, labels.unsqueeze(-1)).squeeze(-1) \
    .gather(1, logp_pos)                              # 只取 completion 位置
```

```
传给 Actor: model(input_ids=gen_out)    ← 完整序列 (prompt + completion)
                    │
代码手动:          logits[:, :-1, :]     vs    labels = gen_out[:, 1:]
                    │
            gather(1, logp_pos)
                    │
            仅 completion 位置的 log_prob  ← 用于 PPO 策略梯度

传给 Critic: model(input_ids=gen_out)    ← 完整序列
                    │
            values_seq.gather(1, logp_pos)  ← 只取 completion 位置的 value
```

> Critic 的输入-标签对齐：CriticModel 返回每个 token 的 value 估计（无 shift 问题，因为是逐位置输出标量），通过 `logp_pos` 选取 completion 位置后和 `returns` 计算 MSE loss。

---

## 阶段 ⑧：Agent RL（Agent 强化学习）

**脚本**：`trainer/train_agent.py`

### 原始数据格式

JSONL 文件，每行包含带工具定义的对话和 ground truth。

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一个有用的AI助手，可以使用工具。",
      "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"get_current_weather\", \"description\": \"获取天气\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}}, \"required\": [\"location\"]}}}]"
    },
    {
      "role": "user",
      "content": "北京今天天气怎么样？"
    }
  ],
  "gt": ["28°C", "晴"]
}
```

数据中的 `conversations` 只包含到 user 的提问，**不包含 assistant 的回答**（模型在训练中自行多轮生成）。

### 数据处理流程

数据加载类：`AgentRLDataset`（定义在 `dataset/lm_dataset.py`）

```
原始 JSONL
─────► sample
        │
        ├── parse_conversations()
        │   ├── 提取 messages（除去最后一个 assistant 回答——这里没有回答，全部保留）
        │   └── 提取 tools 定义（从 system 消息的 "tools" 字段）
        │
        └── return {'messages': messages, 'tools': tools, 'gt': gt}
```

**Dataset 输出**：原始的 Python 对象（`messages` 是 `[{role, content}, ...]` 列表，`tools` 是函数定义字典）。真正渲染为模型输入文本是在训练循环中的 `rollout_single()` 函数里完成的。

### 工具定义与模拟执行

工具的定义、模拟数据和执行逻辑都在 `train_agent.py` 脚本的顶部（属全局配置，非 Dataset 的一部分）：

**① 工具 Schema 定义**（`train_agent.py` 第 41-48 行）：

```python
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math",      ...}},
    {"type": "function", "function": {"name": "unit_converter",      ...}},
    {"type": "function", "function": {"name": "get_current_weather", ...}},
    {"type": "function", "function": {"name": "get_current_time",    ...}},
    {"type": "function", "function": {"name": "get_exchange_rate",   ...}},
    {"type": "function", "function": {"name": "translate_text",      ...}},
]
```

每个工具定义了名称、描述和参数规范（符合 OpenAI function calling 格式）。这些 schema 会在 `apply_chat_template(..., tools=TOOLS)` 时被渲染到 system prompt 中。

**② 模拟数据**（第 50-62 行）：

```python
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), ...}
TIME_DATA    = {"Asia/Shanghai": "2025-03-07 14:30:00", ...}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ...}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ...}
UNIT_DATA      = {"km_miles": 0.621371, ...}
```

**③ 模拟执行函数**（第 64-74 行）：

```python
MOCK_RESULTS = {
    "calculate_math":      lambda args: eval(...),                     # 真正计算表达式
    "unit_converter":      lambda args: args.value * UNIT_DATA[...],   # 查表换算
    "get_current_weather": lambda args: WEATHER_DATA.get(args.location, ("22°C", "晴")),  # 查固定表
    "get_current_time":    lambda args: TIME_DATA.get(args.timezone, ...),                 # 查固定表
    "get_exchange_rate":   lambda args: EXCHANGE_DATA.get(...),                            # 查固定表
    "translate_text":      lambda args: TRANSLATE_DATA.get(...),                           # 查固定表
}
```

**④ 工具调用解析**（第 86-88 行）：

```python
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        calls.append(json.loads(m.strip()))
    return calls
```

从模型生成的文本中用正则提取 `<tool_call>...</tool_call>` 之间的 JSON。

**⑤ 工具执行与超时保护**（第 90-99 行）：

```python
def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    signal.alarm(1)   # 1 秒超时，防止 eval 死循环
    return fn(args)
```

**总结**：所有这些工具都是**模拟（mock）执行**，并非真实 API 调用：
- `calculate_math` 用 Python `eval()` 实时计算，算"真"的
- 其他工具全部查**预定义的硬编码数据表**，无论训练多少次，北京天气永远是 `28°C/晴`
- 训练目的是让模型学会**工具调用的格式和流程**（何时输出 `<tool_call>`、如何填参数、如何解析结果），而非真正获取外部数据

### 多轮 Rollout 完整流程详解

Agent RL 的核心是 `rollout_single()` 函数，它实现最多 3 轮的对话式工具调用。下面以一个具体例子逐步展示。

#### 第 1 轮：模型生成工具调用

```
原始 messages（含 tools 定义）:
  system: "你是一个有用的AI助手，可以使用工具。"
          + tools: [get_current_weather 定义]
  user:   "北京今天天气怎么样？"

apply_chat_template(tokenize=False, add_generation_prompt=True, tools=...)
  │
  ▼
渲染后的 prompt 文本（模型输入）:
"<|im_start|>system
你是一个有用的AI助手，可以使用工具。

# Tools

You may call one or more functions...

<tools>
{"type": "function", "function": {"name": "get_current_weather", ...}}
</tools>

For each function call, return a json object...
<|im_end|>
<|im_start|>user
北京今天天气怎么样？<|im_end|>
<|im_start|>assistant
<think>
</think>"

模型在 "<|im_start|>assistant\n<think>\n" 之后开始生成...
  │
  ▼
模型生成文本:
"用户想知道北京的天气，我需要调用 get_current_weather 工具。
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "北京"}}
</tool_call>"
```

#### 第 2 轮：工具执行结果反馈

```
解析 <tool_call> → 调用模拟执行函数 → 得到结果

工具执行结果:
{"city": "北京", "temperature": "28°C", "condition": "晴"}

将结果拼接回对话，再次构造 prompt:
messages 更新为:
  system: (同上)
  user:   "北京今天天气怎么样？"
  assistant: (上一轮生成的工具调用文本)
  tool:  '{"city": "北京", "temperature": "28°C", "condition": "晴"}'

apply_chat_template() 会渲染 tool 响应为:
"<|im_start|>user
<tool_response>
{"city": "北京", "temperature": "28°C", "condition": "晴"}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
</think>"

模型继续生成...
  │
  ▼
模型生成文本:
"北京今天天气晴朗，气温28°C。"
```

#### 第 3 轮（如果有需要）：检查是否还需调用工具

```
没有 <tool_call> → 结束对话
```

#### 多轮生成的 token 序列合并

```
所有轮次拼接成的完整 input_ids（示意）:

[prompt_token_ids...,                          ← 原始 prompt（不参与 loss）
 gen1_token_ids...,                             ← 第1轮生成（参与 loss）
 tool_response_ids...,                          ← 工具响应[:](不参与 loss)
 gen2_token_ids...,                             ← 第2轮生成（参与 loss）
 PAD, PAD, PAD, ...]                           ← padding（不参与 loss）

对应的 response_mask:  [0,0,...,0,  1,1,1...,1,  0,0,...,0,  1,1,1...,1,  0,0,...]
                        ↑prompt     ↑gen1       ↑tool_resp  ↑gen2        ↑pad

tool_response 不参与 loss：因为工具响应是外部系统返回的，不是模型生成的，
模型不应该学习"预测工具会返回什么"。
```

### 损失计算——多轮 GRPO 风格

```
1. 多轮 Rollout（最多 3 轮）：
   │
   ├── 模型生成 → 解析 <tool_call> → 模拟执行工具 → 反馈结果
   ├── 模型再次生成 → 解析工具调用或给出最终答案
   └── 记录所有轮的 completion token 和 log probability

2. Reward 计算（分场景）：
   │
   ├── 无工具调用：
   │   ├── 长度分 [±0.5]
   │   ├── thinking 格式分 [-0.5~+1.25]
   │   ├── RM 分 [-3~+3]
   │   └── 重复惩罚（详见下方算法说明）
   │
   └── 有工具调用：
       ├── 工具对齐分（工具调用数量与 ground truth 匹配度）[±0.5/差值]
       ├── GT 答案匹配分 [0~+2.5]
       ├── 未完成扣分 [-0.5]
       ├── 标签不匹配扣分（<tool_call> 标签未闭合）
       └── 重复惩罚（详见下方算法说明）

3. Advantage：组内标准化（同 GRPO）

4. Policy Loss：CiSPO 或 GRPO，仅生成的回答 token

5. KL 惩罚：与参考模型的 KL 散度

6. aux_loss（MoE）
```

| 项目 | 说明 |
|------|------|
| **算 loss 的部分** | 仅多轮生成的 completion token（gen1 + gen2 + ...，即 `response_mask=1` 的位置） |
| **不算 loss 的部分** | prompt（含工具定义）、tool_response（外部反馈）、padding |
| **特殊处理** | `completion_mask` 标记所有轮次的生成内容，并通过 EOS 位置截断 |

#### 📌 输入-标签对齐方式

Agent RL 与 GRPO 类似，通过**手动 shift + `completion_mask` 过滤**：

```python
res = model_unwrapped(input_ids, attention_mask=full_mask)

# 手动 shift: logits[:, :-1, :] vs input_ids[:, 1:]
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1) \
    .gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)    # (B, L-1)
```

然后通过 `completion_mask`（标记所有轮次生成的 token）过滤出有效位置：

```python
completion_mask = full_response_masks[:, 1:]       # mask 也 shift 成 (B, L-1)
# 再根据 EOS 位置截断
is_eos = (input_ids[:, 1:] == eos_token_id) & completion_mask.bool()
eos_idx = is_eos.int().argmax(dim=1)
pos = torch.arange(completion_mask.size(1)).unsqueeze(0)
completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
# 最终 per_token_loss 只在这里有 1 的位置计算
```

```
传给模型: model(input_ids=full_seq)    ← 完整序列 (多轮对话)
                                                    │
                                                    ▼
完整序列示意:
 [prompt_0, ..., prompt_n,                          ← 原始 prompt
  gen1_0, ..., gen1_m,                              ← 第1轮生成 ✓
  tool_resp_0, ..., tool_resp_k,                    ← 工具响应 ✗
  gen2_0, ..., gen2_p, EOS,                         ← 第2轮生成 ✓
  PAD, PAD, ...]                                    ← padding ✗

代码手动 shift:
  logits[:, :-1, :] 预测 → [prompt_0, ..., prompt_n, gen1_0, ..., gen1_{m-1}, tool_resp_0, ..., gen2_{p-1}, EOS, PAD]
  input_ids[:, 1:]    目标 → [prompt_1, ..., gen1_0, gen1_1, ..., gen1_m,     tool_resp_1, ..., gen2_p,     PAD, PAD]
                              │                      │                           │
                     × completion_mask 过滤:   0=丢弃                      1=保留                         0=丢弃
                              │                      │                           │
                              ▼                      ▼                           ▼
                       不参与 loss             参与 loss（策略梯度+KL）       不参与 loss
```

> **关键区别**：Agent RL 的 `completion_mask` 不像 GRPO 那样通过 `logp_pos` 索引选取，而是直接用 0/1 mask 乘以 `per_token_logps`，mask=1 的位置（多轮中所有模型生成的 token）参与 loss，mask=0 的位置（prompt、tool_response、padding）被丢弃。

---

## Agent RL vs GRPO：全面对比

虽然 Agent RL 和 GRPO 共享相同的策略梯度框架（CiSPO/GRPO loss、组内 Advantage 标准化、KL 惩罚），但在多个关键维度上有显著差异。

### 核心差异总览

| 维度 | GRPO | Agent RL |
|------|------|----------|
| **Rollout 轮数** | 单轮（1 次生成） | 多轮（最多 3 轮，含工具调用） |
| **数据来源** | `RLAIFDataset`（纯对话 prompt） | `AgentRLDataset`（对话 + 工具定义 + ground truth） |
| **completion 标记方式** | `logp_pos` 索引定位 | 0/1 `completion_mask` |
| **Reward 信号** | 格式分 + RM 分 | **分场景**：有/无工具调用，含 GT 匹配 |
| **每个 prompt 生成数** | `num_generations`（默认 6） | `num_generations`（默认 4） |
| **策略更新** | on-policy（每步 1 次更新） | on-policy（每步 1 次更新） |
| **Loss 公式** | CiSPO / GRPO | CiSPO / GRPO |
| **KL 惩罚** | ✅ 对 reference model | ✅ 对 reference model |
| **依赖外部模型** | Reward Model | Reward Model（可选） |

### Rollout 机制对比

**GRPO（单轮）**：
```
每个 prompt ──→ rollout_engine ──→ num_generations 个完整回答（一次性生成）
               │
               └── 所有回答是独立的、并行的
```

**Agent RL（多轮）**：
```
每个 prompt ──→ rollout_single() ──→ Turn 1: 模型生成文本
                                         │
                                   解析 <tool_call>?
                                         │
                                    是 ──→ 模拟执行工具 → 反馈结果 → Turn 2
                                         │
                                    否 ──→ 结束
                                         │
                                    Turn 2: 模型继续生成
                                         │
                                    ...（最多 3 轮）
                                         │
                                    └── 所有轮的 token 合并为一个序列
```

### Reward 设计对比

#### GRPO Reward（单场景）

GRPO 的 reward 对所有回答一视同仁，不管内容是什么：

| 规则 | 分值 | 设计意图 |
|------|------|----------|
| 回答长度 20~800 字 | +0.5 | 鼓励有效回答，拒绝空话 |
| thinking 内容 20~300 字 | +1.0 | 鼓励适度思考 |
| `</think>` 有且仅有一个 | +0.25 | 鼓励格式规范 |
| 重复惩罚 | -0~0.5 | 抑制 n-gram 重复 |
| RM 评分 | [-3, +3] | 来自 Reward Model 的语义评分 |

所有分数加和后 clip 到 `[-3, 3]`。

#### Agent RL Reward（双场景分支）

Agent RL 的 reward 函数是一个**带分支的决策树**，根据模型是否调用了工具走不同的评分路径：

```
                          ┌─── 无工具调用 ─── 纯格式分 + RM 分（同 GRPO）
                          │
模型输出 ──→ 是否有 <tool_call>?
                          │
                          └─── 有工具调用 ─── 工具对齐分 + GT 匹配分
```

**场景 A：无工具调用**（与 GRPO 基本相同）

| 规则 | 分值 |
|------|------|
| 回答长度 5~800 字 | ±0.5 |
| thinking 格式分 | -0.5~+1.25 |
| RM 评分 | [-3, +3] |
| 重复惩罚 | -0~0.5 |

总分 clip 到 `[-3, 3]`。

**场景 B：有工具调用**（Agent RL 独有的复杂评分）

| 规则 | 分值 | 说明 |
|------|------|------|
| 标签不匹配 | `-0.5 × \|count(<tool_call>) - count(</tool_call>)\|` | 每轮的标签不闭合都要扣分 |
| 工具对齐分 | `+0.5` 或 `-0.5 × tool_gap` | `tool_gap = \|有效调用数 - GT 工具数\| + 多余无效调用数` |
| GT 答案匹配 | `+2.5 × len(matched) / len(gt)` | 用关键词/数值匹配检测答案正确性 |
| 未完成扣分 | -0.5 | 达到最大轮数仍未闭合对话 |
| 重复惩罚 | -0~0.5 | n-gram 重复 |

总分 clip 到 `[-3, 3]`。

### 多轮 Reward 的分配方式

**关键设计：Agent RL 的多轮对话只有一个整体 reward，而不是每轮分配一个 reward。**

```python
# train_agent.py calculate_rewards()
rewards = torch.zeros(len(completions), device=device)

for idx, response in enumerate(completions):
    turn_outputs = turn_outputs_batch[idx]       # 所有轮的输出文本
    turn_answers = [...]                          # 每轮去掉 think 后的内容
    tool_calls = []                                # 从所有轮中解析工具调用
    for turn_answer in turn_answers:
        tool_calls.extend(parse_tool_calls(turn_answer))

    # 对整个多轮轨迹计算单一 reward
    reward = 格式分 + 思考分 + RM分 - 重复惩罚               # 无工具场景
    # 或
    reward = 工具对齐分 + GT匹配分 - 未完成扣分 - 重复惩罚   # 有工具场景

    rewards[idx] = clip(reward, -3.0, 3.0)  # 一个 scalar，作用于所有 completion token
```

**这个单一 reward 被应用到整个序列的所有 completion token 上**——通过组内标准化广播到每个 token，而不是通过时间差分（TD）分配：

```python
# Group-level advantage: 每个 sequence 一个 scalar
advantages = (rewards - group_mean) / (group_std + 1e-4)  # [B*num_gen]

# 直接广播到所有 token 位置，没有 GAE！
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - beta * per_token_kl)
#                       ↑ (B*num_gen, 1)  →  广播到 (B*num_gen, R)
#  同一个 sequence 内，位置 0 和位置 R-1 的 advantage 一模一样
```

这意味着：
- **第 1 轮生成和第 3 轮生成的 token，advantage 值完全相同**
- **没有显式给任何 token 或步骤分配 reward**——不存在"第 1 步调对了工具给 +0.5，第 2 步答案对了再给 +1.0"这样的逐步骤奖励
- 如果模型第 1 轮正确调用了工具但最终答案错了，**所有 token 统一受到惩罚**
- 工具响应 token 被 `response_mask=0` 排除，不参与 loss

---

#### ⚡ 和 PPO 的关键对比：PPO 有逐 token 的 reward 分配

GRPO 和 Agent RL 不做逐 token 分配，但 **PPO 做了**。

在 PPO 中，外部 reward **只被放到 completion 的最后一个 token 上**，然后通过 GAE（Generalized Advantage Estimation）**向后传播**到前面的每个 token：

```python
# PPO: 外部 reward 只放在最后一个 token
token_rewards = torch.zeros_like(old_resp_logp)   # [B, R]，初始全 0
last_idx = resp_lengths - 1                        # 最后一个有效 token 的索引
token_rewards[torch.arange(B), last_idx] += rewards  # ⚡ 只在末尾放 reward

# GAE 反向传播：从最后一个 token 往前，每个 token 得到不同的 advantage
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
# → advantages: [B, R]，每个位置的 advantage 都不相同！
```

| 方法 | Reward 分配方式 | 每个 token 的 advantage |
|------|---------------|----------------------|
| **GRPO / Agent RL** | 同一个 scalar 广播到所有 token | 完全相同（无时间维度） |
| **PPO** | reward 放在末尾 → GAE 反向传播 | **各不相同**（靠近末尾的 token 影响更大） |

```
GRPO / Agent RL 的 advantage 分布（一个 sequence）:
  token:  [t0, t1, t2, t3, t4, t5]
  adv:    [a,  a,  a,  a,  a,  a ]    ← 一模一样

PPO 的 advantage 分布（一个 sequence）:
  token:  [t0, t1, t2, t3, t4, t5]
  adv:    [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]   ← 靠后的 token 影响更大
                                   ↑ 外部 reward 放在这里
```

**所以你的理解完全正确**：Agent RL 没有给任何 token 或步骤显式分配 reward，所有 token 共享同一个 advantage。PPO 则是唯一通过 GAE 做了逐 token 差异化分配的方法。

### 参数对比

| 参数 | GRPO 默认值 | Agent RL 默认值 |
|------|-------------|-----------------|
| `--num_generations` | 6 | 4 |
| `--beta`（KL 惩罚系数） | 0.1 | 0.1 |
| `--loss_type` | cispo | cispo |
| `--epsilon_high` | 5.0 | 5.0 |
| `--thinking_ratio` | 0.9 | 0.1 |
| `--learning_rate` | 3e-7 | 3e-7 |
| 最大序列长度 | `max_seq_len + max_gen_len` | `max_seq_len + 额外多轮`（通过 `max_total_len` 兜底） |

最关键的参数差异是 `thinking_ratio`（0.9 vs 0.1）和 `num_generations`（6 vs 4），反映了两个任务的不同侧重：GRPO 需要深入推理，Agent RL 需要精确执行。

---

## Agent RL 常见问题（FAQ）

### ① 工具在哪里定义的？Eval 时有真正能调用的工具吗？

工具定义在 **3 个地方**，各自用途不同：

| 文件 | 用途 | 工具数量 |
|------|------|---------|
| `trainer/train_agent.py` 第 41-48 行 | **训练时**的 tool schema 定义 | 6 个 |
| `scripts/eval_toolcall.py` 第 18-37 行 | **评估/测试时**的 tool schema | 8 个 |
| `scripts/web_demo.py` 第 107 行 | **Web UI 演示**时的 tool schema | 8 个 |

训练和 eval 的工具**不完全一样**——eval 比训练多了 `random_number` 和 `text_length`。

**没有真正的 API 调用，全部是模拟（mock）执行**。`get_current_weather`、`get_exchange_rate`、`translate_text` 全部返回硬编码的固定值：

```python
# eval_toolcall.py MOCK_RESULTS
"calculate_math":       lambda args: eval(...),                    # ✅ 真正计算表达式
"get_current_time":     lambda args: datetime.now().strftime(...), # ✅ 真的获取当前时间
"random_number":        lambda args: random.randint(...),          # ✅ 真的生成随机数
"text_length":          lambda args: len(...),                     # ✅ 真的计算文本长度
"unit_converter":       lambda args: {"result": 0.621371},         # ✗ 硬编码
"get_current_weather":  lambda args: {"temperature": "22°C"},      # ✗ 硬编码
"get_exchange_rate":    lambda args: {"rate": 7.15},               # ✗ 硬编码
"translate_text":       lambda args: {"translated": "hello world"},# ✗ 硬编码
```

其中真正"实时"的只有 `calculate_math`（Python eval 实时计算）、`get_current_time`（`datetime.now()` 返回真实时间）、`random_number`（`random.randint()` 生成随机数）。其他全部返回固定值，无论你问北京还是东京，天气永远是 `"22°C"`，汇率永远是 `7.15`。

**训练目的**是让模型学会**工具调用的格式和流程**（何时输出 `<tool_call>`、如何填参数、如何解析结果并给出最终答案），而非真正获取外部数据。

---

### ② Agent RL 只是 GRPO 加了 function calling 吗？

**是的，但不止。** Agent RL = GRPO 框架 + function calling 带来的 3 项结构性变化：

**① 单轮 → 多轮 Rollout**

| | GRPO | Agent RL |
|--|------|----------|
| 生成方式 | 一次性生成 `num_generations` 个回答 | 最多 3 轮串行生成 |
| completion 标记 | `logp_pos` 索引定位 | 0/1 `completion_mask` |
| 中间产物 | 无 | 每轮的 `tool_response` token 被 mask=0 排除 |

GRPO 的 rollout 是一次 `rollout_engine.rollout()` 调用就拿到所有回答。Agent RL 要在循环中反复调用，每轮把工具响应拼回 messages 后再调下一轮。

**② Reward 从单一评分变成双场景分支**

```
GRPO:      格式分 + RM 分  →  同一套规则适用于所有回答

Agent RL:  是否有 <tool_call>?
              ├── 否 → 格式分 + RM 分（和 GRPO 一样）
              └── 是 → 工具对齐分 + GT 答案匹配分 + 未完成扣分
                        （完全不同的评分逻辑）
```

有工具调用时，reward **不再看 RM 分**，而是看工具调用数量是否和 ground truth 匹配、最终答案是否包含 GT 中的关键词/数值。

**③ 需要 ground truth（gt）字段**

| | GRPO 数据 | Agent RL 数据 |
|--|----------|--------------|
| 数据字段 | `conversations` | `conversations` + **`gt`** |
| gt 是什么 | 无 | 期望的工具名/答案值，如 `["28°C", "晴"]` |
| gt 的用途 | 无 | 用于计算工具对齐分和 GT 匹配分 |

---

### ③ Reward 是最后算一个 scalar，每个 token 平分吗？有显式的逐 token 分配吗？

**完全正确。** Agent RL（和 GRPO）都没有逐 token 或逐步骤的 reward 分配。同一个序列的所有 completion token 共享完全相同的 advantage 值：

```python
# GRPO & Agent RL：一个 sequence 一个 scalar，广播到所有 token
advantages = (rewards - group_mean) / (group_std + 1e-4)   # [B*num_gen]
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - ...)
#                       ↑ 广播到所有 token 位置
```

```
token:   [t0, t1, t2, t3, t4, t5]
adv:     [a,  a,  a,  a,  a,  a ]    ← 广播，全部一样
```

**相比之下，PPO 做了逐 token 分配**——把外部 reward 只放在最后一个 token 上，通过 GAE 反向传播得到每个位置各不相同的 advantage：

```python
# PPO：reward 只放在末尾 token，GAE 反向传播
token_rewards = torch.zeros_like(old_resp_logp)
token_rewards[..., last_idx] += rewards   # 只在最后一个 token 放 reward

for t in reversed(range(gen_len)):
    delta = token_rewards[:, t] + gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    # → advantages: [B, R]，每个 token 各不相同
```

```
token:   [t0, t1, t2, t3, t4, t5]
adv:     [0.1, 0.2, 0.5, 0.8, 1.5, 2.0]   ← 靠后的 token 影响更大
                                      ↑ 外部 reward 放在这里
```

| 方法 | 分配方式 | 每个 token 的 advantage |
|------|---------|----------------------|
| **GRPO / Agent RL** | 广播同一个 scalar | 完全相同 |
| **PPO** | reward 放末尾 → GAE 传播 | **各不相同** |

---

## 概率性处理方式的设计意图

MiniMind 的代码中有多处使用了 **概率性的数据处理方式**（非确定性，每次运行结果可能不同）。这些设计不是随意的，各有其明确的训练目的。下面逐一解释。

---

### 1. SFT 阶段：`pre_processing_chat` — 20% 概率添加 system prompt

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:  # 20% 概率
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
```

**目的**：让模型在**有 system prompt** 和 **无 system prompt** 两种场景下都能正常工作。

- **为什么不是 100% 都加？** 如果每条数据都加 system prompt，模型会"依赖" system prompt 的存在。实际推理时用户可能不提供 system prompt，模型性能会下降。
- **为什么不是 0%（不加）？** 很多对话数据本身没有 system prompt，但实际部署中 system prompt 是常用功能。完全不训练这个场景，模型遇到 system prompt 时可能表现不佳。
- **为什么是 20%？** 这是一个经验值——在真实部署中，system prompt 的出现频率通常远低于 user/assistant 对话，20% 的比例反映了实际分布的粗略估计。同时，少量样本足以让模型学会处理 system prompt 格式，又不会让它过度依赖。

**补充**：`SYSTEM_PROMPTS` 包含 10 条不同的中英文 system prompt，这样模型不会只记忆某一条固定的 system prompt，而是学会泛化处理各种 system prompt。

---

### 2. SFT/DPO 阶段：`post_processing_chat` — 80% 概率移除空 think 标签

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
```

**目的**：控制模型在**有思考过程**和**无思考过程**之间的行为。

`apply_chat_template` 的 generation prompt 部分是这样处理的：

```
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if open_thinking is defined and open_thinking is true %}
        {{- '<think>\n' }}              ← open_thinking=True：<think 不闭合，模型接着生成思考内容
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }} ← open_thinking=False：注入一个空的已闭合的 think 块
    {%- endif %}
{%- endif %}
```

#### 为什么 `open_thinking=False` 要注入一个空的 `<think>\n\n</think>\n\n`，而不是直接不加 think 标签？

这个设计的核心目的是：**通过在 prompt 中控制 `<think>` 标签的闭合状态，来实现推理时"是否思考"的可控切换，而无需改变模型权重或添加额外的控制 token。**

**两种模式的对比：**

| 模式 | prompt 结尾 | 模型接着生成 | 效果 |
|------|------------|-------------|------|
| `open_thinking=True` | `<|im_start|>assistant\n<think>\n` | 生成思考内容 → `</think>` → 答案 | ✅ 模型会思考 |
| `open_thinking=False` | `<|im_start|>assistant\n<think>\n\n</think>\n\n` | 直接在 `\n\n` 后生成答案 | ✅ 模型直接回答 |

**为什么空 think 块能做到两用？**

关键在 `<think>\n\n</think>\n\n` 的格式——`<think>` 和 `</think>` 之间有两个换行符，之后又有两个换行符：

1. **当 `open_thinking=True`**：`<think>` 没有 `</think>` 闭合，模型自然会接着 `<think>\n` 继续生成思考内容，然后在某个位置自己输出 `</think>` 后给出答案。

2. **当 `open_thinking=False`**：`<think>\n\n</think>\n\n` 中的 think 块**已经闭合**，模型不会在已闭合的标签内生成内容，而是在最后的 `\n\n` 之后直接开始生成答案。看起来就像完全没有 think 标签一样。

**这样做的好处**：训练时模型在两种 prompt 格式下都见过，因此推理时只需**切换 prompt 中 `<think>` 的闭合状态**，就能控制模型是否进入思考模式——不需要加载不同的 LoRA 权重或模型副本。

#### 那为什么 SFT 中还要 80% 概率移除空 think？

既然空 think 块有上述设计优势，为什么 SFT 中用 `post_processing_chat` 把它移除掉？

原因在于**SFT 和 RL 的训练目标不同**：

| 阶段 | 数据来源 | 移除空 think | 原因 |
|------|---------|-------------|------|
| **SFT** | 人工标注的标准回答 | ✅ 80% 移除 | SFT 数据是"标准答案"，不需要思考过程。如果每条数据都有 `<think>\n\n</think>`，模型可能学会输出空 thinking 装样子，浪费 token |
| **RL（GRPO/PPO）** | 模型自行生成 | ❌ 不移除 | RL 中模型自己决定是否思考（通过 `thinking_ratio`），需要 `<think>` 标签结构作为控制开关 |

SFT 中保留 20% 的样本含空 think 标签，是为了让模型**知道有 think 这个格式**，但又不至于形成依赖。到了 RL 阶段，模型已经完全理解了 think 格式，就可以通过 `open_thinking` 自由控制思考行为。

> 注意：`post_processing_chat` 只影响**空** thinking 标签。如果数据中本来就含有**非空** thinking 内容（如 CoT 数据），`<think>...</think>` 不会被移除。`post_processing_chat` 检查的是精确的字符串 `<think>\n\n</think>\n\n`。

---

### 3. GRPO/PPO/Agent RL 阶段：`thinking_ratio` — 按概率开启 thinking 模式

```python
# RLAIFDataset.create_chat_prompt()
use_thinking = random.random() < self.thinking_ratio  # 默认 0.5~0.9
return self.tokenizer.apply_chat_template(
    conversations[:-1],
    tokenize=False,
    open_thinking=use_thinking,  # True → 注入未闭合的 <think>\n；False → 注入已闭合的 <think>\n\n</think>\n\n
    add_generation_prompt=True
)
```

**目的**：在强化学习阶段，控制模型**主动思考的频率**。

- **和 SFT 阶段的区别**：SFT 中通过 `post_processing_chat` 处理**已有的** thinking 内容是否保留；而 RL 阶段通过 `open_thinking` 控制的是**prompt 中 `<think>` 标签的闭合状态**，从而控制模型生成时是否进入思考模式。
- **`open_thinking=True`**（prompt 以 `<think>\n` 结尾）：`<think>` 未闭合，模型接着生成 `<think>` 内部的内容（即思考过程），之后再输出 `</think>` 和答案。**模型会思考。**
- **`open_thinking=False`**（prompt 以 `<think>\n\n</think>\n\n` 结尾）：`<think>` 已经闭合，模型在 `\n\n` 之后直接生成答案。**模型直接回答。**
- **GRPO 的 `thinking_ratio=0.9`**：高概率开启 thinking，因为强化学习阶段通常针对复杂推理任务（数学、代码、逻辑等），希望模型多思考再回答。但保留 10% 的概率不开启，让模型也学会不用思考就能直接回答的简洁场景。
- **Agent RL 的 `thinking_ratio=0.1`**：低概率开启 thinking，因为工具调用场景更注重准确执行函数调用，冗长的思考过程反而干扰工具调用格式。但也保留 10% 的概率开启，防止模型在需要简单推理的工具场景中完全不思考。
- **PPO 的 `thinking_ratio=0.9`**：同 GRPO，针对复杂推理任务。

---

### 4. Tokenizer 训练中为什么只取前 10000 行

```python
def get_texts(data_path):
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10000: break  # 只取 10000 行
            ...
```

**目的**：快速验证 tokenizer 训练流程，而非生产级训练。

- 完整的 tokenizer 训练需要大量数据和较长时间。取前 10000 行可以在几秒内完成训练，方便开发者测试和调试代码逻辑。
- 注释明确说明"不建议再重复训练 tokenizer"，项目已自带训练好的 tokenizer，此脚本仅供学习参考。

---

### 5. SFT 阶段：`add_system_ratio` 为什么用于排除工具数据

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations):
        return conversations  # 工具数据不做任何处理
    ...
```

**目的**：保护工具调用数据格式的完整性。

- 工具调用数据中的 system prompt 通常包含**完整的工具定义 JSON**（functions schema），具有严格的结构要求。
- 如果随机替换或添加 system prompt，工具定义会被破坏，模型会学到错误的工具调用格式。
- 因此对含有 `tools` 字段的数据，跳过所有的 system prompt 随机处理。

---

### 总结：概率性处理的核心设计哲学

| 处理方式 | 概率值 | 主要目的 |
|----------|--------|----------|
| SFT 添加 system prompt | 20% | 让模型适应有无 system prompt 两种场景 |
| SFT/DPO 移除空 think | 80% | 让模型默认直接回答，保留思考能力但不依赖 |
| GRPO 开启 thinking | 90% | 强化学习任务鼓励多思考 |
| Agent RL 开启 thinking | 10% | 工具调用任务鼓励简洁执行 |
| Tokenizer 采样 | 10000 行 | 快速验证，非生产使用 |
| 排除工具数据 | 100% | 保护工具格式完整性 |

这些概率值的共同目标是：**在训练数据中引入可控的多样性，让模型在面对不同输入分布时都能有稳定的表现**。它们本质上是**数据增强**的一种形式，而非随机噪声。

---

| 方面 | 预训练 | SFT / LoRA | 蒸馏 | DPO | GRPO / Agent RL | PPO |
|------|--------|-------------|------|-----|------------------|-----|
| **学习范式** | 自监督 | 监督学习 | 监督+蒸馏 | 偏好优化 | 强化学习 | 强化学习 |
| **数据是否含回答** | 纯文本 | ✅ 含标准回答 | ✅ 含标准回答 | ✅ chosen/rejected 对 | ❌ 无（模型自生成） | ❌ 无（模型自生成） |
| **Loss 作用范围** | 全文所有 token | 仅 assistant 回答 | 仅 assistant 回答 | 仅 assistant 回答 | 仅模型生成的 completion | 仅模型生成的 completion |
| **Loss 是否需要 mask** | 仅 padding mask | ✅ generate_labels | ✅ loss_mask | ✅ generate_loss_mask | ✅ completion_mask | ✅ completion_mask |
| **是否使用 RM** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **是否使用 ref model** | ❌ | ❌ | ✅（教师） | ✅ | ✅ | ✅ |
| **是否更新全部参数** | ✅ | ✅ / ❌（仅 LoRA） | ✅（学生） | ✅ | ✅ | ✅（Actor + Critic） |
| **是否多轮生成** | ❌ | ❌ | ❌ | ❌ | ✅（Agent 最多 3 轮） | ❌ |
| **是否支持多卡 DDP** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 关于 aux_loss（MoE 辅助损失）

当启用 MoE（`use_moe=True`）时，**所有 8 个训练阶段**都会额外增加一个路由器负载均衡损失，定义在 `MOEFeedForward.forward` 中：

```python
load = F.one_hot(topk_idx, num_experts).float().mean(dim=(0, 1))    # [E]
prob_mean = scores.mean(dim=0)                                       # [E]
aux_loss = (load * prob_mean).sum() * num_experts * router_aux_loss_coef
```

**作用**：鼓励各专家接收的 token 数量均匀分布，防止路由器"偏科"。

**典型值**：`router_aux_loss_coef = 5e-4`，`num_experts = 4`

在训练日志中，总损失被拆分为：
```
loss = logits_loss + aux_loss
     ↓
current_loss = res.loss + res.aux_loss
current_logits_loss = current_loss - current_aux_loss    # 仅 lm_head 的 CE/策略损失
current_aux_loss = res.aux_loss                           # 仅 MoE 路由损失
```

---

---

## 各阶段超参数对比

### RoPE base 是否调整？

**没有。** 所有训练阶段均使用 `rope_theta=1e6`（标准值），`inference_rope_scaling`（YaRN，factor=16）仅用于推理时的长文本外推（`eval_llm.py` 的 `--inference_rope_scaling` 参数），**训练阶段完全不开启**：

```python
# MiniMindConfig 默认值（所有训练阶段共用）
self.rope_theta = 1e6            # 训练时不变
self.inference_rope_scaling = False  # 训练时始终为 False
# YaRN scaling 只在 eval_llm.py 中通过 --inference_rope_scaling 开启
```

### 各阶段 max_seq_len 变化

这是各阶段之间**最关键的架构差异**——序列长度随训练阶段递增：

| 阶段 | max_seq_len | 说明 |
|------|-------------|------|
| 预训练 | **340** | 短文本，降低计算量 |
| SFT / LoRA / 蒸馏 | **768** | 标准对话长度 |
| DPO | **1024** | chosen/rejected 对更长 |
| GRPO / PPO | **768 + 1024 = 1792** | prompt 最多 768 + 生成最多 1024 |
| Agent RL | **1024 + 768 = 1792** | prompt 最多 1024 + 生成最多 768 |

注意 GRPO/PPO/Agent RL 的 `max_seq_len` 是 `prompt_len + max_gen_len`，传入 `MiniMindConfig` 的 `max_seq_len` 参数。但 `max_position_embeddings` 全程保持 `32768` 不变（在 `MiniMindConfig` 中硬编码），训练长度远低于此值，所以不需要调整 RoPE。

#### "prompt 768" 的具体含义：不足 768 会 padding 吗？

**会左 padding，但"768"是上限而非目标长度。** 实际代码逻辑：

```python
# GRPO 中对 prompt 的处理
prompt_inputs = tokenizer(
    prompts,
    padding=True,              # ① batch 内左 padding 到最长 prompt
    padding_side="left",       #    左 padding 保证生成时右对齐
    add_special_tokens=False,
)

prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
# ② 截断：只取最后 768 个 token，超长的 prompt 被砍掉前半部分
```

具体行为取决于 prompt 的实际长度：

```
场景 1: prompt 很短（100 token）
  ← PAD...PAD(668个) ──────── 实际 prompt 内容(100个) →     ← 生成 1024 个 token →
  ├─────────────── 768 ───────────────┤ ← 实际内容只有 100，[:, -768:] 是无操作的
                                      │
场景 2: prompt 刚好 768
  ─────────────── 实际 prompt 内容(768个) ───────────────    ← 生成 1024 个 token →
  ├─────────────── 768 ───────────────┤ ← 全部保留
                                      │
场景 3: prompt 很长（1500 token）
  ───────── 被截掉的前 732 个 token ─────── 实际 prompt 内容(后768个) → ← 生成 1024 个 token →
                                          ├──── 768 ────┤ ← [:, -768:] 只保留最后 768
```

**为什么用左 padding + 截断尾部？**

1. **左 padding**：`padding_side="left"` 保证所有有效 token 右对齐。这对于自回归生成至关重要——模型从序列的**最右侧**开始生成，如果 padding 在右边，模型会先生成 padding token

左 padding 和右 padding 的对比：
```
batch 里有两条 prompt：
  A: "请写一首诗"                   → 5 个 token
  B: "请用一句话描述春天的美丽景色"   → 12 个 token

右 padding（错误 ❌）:
  A: [t0, t1, t2, t3, t4, PAD, PAD, PAD, PAD, PAD, PAD, PAD]
                          ↑ 模型从 PAD 后面开始生成 → ❌ 先产生无意义输出
  B: [t0, t1, t2, t3, t4, t5,  t6,  t7,  t8,  t9,  t10, t11]
                          ↑ 正常生成

左 padding（正确 ✅）:
  A: [PAD, PAD, PAD, PAD, PAD, PAD, PAD, t0, t1, t2, t3, t4]
                                                   ↑ 模型从有效内容末尾开始生成
  B: [t0, t1, t2, t3, t4, t5,  t6,  t7,  t8,  t9,  t10, t11]
                                                   ↑ 正常生成
```

2. **截断尾部（取最后 768）**：对话中越靠后的内容越重要（如最近的 user 消息），截断前半部分比截断后半部分损失更少的关键信息
3. **prompt 768 是上限**：prompt 可以短到几十个 token（如"请写一首诗"），不会强制填充到 768。只有**跨样本的 batch 内**才会 padding 到该 batch 的最长 prompt 长度

**生成阶段**：`model.generate(max_new_tokens=1024)` 在 prompt 后面追加最多 1024 个新 token。最终模型输入的总长度 = prompt 长度（≤768）+ 实际生成的 token 数（≤1024）。这 1024 是 **max_new_tokens**——模型可能在生成到第 50 个 token 时就输出 EOS 提前停止，不会强制生成 1024 个。

### RL 阶段生成长度远超 SFT，会不会导致 rollout 质量差？

**这是一个真实存在的问题，但有几个缓解因素：**

#### 问题：各阶段训练位置覆盖范围

```
位置编码 0                 340                768               1792
         ├──────────────────┤
         预训练（max=340）

         ├──────────────────────────────────┤
         SFT（max=768）

         ├──────────────────────────────────────────────────────────┤
         GRPO rollout（prompt ≤768 + generation ≤1024 = ≤1792）
                             └──── 如果 prompt 正好 768，生成
                                   位置从 768 开始，全部超出 SFT
                                   见过的位置范围 ────┘
```

预训练只见过位置 0~340，SFT 只见过位置 0~768。如果 prompt 较短（如 100 token），生成从位置 100 开始，**仍落在 SFT 见过的范围内**。但最坏情况下 prompt 截断到 768，生成从位置 **768** 开始，此后所有生成位置**模型从未在任何训练阶段见过**。

#### 缓解因素 ①：RoPE 的相对位置编码特性

RoPE 编码的是**相对位置**而不是绝对位置。当模型在位置 800 预测 token 时，它依赖的是 `q·k` 的点积，而 RoPE 使这个点积只与**相对距离** `m-n` 有关：

```python
# RoPE 的 q·k 点积只依赖于相对位置差
<q_m, k_n> = f(q, m) · f(k, n) = g(q, k, m-n)
```

这意味着**模型不需要外推绝对位置**，只需要泛化到更大的**相对距离**。预训练和 SFT 中见过的相对距离范围是 [0, 767]，GRPO rollout 需要 [0, 1791]。虽然超出了，但 RoPE 的 **long-term decay** 特性（长距离注意力自然衰减）让超出部分的影响有限。

#### 缓解因素 ②：max_position_embeddings 预计算

```python
# 模型初始化时就已经预计算了 32768 个位置的 RoPE
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=config.head_dim,
    end=config.max_position_embeddings,  # = 32768
    rope_base=config.rope_theta,         # = 1e6
)
```

位置 768~1791 的 RoPE 值在模型初始化时就已计算好，不需要外推或插值，只是**模型没在这些位置上训练过**。

#### 缓解因素 ③：SFT 的 max_seq_len 设置

```
SFT 的 max_seq_len = 768
                   = prompt(768) 的极限
                   = 模型需要回答的 token 位置刚好到 768
```

在 SFT 阶段，一条完整样本是 `prompt + response`。如果 prompt 占 500、response 占 200，模型实际训练到的最大位置是 700。GRPO 的 **prompt 也被截断到 768**，模型从位置 768 开始生成——这意味着 **prompt 本身仍落在 SFT 见过的区间内**，出问题的是 model.generate() 生成的尾巴。

#### 缓解因素 ④：推理时的 YaRN scaling（可选）

如果 rollout 质量确实因长序列下降，可以使用 SGLang 引擎并在启动服务时开启 YaRN。但训练阶段默认不开启，因为 YaRN 是在推理时动态调整 RoPE 频率，**不影响模型参数**。

---

### YaRN（Yet another RoPE extensioN）原理详解

YaRN 是一种**推理时**的位置编码外推方法，通过**按维度选择性拉伸 RoPE 频率**来让模型处理比训练时更长的序列。MiniMind 的实现位于 `model_minimind.py` 的 `precompute_freqs_cis()` 中。

#### 核心问题

RoPE 的标准频率公式为：

$$\omega_i = \frac{1}{base^{2i/d}} \quad (i = 0, 1, ..., d/2-1)$$

其中 $base=10^6$，$d$ 为 head_dim。训练时的最大位置是 $L_{train}$，RoPE 频率在区间 $[0, L_{train})$ 上表现良好。当推理位置 $L_{test} \gg L_{train}$ 时，高频维度（$i$ 较小）对超出训练范围的位置产生错误的 `cos/sin` 值。

#### YaRN 的解决方案：按维度分段的频率拉伸

YaRN 不对所有维度做相同的缩放，而是**按频率高低分两段处理**：

```python
def precompute_freqs_cis(dim, end=32768, rope_base=1e6, rope_scaling=None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2] / dim))
    
    if rope_scaling is not None:  # ← YaRN 从这里开始
        orig_max = rope_scaling["original_max_position_embeddings"]  # 2048
        factor = rope_scaling["factor"]                                # 16
        beta_fast, beta_slow = rope_scaling["beta_fast"], rope_scaling["beta_slow"]
        # 32, 1
        
        # ① 计算两个边界维度
        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        low = max(math.floor(inv_dim(beta_fast)), 0)   # 高频边界
        high = min(math.ceil(inv_dim(beta_slow)), dim//2-1)  # 低频边界
        
        # ② 构建线性插值斜坡
        ramp = torch.clamp((arange - low) / (high - low), 0, 1)
        
        # ③ 频率拉伸：高频不变，低频拉伸
        freqs = freqs * (1 - ramp + ramp / factor)
    
    return freqs
```

#### 分三步理解

**Step 1：计算两个边界维度**

$inv\_dim(b) = \frac{d \cdot \ln(orig\_max/(b \cdot 2\pi))}{2 \cdot \ln(rope\_base)}$

| 参数 | 含义 | MiniMind 值 |
|------|------|------------|
| $\beta_{fast}=32$ | 高频阈值（周期 ≤ 32 的维度） | `low` = 维度 4 |
| $\beta_{slow}=1$ | 低频阈值（周期 ≥ 1 的维度） | `high` = 维度 21 |
| $orig\_max=2048$ | 训练时的最大位置 | 用于计算哪些维度的 RoPE 周期超出训练范围 |

高频维度（$i < low$）：波长较短，周期远小于 $L_{train}$，**不做任何处理**。
中频维度（$low \leq i \leq high$）：波长逐渐接近 $L_{train}$，逐步插值到拉伸频率。
低频维度（$i > high$）：波长超过 $L_{train}$，**完全拉伸**，频率降低为 $1/factor$。

**Step 2：构建斜坡函数**

```python
ramp = clamp((维度索引 - low) / (high - low), 0, 1)
```

```
ramp
 ↑
1 │              ───────────
   │            ╱
   │          ╱
   │        ╱
0 │───────
   └──────────────────────→ 维度索引 i
          low         high
   (高频，不拉伸)   (低频，完全拉伸)
```

**Step 3：频率拉伸**

$$\omega_i' = \omega_i \cdot (1 - ramp + \frac{ramp}{factor})$$

- 当 $ramp=0$（高频维度）：$\omega_i' = \omega_i$，频率不变 ✅
- 当 $ramp=1$（低频维度）：$\omega_i' = \omega_i / factor$，频率降低 16 倍，周期延长 16 倍
- 当 $0<ramp<1$（中频维度）：$\omega_i' = \omega_i \cdot ((1-ramp) + ramp/factor)$，线性插值

#### 为什么按维度分段？

下图展示不同维度 $i$ 的 RoPE 周期 $T_i = 2\pi / \omega_i$：

| 维度 $i$ | 周期 $T_i$ | 分类 | 训练位置 2048 内见过几个周期？ | YaRN 处理 |
|---------|-----------|------|---------------------------|----------|
| 0 | $2\pi \times base^{0} \approx 6.28$ | 高频 | 326 个周期 ✅ | 不变 |
| 4 | $2\pi \times base^{4/96} \approx 28$ | 中-高频 | 73 个周期 ✅ | 轻微拉伸 |
| 21 | $2\pi \times base^{21/96} \approx 1500$ | 中-低频 | 1.4 个周期 ⚠️ | 开始拉伸 |
| 47（最后）| $2\pi \times base^{47/96} \approx 10^6$ | 低频 | 0.002 个周期 ❌ | 完全拉伸 |

高频维度在训练范围内见过足够多的周期，模型已经学会了其模式，不需要改变。低频维度在训练范围内连一个完整周期都没走完，模型没有学会其规律，需要拉伸来避免在长序列中产生错误的值。

#### 注意力缩放因子

除了频率拉伸，YaRN 还调整了注意力 softmax 的温度：

```python
attn_factor = rope_scaling.get("attention_factor", 1.0)
freqs_cos = concat(cos(freqs), cos(freqs)) * attn_factor
freqs_sin = concat(sin(freqs), sin(freqs)) * attn_factor
```

拉伸频率后，RoPE 点积的幅度会下降，通过 $attn\_factor$ 补偿，保持注意力分布的锐度。

#### 与直接插值的区别

| 方法 | 处理方式 | 高频维度 | 低频维度 | 效果 |
|------|---------|---------|---------|------|
| **PI（Position Interpolation）** | 所有维度统一缩放 | ❌ 高频被过度拉伸 | ✅ 低频正确 | 短距离 token 区分度下降 |
| **NTK-aware** | 按维度指数缩放 | 不变 | 拉伸 | 高频保留但低频拉伸不足 |
| **YaRN（本实现）** | 分三段线性过渡 | 不变 | 拉伸 | **兼顾长短距离** ✅ |
| **不处理** | 什么都不做 | ✅ | ❌ 超出范围后产生混乱 | 长序列质量差 |

#### 在 MiniMind 中的实际使用

MiniMind 没有在训练时启用 YaRN，只在推理时通过 `--inference_rope_scaling` 开启：

```python
# eval_llm.py
MiniMindConfig(
    inference_rope_scaling=args.inference_rope_scaling,  # 默认为 False
)
```

```bash
# 推理时启用（4倍外推）
python eval_llm.py --inference_rope_scaling --max_new_tokens 8192
```

开启后 `max_position_embeddings=32768` 结合 `factor=16`，可以实现约 $2048 \times 16 \approx 32K$ 的有效位置编码外推（因为低频维度被拉伸了 16 倍）。训练阶段不开启是因为 YaRN 只是推理时的频率调整技巧，**不参与训练就不会影响模型参数**，RL 阶段如果在 rollout 时开启 YaRN，可以提升长序列的生成质量但又不需要修改 SFT 权重。

#### 实际影响

| 因素 | 影响方向 |
|------|---------|
| 缺少长序列训练 | ❌ 生成质量可能下降 |
| RoPE 相对位置编码 | ✅ 部分缓解 |
| RoPE long-term decay | ✅ 长距离注意力自然衰减，错误不易累积 |
| prompt 被截断到 768 | ✅ 模型熟悉 prompt 区间 |
| SFT 见过 prompt+response 混合 | ⚠️ 部分缓解，但 response 通常很短 |
| 预训练见过更多样化的文本分布 | ✅ 有助于泛化 |

**实践中**，对于 MiniMind 这种规模（~42M 参数）的模型，生成长度从 768 扩展到 1792 通常不会导致灾难性退化，因为：
1. RoPE 对相对位置的良好泛化性
2. 长距离的 attention 权重本来就很小（`softmax` 的 long-term decay）
3. 生成的前几步（位置 768~800）仍接近 SFT 见过的区间
4. 最远的 token（~1791）主要在 EOS 附近，模型倾向于先输出 EOS 结束

但如果模型尺寸更小（如 10M）或生成长度更大（如 4096+），这个问题的确会显著影响 rollout 质量，此时应考虑在 SFT 阶段使用更长的 `max_seq_len` 或启用 YaRN scaling。

### 学习率变化

| 阶段 | 默认学习率 | 与预训练的比例 |
|------|-----------|---------------|
| 预训练 | **5e-4** | 1×（基准） |
| SFT | **1e-5** | **1/50** |
| LoRA SFT | **1e-4** | **1/5**（比全量 SFT 大 10 倍） |
| 蒸馏 | **5e-6** | **1/100** |
| DPO | **4e-8** | **1/12500** |
| GRPO / Agent RL / PPO | **3e-7** | **1/1667** |

### 其他关键超参数对比

| 参数 | 预训练 | SFT | LoRA | 蒸馏 | DPO | GRPO | PPO | Agent RL |
|------|--------|-----|------|------|-----|------|-----|---------|
| `--epochs` | 2 | 2 | 10 | 6 | 1 | 1 | 1 | 1 |
| `--batch_size` | 32 | 16 | 32 | 32 | 4 | 2 | 2 | 2 |
| `--accumulation_steps` | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `--learning_rate` | 5e-4 | 1e-5 | 1e-4 | 5e-6 | 4e-8 | 3e-7 | 3e-7 | 3e-7 |
| `--warmup_ratio` | 0 | — | — | — | — | — | — | — |
| `--grad_clip` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `--dtype` | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| `--max_seq_len` | 340 | 768 | 340 | 340 | 1024 | 768+1024 | 768+1024 | 1024+768 |
| LR scheduler | Cosine+warmup | Cosine | Cosine | Cosine | Cosine | CosineAnnealing | CosineAnnealing | CosineAnnealing |

### 学习率变化的原因

```
预训练 5e-4  ────────────────────────────────────────────────────────────────── 最大
  │
  │  预训练需要大学习率：从零开始训练，快速收敛
  │
  ▼
SFT 1e-5    ─────── 比预训练小 50 倍 ──── 微调阶段，只需在已有知识上调整，过大学习率会破坏预训练知识
  │
  │  LoRA 1e-4 比 SFT 大 10 倍：只更新 ~1% 参数，梯度信号弱，需要更大 LR
  │
  ▼
蒸馏 5e-6   ─────── 比 SFT 还小 ────── 学生需要稳定地拟合教师分布，小学习率更平稳
  │
  ▼
DPO 4e-8   ─────── 极小 ──────────── 偏好优化极其敏感，注释明确说"建议<=5e-8避免遗忘"
  │
  ▼
RL 3e-7    ─────── 接近 DPO ──────── 强化学习的 KL 惩罚已经约束策略变化，学习率不能太大
```

---

## 训练流程建议顺序

根据项目的默认配置和权重依赖关系，推荐的训练流程为：

```
Tokenizer (可选)
    │
    ▼
预训练 (Pretrain)        ─── 从零开始训练，产出 pretrain 权重
    │
    ▼
全量 SFT                 ─── 基于 pretrain 权重，产出 full_sft 权重
    │
    ├──► LoRA SFT         ─── 基于 full_sft 权重，产出 lora 权重（可选）
    ├──► 知识蒸馏          ─── 基于 full_sft 权重，产出蒸馏后权重（可选）
    ├──► DPO              ─── 基于 full_sft 权重，产出 dpo 权重
    ├──► GRPO / Agent RL  ─── 基于 full_sft 权重，产出 grpo/agent 权重
    └──► PPO              ─── 基于 full_sft 权重，产出 ppo 权重
```

---

*本报告基于 MiniMind 项目代码自动分析生成。*
