# Agent RL / Rollout / RoPE / WandB 打点总结（MiniMind）

本文总结本仓库在不同训练/推理脚本中：

- rollout 时 prompt 是否会 padding、padding 到什么长度
- 为什么不同训练方法的 padding 行为不一致
- GRPO/PPO 的 left padding 对 RoPE 位置计数的影响
- SFT 阶段长度（~700）与 RL rollout 总长度变长的关系与风险

> 代码定位说明：文中引用格式为 `file_path:line`。

---

## 1. Rollout 时 prompt 会被 padding 到固定长度吗？

结论：**不会 padding 到“全局固定长度”（比如永远补到某个常数 max_seq_len）**；但在**批量 rollout**的脚本里，会为了拼 batch 做 **padding 到“本 batch 的最长长度”（动态长度）**。

### 1.1 Agent RL（多轮工具调用）

- `rollout_single()` 每次只对一个样本的对话上下文做 tokenize（batch=1），未开启 `padding=True`，因此 rollout 阶段**基本不需要 padding**。
  - 见：`trainer/train_agent.py:119-130`
- 真正进入训练 forward 前，会把 `prompt_ids + response_ids` 打包后，为组成张量，按本 batch 的 `max_len` 手动 padding。
  - 见：`trainer/train_agent.py:411-418`

### 1.2 GRPO / PPO（批量 prompts 的 RL）

- 在 rollout 前先对 `prompts: list[str]` 批量 tokenize，为形成 `[B, L]` 张量，必须 `padding=True`。
  - GRPO：`trainer/train_grpo.py:74-75`
  - PPO：`trainer/train_ppo.py:86-88`
- 这里的 `padding=True` 是**pad 到本 batch 最长**，不是 `padding="max_length"` 那种固定长度。
- 通常选择 `padding_side="left"`，方便后续按 `prompt_lens` 构造 completion 的 logprob 位置索引。
  - GRPO logp 位置对齐：`trainer/train_grpo.py:91-94`
  - PPO logp 位置对齐：`trainer/train_ppo.py:121-123`

### 1.3 SGLang rollout

即使上游做了 left padding，SGLang 引擎在发请求前会根据 `attention_mask` **剔除 padding token**，只把有效 token 发送给服务端。

- 见：`trainer/rollout_engine.py:108-113`

---

## 2. 为什么这几种训练方法 padding 行为不一致？

核心原因不是“算法必须不同”，而是**实现的控制流与对齐需求不同**：

1. **Agentic RL** 是多轮对话/工具交互：每一轮都要先生成，再解析 `<tool_call>`，执行工具后把结果拼回上下文，决定下一轮 prompt。
   - 每个样本的轮数/路径不同，batch 内很难锁步执行，工程上通常先用“单样本 rollout”保证正确性与可调试性。
   - 见工具交互控制流：`trainer/train_agent.py:143-164`
2. **GRPO/PPO** 的 rollout 更像“批量 prompt → 批量生成”，控制流整齐、每个样本结构一致，天然适合 batch 推理，因此会在 tokenize 阶段做 padding。
   - 见：`trainer/train_grpo.py:74-86`、`trainer/train_ppo.py:84-96`

---

## 3. GRPO/PPO 的 left padding 时，RoPE 位置是怎么计数的？

结论：在当前 MiniMind 实现里，**RoPE 位置是从张量位置 0 开始计数（包含左侧 padding 的位置）**，而不是“从第一个非 pad 的有效 token 重新从 0 开始”。

原因：MiniMind 的 RoPE `cos/sin` 是直接按序列长度切片，不依赖 `attention_mask` 或 `position_ids`：

- `position_embeddings = (freqs_cos[start_pos:start_pos + seq_length], ...)`
  - 见：`model/model_minimind.py:228-239`

`attention_mask` 仅用于把注意力分数中 padding 位置打成 `-1e9`，防止关注 padding token：

- 见：`model/model_minimind.py:128-131`

因此：**left padding 会让“有效 token 的绝对 position index”整体向右平移**；但在 RoPE 机制下通常不致命（相对位置关系仍可学习），真正的硬风险来自：总长度是否超过 RoPE 表上限。

---

## 4. 会不会出现“rollout 后总长度超过训练长度”的问题？

### 4.1 “超过 SFT 的 700 左右”是可能的

SFT 阶段的 `max_seq_len≈768`（你说的 700 左右）通常是**训练数据截断长度**，例如：

- `SFTDataset(..., max_length=args.max_seq_len)`
  - 见：`trainer/train_full_sft.py:175,276`

RL 阶段（例如 GRPO）常见会设置：

- prompt 上限 `args.max_seq_len`
- 生成上限 `args.max_gen_len`
- 总长度预算 `args.max_seq_len + args.max_gen_len`

GRPO 脚本里明确把模型 config 的 `max_seq_len` 设为 `prompt+gen`：

- `MiniMindConfig(..., max_seq_len=args.max_seq_len + args.max_gen_len, ...)`
  - 见：`trainer/train_grpo.py:253-254`

所以 RL rollout 的总长度**很可能**大于 SFT 的 700。

### 4.2 这会不会“报错/越界”？

不一定。

- MiniMind 的 RoPE 表上限由 `max_position_embeddings` 控制，默认 **32768**：
  - 见：`model/model_minimind.py:27`
- 只要 RL 的 `prompt_len + gen_len` 不超过 `max_position_embeddings`，前向不会因为位置越界而崩。

### 4.3 真正的风险：分布外与长上下文能力不足

即便不越界，SFT 只在较短长度上训练过，RL 阶段突然让模型在更长长度学习，常见风险：

- 长上下文生成质量/一致性下降
- KL/优势估计更不稳定
- 训练更容易出现“长样本主导显存/吞吐”，以及 reward 变得更噪

如果你确实要学长上下文，通常需要：

- 提高 SFT 阶段的训练截断长度，或
- 在 RL/推理阶段使用一致的 RoPE 外推策略（仓库里有 `inference_rope_scaling` 相关开关，Agent RL 参数中有体现）
  - 见：`trainer/train_agent.py:677-712`

---

## 5. 一句话建议（配置层面）

1. 如果想让 RL 尽量贴近 SFT：控制 `max_seq_len + max_gen_len` 接近 SFT 的训练长度。
2. 如果要做长输出 RL：确保不超过 `max_position_embeddings`，并考虑统一开启/校准 RoPE 外推策略（policy/ref/rollout 一致）。
3. 使用 SGLang 引擎时，上游即使 left pad，也会被服务端请求前剔除 padding（减少“位置编号被 padding 浪费”）。

---

## 6. Agent RL（train_agent.py）整体流程是怎么做的？

核心是 **“多轮对话 rollout（可能多次工具调用）→ 对整段轨迹打一个 reward → 组内标准化 advantage → token 级策略更新（GRPO/CISPO）”**。

### 6.1 多轮 rollout（可能多次工具调用）

入口：`rollout_single()`（`trainer/train_agent.py:109`）

每个样本最多跑 `max_turns` 轮：

1) 拼接当前 messages 为 prompt：`tokenizer.apply_chat_template(...)`（`trainer/train_agent.py:119`）
2) 调模型生成当前轮 assistant 输出（可能包含 `<tool_call>...</tool_call>`）：`rollout_engine.rollout(...)`（`trainer/train_agent.py:124-130`）
3) 若解析到 tool call：执行 mock 工具，把 tool result 作为 `role=tool` 追加回 messages（`trainer/train_agent.py:143-155`），再进入下一轮。
4) 若没有 tool call：认为对话结束，提前 break（`trainer/train_agent.py:143-145`）。

> 另外：每轮生成的 token 会被累积到 `response_ids/response_old_logps`，并用 `response_mask` 区分“模型生成 token(1)”与“环境/工具观察 token(0)”（`trainer/train_agent.py:112-164`）。

### 6.2 训练侧输入如何拼接

rollout 完后会把每条样本拼成：

- `ids = prompt_ids + response_ids`
- `mask = [0]*len(prompt_ids) + response_mask`（prompt 区域不参与 loss）

并在组成 batch 时按 batch 内 `max_len` 做右 padding（`trainer/train_agent.py:265-281`）。

### 6.3 优势函数与策略更新

1) 对每个 prompt 采样 `num_generations` 条（`rollout_batch()` 里循环生成，`trainer/train_agent.py:170-191`）
2) reward reshape 成 `[B, G]`，按组做 mean/std 标准化，得到 advantage（`trainer/train_agent.py:460-463`）
3) 计算 token 级 ratio / KL，并按 `loss_type` 选择：

- CISPO：只对 ratio 做上界截断 `clamp(max=epsilon_high)`（`trainer/train_agent.py:468-472`）
- GRPO：PPO clip 到 `[1-eps, 1+eps]`（`trainer/train_agent.py:473-476`）

最后只在 `completion_mask`（回答 token）区域聚合 loss（`trainer/train_agent.py:451-480`）。

---

## 7. Reward 是“每轮对话算一次”还是“多轮结束算一次”？

结论：**每条样本（一次多轮 rollout 轨迹）只在最后计算一次 reward**。

证据：

- rollout 阶段会把所有轮的输出保存为 `turn_outputs`（`trainer/train_agent.py:110-168`），同时返回 `unfinished`。
- reward 阶段在 `calculate_rewards()` 中，对每个最终 completion 计算一个标量 reward（`trainer/train_agent.py:199-249`），并且会把 **所有轮的文本**拼起来解析 tool call：
  - `for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))`（`trainer/train_agent.py:211-212`）

所以它不是逐轮打分（没有 per-turn reward），而是 **episode-level reward**（用整条轨迹的行为来打分）。

---

## 8. train_agent 的 wandb/swanlab 打点说明（都代表啥）

日志构造位置在 `trainer/train_agent.py:483-630`。

### 8.1 核心训练信号

- `reward`：当前 step 下所有生成样本 reward 的均值（`trainer/train_agent.py:487-489`；reward 来自 `calculate_rewards()`：`trainer/train_agent.py:199-249`）。
- `policy_loss`：策略 loss（已经乘回 accumulation_steps）（`trainer/train_agent.py:486-488`）。
- `kl_ref`：`(ref_logp - policy_logp)` 在回答 token 区域的加权平均（`trainer/train_agent.py:489`）。
- `learning_rate`：当前学习率。

### 8.2 回答长度与终止

- `avg_response_len`：回答 token 数均值（只算 response，不含 prompt；按 eos 截断后统计），由 `token_counts = completion_mask.sum(...)` 得到（`trainer/train_agent.py:431-438,488`）。
- `eos/rate`：回答中是否出现 eos 的比例（`trainer/train_agent.py:507`）。

### 8.3 Reward 分布

- `reward/std`、`reward/p50`、`reward/p90`、`reward/min`、`reward/max`：用于判断 reward 是否“少数特别好/特别坏”或整体在涨（`trainer/train_agent.py:495-501,546-566`）。

### 8.4 KL / ratio / clip（稳定性）

- `kl_div/*`：token 级 `(ref_logp - policy_logp)` 在回答区间的统计。
- `per_token_kl/*`：`exp(kl_div) - kl_div - 1` 的统计（对应 loss 里 KL 惩罚项）。
- `ratio/*`：`exp(policy_logp - old_logp)` 的统计。
- `clip/frac`：
  - CISPO：`ratio > epsilon_high` 的比例（`trainer/train_agent.py:518`）
  - GRPO：`ratio` 超出 `[1-eps, 1+eps]` 的比例

### 8.5 工具调用健康度（Agent RL 最关键）

这部分由 `_compute_tool_health_metrics()` 统计（`trainer/train_agent.py` 内部函数），主要包含：

- `tools/tool_call_rate`：有 tool call 的样本占比
- `tools/invalid_tool_call_rate`：工具名不在允许集合的 call 占比
- `tools/arg_invalid_call_rate`：参数校验失败的 call 占比（用 `CHECK_ARGS`，`trainer/train_agent.py:76-84`）
- `tools/valid_call_rate`：完全合法的 call 占比
- `tools/tool_gap_mean`：合法调用数与 gt 需求的差距均值（与 reward 里对齐逻辑一致）
- `tools/unfinished_rate`：达到 max_turns 仍未结束的样本占比
- `tools/gt_hit_rate`：最终答案命中 gt 的比例

以及 top 工具频次以数值形式展开为：

- `tools/top/<tool_name> = count`（避免 swanlab 对 string scalar 报错）

### 8.6 数值健康

- `loss/total`：`policy_loss + aux_loss`
- `loss/aux`：MoE 的 aux loss
- `grad/norm`：梯度 L2 范数
- `cuda/mem_allocated_gb` / `cuda/mem_reserved_gb`：显存

### 8.7 抽样样例

- `samples/best` / `samples/worst`：当前 step 内 reward 最高/最低的样本文本（含 gt、unfinished、prompt/completion 截断片段），用于肉眼排查（`trainer/train_agent.py:606-630`）。

---

## 9. YaRN（RoPE scaling）在 Agent RL 是否需要开？

Agent RL 已加开关：`--inference_rope_scaling`（`trainer/train_agent.py:677`），传入 `MiniMindConfig(..., inference_rope_scaling=...)`（`trainer/train_agent.py:706-712`）。

注意：如果要开，**policy/ref/rollout 必须一致**，否则 `old_logps / policy_logps / ref_logps` 不是同一个 RoPE 设定，ratio/KL 会失真。

YaRN 默认参数在 `model/model_minimind.py:31-39`。

---

## 10. RoPE 位置到底从 padding 0 开始还是从第一个有效 token 开始？

在 MiniMind 模型实现里，RoPE 位置是按序列 index 切片：

- `position_embeddings = freqs_[start_pos:start_pos + seq_length]`（`model/model_minimind.py:228-239`）

它 **不使用 attention_mask 生成 position_ids**，因此：

- 如果 input_ids 左侧真的有 padding token，那有效 token 的“绝对 position index”会整体右移。
- 但在 Agent RL 的 torch rollout 中，`rollout_single()` 是 batch=1 tokenize（未启用 `padding=True`），通常不会引入左 padding（`trainer/train_agent.py:119-130`）。
- 若使用 SGLang 引擎，即使上游 left pad，也会在发请求前剔除 padding，只保留有效 token（`trainer/rollout_engine.py:108-113`）。

---

## 11. “一轮/turn”在不同 RL 训练脚本里各指什么？

### 11.1 Agentic RL（train_agent.py）里的“turn”

在 agentic RL 里，turn 是一个**控制流概念**：

- 每一轮："基于当前 messages 生成一次 assistant 输出"。
- 若输出包含 `<tool_call>...</tool_call>`：则会执行工具，把工具结果追加回 messages，然后进入下一轮。
- 若不包含 tool call：认为已经给出最终回答，提前结束。

对应代码：

- turn 循环：`for turn in range(max_turns)`（`trainer/train_agent.py:118`）
- 结束条件（无 tool call）：`if not calls: break`（`trainer/train_agent.py:143-145`）
- 达到最大轮数：`unfinished = turn == max_turns - 1`（`trainer/train_agent.py:145-147`）

因此：agentic RL 的 episode 长度是**可变 turn 数**，由“是否继续调用工具”决定，并受 `max_turns` 上限截断。

#### 11.1.1 当前最多 rollout 多少轮？

当前 `train_agent.py` 在训练时把 `max_turns` **写死为 3**：

- 调用：`rollout_batch(..., max_turns=3, ...)`（`trainer/train_agent.py:410-412`）
- 循环：`for turn in range(max_turns)`（`trainer/train_agent.py:118`）

因此每个样本（一次多轮轨迹）最多经历 3 次“生成→（可选）工具交互”的 turn。

### 11.2 非 Agent GRPO（train_grpo.py）里的“一轮”

在非 agent 的 GRPO 脚本中没有 turn 循环，训练形式是：

- 一次性：`prompts -> rollout_engine.rollout(...) -> completions`。
- 然后对每条 completion 计算 reward，并做一次策略更新。

所以如果要类比“一轮/episode”，可以认为：

- **每次 rollout 得到的一条 completion（prompt + response）就是一条轨迹/一个 episode**。
- `num_generations` 表示对同一个 prompt 采样多条轨迹，用于组内标准化 advantage（`trainer/train_grpo.py:121-125`）。

对应代码：

- batch tokenize/padding：`tokenizer(prompts, ..., padding=True, padding_side="left")`（`trainer/train_grpo.py:74-75`）
- 一次性 rollout：`rollout_engine.rollout(...)`（`trainer/train_grpo.py:80-86`）
- 一次性 reward：`rewards = calculate_rewards(prompts, completions, ...)`（`trainer/train_grpo.py:95`）
