#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

###############################################################################
# 统一配置区：torch rollout 版本（不依赖 sglang）
###############################################################################

AUTO_STOP_TORCHRUN="${AUTO_STOP_TORCHRUN:-1}"

# --- 一个字段对齐权重/结构 ---
MODEL_TAG="${MODEL_TAG:-MiniMind-Full-SFT-DSsft_t2t-L8-H768-S768-MoE4K1-BS128-GA1-LR1e-05-Ep2-P198p4M-A63p9M}"

# --- 训练超参（train_agent.py）---
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-7}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
BETA="${BETA:-0.1}"
LOSS_TYPE="${LOSS_TYPE:-cispo}"
EPSILON="${EPSILON:-0.2}"
EPSILON_HIGH="${EPSILON_HIGH:-5.0}"
FROM_RESUME="${FROM_RESUME:-0}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_GEN_LEN="${MAX_GEN_LEN:-768}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-2500}"

# 结构参数：可以不填，让脚本从 MODEL_TAG 解析（后面会自动解析补齐）
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-}"
HIDDEN_SIZE="${HIDDEN_SIZE:-}"
USE_MOE="${USE_MOE:-}"

# --- 数据 ---
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/agent_rl.jsonl}"

TORCHRUN_PIDFILE="${TORCHRUN_PIDFILE:-$ROOT_DIR/trainer/torchrun.pid}"

###############################################################################
# 自动推导/解析区（一般不需要改）
###############################################################################

if [[ -z "${NUM_HIDDEN_LAYERS}" ]] && [[ "${MODEL_TAG}" =~ -L([0-9]+) ]]; then
  NUM_HIDDEN_LAYERS="${BASH_REMATCH[1]}"
fi
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-8}"

if [[ -z "${HIDDEN_SIZE}" ]] && [[ "${MODEL_TAG}" =~ -H([0-9]+) ]]; then
  HIDDEN_SIZE="${BASH_REMATCH[1]}"
fi
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"

if [[ -z "${USE_MOE}" ]] && [[ "${MODEL_TAG}" =~ -MoE([0-9]+) ]]; then
  if [[ "${BASH_REMATCH[1]}" == "0" ]]; then
    USE_MOE=0
  else
    USE_MOE=1
  fi
fi
USE_MOE="${USE_MOE:-0}"

TORCHRUN_STARTED_BY_SCRIPT=0

cleanup() {
  if [[ "${AUTO_STOP_TORCHRUN}" != "1" ]]; then
    return 0
  fi
  if [[ "${TORCHRUN_STARTED_BY_SCRIPT}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${TORCHRUN_PIDFILE}" ]]; then
    local tpid
    tpid="$(cat "${TORCHRUN_PIDFILE}" || true)"
    if [[ -n "${tpid}" ]] && kill -0 "${tpid}" 2>/dev/null; then
      echo "[agent.sh] 脚本退出，停止 torchrun(pid=${tpid})"
      kill -- "-${tpid}" 2>/dev/null || kill "${tpid}" 2>/dev/null || true
    fi
    rm -f "${TORCHRUN_PIDFILE}" || true
  fi
}

trap cleanup EXIT INT TERM

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${ROOT_DIR}/trainer/train_agent.py" \
  --use_wandb \
  --rollout_engine torch \
  --data_path "${DATA_PATH}" \
  --epochs="${EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --accumulation_steps="${ACCUMULATION_STEPS}" \
  --learning_rate="${LEARNING_RATE}" \
  --num_generations="${NUM_GENERATIONS}" \
  --beta="${BETA}" \
  --loss_type="${LOSS_TYPE}" \
  --epsilon="${EPSILON}" \
  --epsilon_high="${EPSILON_HIGH}" \
  --max_seq_len="${MAX_SEQ_LEN}" \
  --max_gen_len="${MAX_GEN_LEN}" \
  --max_total_len="${MAX_TOTAL_LEN}" \
  --num_hidden_layers="${NUM_HIDDEN_LAYERS}" \
  --hidden_size="${HIDDEN_SIZE}" \
  --use_moe="${USE_MOE}" \
  --from_weight="${MODEL_TAG}" \
  --from_resume="${FROM_RESUME}" &

TORCHRUN_PID=$!
echo "${TORCHRUN_PID}" >"${TORCHRUN_PIDFILE}"
TORCHRUN_STARTED_BY_SCRIPT=1
wait "${TORCHRUN_PID}"

