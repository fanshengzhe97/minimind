#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

###############################################################################
# 统一配置区：所有“需要配置的超参数/路径”都放这里（可用环境变量覆盖）
###############################################################################

# --- sglang 启停 ---
AUTO_START_SGLANG="${AUTO_START_SGLANG:-1}"
AUTO_STOP_SGLANG="${AUTO_STOP_SGLANG:-1}"

# --- 训练进程清理 ---
# 说明：torchrun 会拉起多进程（按 NPROC_PER_NODE）。为了避免你 kill 掉 agent.sh 后 torchrun 变成孤儿进程，
# 这里默认也在脚本退出时尝试停止 torchrun（杀进程组）。
AUTO_STOP_TORCHRUN="${AUTO_STOP_TORCHRUN:-1}"

# 若本地没有 Transformers 基座目录（SGLANG_MODEL_PATH），是否自动从 .pth convert 一次生成
AUTO_CONVERT_SGLANG_MODEL="${AUTO_CONVERT_SGLANG_MODEL:-1}"

# --- 一个字段对齐权重/基座 ---
# 只需要设置 MODEL_TAG（推荐），脚本会自动对齐：
# - train_agent.py 的 --from_weight
# - 首次 convert 的 TORCH_CKPT_PATH（默认从 $ROOT_DIR/out/${MODEL_TAG}.pth 推导）
# - sglang server 的 SGLANG_MODEL_PATH（Transformers 基座目录）
MODEL_TAG="${MODEL_TAG:-MiniMind-Full-SFT-DSsft_t2t-L8-H768-S768-MoE4K1-BS128-GA1-LR1e-05-Ep2-P198p4M-A63p9M}"

# --- 训练超参（train_agent.py）---
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
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

# --- 数据与 tokenizer ---
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/agent_rl.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT_DIR/model}"

# --- convert 输入（用于生成 sglang 启动的 Transformers 基座目录）---
TORCH_CKPT_PATH="${TORCH_CKPT_PATH:-}"  # 例如：$ROOT_DIR/out/xxx.pth
MAX_SEQ_LEN_EXPORT="${MAX_SEQ_LEN_EXPORT:-8192}"
USE_MOE_CONVERT="${USE_MOE_CONVERT:-}"  # 默认后面会对齐 USE_MOE

# --- sglang server 配置 ---
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SGLANG_PORT="${SGLANG_PORT:-8998}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:${SGLANG_PORT}}"

# 路径：通常无需手动配置（下面会用 MODEL_TAG 自动派生）；需要自定义时可用环境变量覆盖
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-}"
SGLANG_SHARED_PATH="${SGLANG_SHARED_PATH:-}"

SGLANG_PIDFILE="${SGLANG_PIDFILE:-$ROOT_DIR/trainer/sglang_server.pid}"
SGLANG_LOGFILE="${SGLANG_LOGFILE:-$ROOT_DIR/trainer/sglang_server.log}"

TORCHRUN_PIDFILE="${TORCHRUN_PIDFILE:-$ROOT_DIR/trainer/torchrun.pid}"

###############################################################################
# 自动推导/解析区（一般不需要改）
###############################################################################

# 目录安全化：避免 MODEL_TAG 中的 '/' 或空格影响路径
_MODEL_TAG_SAFE="${MODEL_TAG//\//_}"
_MODEL_TAG_SAFE="${_MODEL_TAG_SAFE// /_}"

# 自动推导 sglang 基座 Transformers 目录（可被环境变量 SGLANG_MODEL_PATH 覆盖）
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-$ROOT_DIR/sglang_models/${_MODEL_TAG_SAFE}}"

## 从 MODEL_TAG 自动解析关键结构参数（可用同名环境变量覆盖）
## - 期望 tag 包含类似：-L8-H768-...-MoE4K1-...
if [[ -z "${NUM_HIDDEN_LAYERS}" ]] && [[ "${MODEL_TAG}" =~ -L([0-9]+) ]]; then
  NUM_HIDDEN_LAYERS="${BASH_REMATCH[1]}"
fi
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-8}"

if [[ -z "${HIDDEN_SIZE}" ]] && [[ "${MODEL_TAG}" =~ -H([0-9]+) ]]; then
  HIDDEN_SIZE="${BASH_REMATCH[1]}"
fi
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"

# use_moe：训练脚本是 0/1；tag 里一般是 MoE{experts}K{topk}
if [[ -z "${USE_MOE}" ]] && [[ "${MODEL_TAG}" =~ -MoE([0-9]+) ]]; then
  if [[ "${BASH_REMATCH[1]}" == "0" ]]; then
    USE_MOE=0
  else
    USE_MOE=1
  fi
fi
USE_MOE="${USE_MOE:-0}"

# convert 的 MoE 开关默认与训练一致（可用 USE_MOE_CONVERT 覆盖）
USE_MOE_CONVERT="${USE_MOE_CONVERT:-${USE_MOE}}"

# 自动推导 sglang 热更新共享目录（可被环境变量 SGLANG_SHARED_PATH 覆盖）
# 说明：该目录会被训练进程周期性 save_pretrained 覆盖写入，用于 sglang server 热更新权重。
SGLANG_SHARED_PATH="${SGLANG_SHARED_PATH:-$ROOT_DIR/trainer/ckpt_mm/${_MODEL_TAG_SAFE}}"

# 自动推导首次 convert 的 .pth 路径（可被环境变量 TORCH_CKPT_PATH 覆盖）
if [[ -z "${TORCH_CKPT_PATH}" ]]; then
  _guess="$ROOT_DIR/out/${MODEL_TAG}.pth"
  if [[ -f "${_guess}" ]]; then
    TORCH_CKPT_PATH="${_guess}"
  fi
fi

SGLANG_STARTED_BY_SCRIPT=0
TORCHRUN_STARTED_BY_SCRIPT=0

cleanup() {
  # 停止 torchrun（及其 worker 进程）
  if [[ "${AUTO_STOP_TORCHRUN}" == "1" ]] && [[ "${TORCHRUN_STARTED_BY_SCRIPT}" == "1" ]]; then
    if [[ -f "${TORCHRUN_PIDFILE}" ]]; then
      local tpid
      tpid="$(cat "${TORCHRUN_PIDFILE}" || true)"
      if [[ -n "${tpid}" ]] && kill -0 "${tpid}" 2>/dev/null; then
        echo "[agent.sh] 脚本退出，停止 torchrun(pid=${tpid})"
        # 优先杀进程组，确保 8 个 worker 一起退出
        kill -- "-${tpid}" 2>/dev/null || kill "${tpid}" 2>/dev/null || true
      fi
      rm -f "${TORCHRUN_PIDFILE}" || true
    fi
  fi

  if [[ "${AUTO_STOP_SGLANG}" != "1" ]]; then
    return 0
  fi
  if [[ "${SGLANG_STARTED_BY_SCRIPT}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${SGLANG_PIDFILE}" ]]; then
    local pid
    pid="$(cat "${SGLANG_PIDFILE}" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[agent.sh] 训练结束，停止 sglang server(pid=${pid})"
      kill "${pid}" 2>/dev/null || true
    fi
    rm -f "${SGLANG_PIDFILE}" || true
  fi
}

trap cleanup EXIT INT TERM

if [[ ! -f "${SGLANG_MODEL_PATH}/config.json" ]]; then
  if [[ "${AUTO_CONVERT_SGLANG_MODEL}" != "1" ]]; then
    echo "[agent.sh][ERROR] 找不到 Transformers 模型目录：${SGLANG_MODEL_PATH}（缺少 config.json）" >&2
    echo "[agent.sh] 你已关闭自动转换：AUTO_CONVERT_SGLANG_MODEL=0" >&2
    exit 1
  fi
  if [[ -z "${TORCH_CKPT_PATH}" ]]; then
    echo "[agent.sh][ERROR] 未设置 TORCH_CKPT_PATH，无法自动 convert 生成 ${SGLANG_MODEL_PATH}" >&2
    echo "[agent.sh] 请设置例如：TORCH_CKPT_PATH=$ROOT_DIR/out/xxx.pth" >&2
    exit 1
  fi
  echo "[agent.sh] 未找到 ${SGLANG_MODEL_PATH}/config.json，先执行一次 convert 生成 Transformers 模型目录..."
  echo "[agent.sh] torch_ckpt=${TORCH_CKPT_PATH}"
  echo "[agent.sh] export_dir=${SGLANG_MODEL_PATH}"
  python "${ROOT_DIR}/scripts/convert_model.py" \
    --torch_path "${TORCH_CKPT_PATH}" \
    --transformers_path "${SGLANG_MODEL_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --arch qwen_compatible \
    --hidden_size "${HIDDEN_SIZE}" \
    --num_hidden_layers "${NUM_HIDDEN_LAYERS}" \
    --max_seq_len "${MAX_SEQ_LEN_EXPORT}" \
    --use_moe "${USE_MOE_CONVERT}" \
    --dtype float16
fi

check_sglang_health() {
  python - <<PY
import sys
import requests
url = "${SGLANG_BASE_URL}".rstrip("/") + "/health"
try:
    r = requests.get(url, timeout=2)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

if [[ "${AUTO_START_SGLANG}" == "1" ]]; then
  if ! check_sglang_health; then
    echo "[agent.sh] sglang server 未启动，自动拉起：${SGLANG_BASE_URL}"
    echo "[agent.sh] model_path=${SGLANG_MODEL_PATH} shared_path=${SGLANG_SHARED_PATH}"
    mkdir -p "${SGLANG_SHARED_PATH}"
    python -m sglang.launch_server \
      --model-path "${SGLANG_MODEL_PATH}" \
      --attention-backend "${SGLANG_ATTENTION_BACKEND}" \
      --host "${SGLANG_HOST}" \
      --port "${SGLANG_PORT}" \
      >"${SGLANG_LOGFILE}" 2>&1 &
    echo $! >"${SGLANG_PIDFILE}"
    SGLANG_STARTED_BY_SCRIPT=1

    for _ in $(seq 1 60); do
      if check_sglang_health; then
        echo "[agent.sh] sglang server 已就绪"
        break
      fi
      sleep 1
    done
  else
    echo "[agent.sh] sglang server 已在运行：${SGLANG_BASE_URL}"
  fi
else
  echo "[agent.sh] AUTO_START_SGLANG=0，跳过自动启动；请自行确保 ${SGLANG_BASE_URL} 可用"
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${ROOT_DIR}/trainer/train_agent.py" \
  --use_wandb \
  --rollout_engine sglang \
  --sglang_base_url "${SGLANG_BASE_URL}" \
  --sglang_model_path "${SGLANG_MODEL_PATH}" \
  --sglang_shared_path "${SGLANG_SHARED_PATH}" \
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
