#!/usr/bin/env bash
set -euo pipefail
# Check GPU availability before loading data or model weights.
if ! nvidia-smi -L > /tmp/gpu_probe.$$ 2>&1 || grep -qiE "unknown error|unable to determine" /tmp/gpu_probe.$$; then
  echo "[run_sft] GPU PROBE FAILED on $(hostname):"; cat /tmp/gpu_probe.$$; rm -f /tmp/gpu_probe.$$
  exit 86
fi
ngpu_seen=$(grep -c "^GPU" /tmp/gpu_probe.$$ || true); rm -f /tmp/gpu_probe.$$
if [ "${ngpu_seen:-0}" -lt "${NGPUS:-${1:-1}}" ]; then
  echo "[run_sft] GPU PROBE: only $ngpu_seen visible GPUs on $(hostname), need ${NGPUS:-${1:-1}}"
  exit 86
fi
ENV_INIT="${ENV_INIT:-}"
if [ -f "$ENV_INIT" ]; then
  source "$ENV_INIT"
else
  echo "[run_sft] WARN: ENV_INIT not found ($ENV_INIT); assuming HF/WANDB env is already set"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Resolve the reference recipe relative to this entry point.
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPUS=${NGPUS:-${1:-1}}; [ $# -gt 0 ] && shift || true
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  vis=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
  [ "$vis" -lt "$NGPUS" ] && NGPUS=$vis
fi
export NGPUS
# Keep tracking logs outside the code directory.
export WANDB_DIR="${WANDB_DIR:-${FAST_ROOT:-$HOME}/wandb/PredictionBench-trl}"
mkdir -p "$WANDB_DIR"
: "${TRAIN_FILE:?Set TRAIN_FILE to a validated group parquet}"
: "${MODEL_PATH:?Set MODEL_PATH to Qwen3-4B weights}"
: "${TOKENIZER_DIR:?Set TOKENIZER_DIR to the all-thinking training tokenizer}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the new checkpoint directory}"
if [ "$NGPUS" -le 1 ]; then
  # single GPU: plain python, no FSDP (avoids the multi-GPU FSDP bf16 NaN)
  python "$SCRIPTS/sft.py"
else
  # Use a distinct distributed port for concurrent training processes.
  accelerate launch --config_file "${ACCEL_CONFIG:-$SCRIPTS/fsdp.yaml}" --num_processes "$NGPUS" --main_process_port "${MAIN_PORT:-$((20000 + RANDOM % 20000))}" "$SCRIPTS/sft.py"
fi

