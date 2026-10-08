#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to the BFCL model identifier}"
: "${MODEL_PATH:?Set MODEL_PATH to checkpoint weights}"
: "${TAG:?Set TAG to a result prefix ending in -e1}"
: "${BFCL_DATA_DIR:?Set BFCL_DATA_DIR to the pinned v3 data}"
export BFCL_RESULTS="${BFCL_RESULTS:-$PWD/results/bfcl}"
mkdir -p "$BFCL_RESULTS"
export BFCL_REPS="${BFCL_REPS:-1,2,3}" BFCL_DROP_HISTORY_THINK=1
export EVAL_RUN_ID="${EVAL_RUN_ID:-$$}"
export LOCAL_SERVER_PORT="${LOCAL_SERVER_PORT:-$((20000 + $$ % 20000))}"
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/eval.py"
