#!/usr/bin/env bash
set -euo pipefail
checkpoint=${1:?Usage: bash run_offline.sh CHECKPOINT [DEVICE] [PROMPT]}
device=${2:-${TASK_DEVICE:-0}}
prompt=${3:-The capital of France is}
: "${PYPTO_LIB_ROOT:?Source environment.local.sh first}"
: "${PYPTO_SERVING_ROOT:?Source environment.local.sh first}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p "$PYPTO_SERVING_ROOT/build_output"
export PYPTO_PROG_BUILD_DIR
PYPTO_PROG_BUILD_DIR=$(mktemp -d "$PYPTO_SERVING_ROOT/build_output/qwen-preview.XXXXXX")
python -m pypto_serving.cli.main \
  --model "$checkpoint" --platform a2a3 --device "$device" --tp 1 --ep 1 \
  --max-model-len 4096 --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --long-prefill-token-threshold 2048 --block-size 128 --no-enable-prefix-caching \
  --prompt "$prompt" --generate-config '{"max_new_tokens":8,"temperature":0.0}'
