#!/usr/bin/env bash
set -euo pipefail
result_dir=$1
run_id=$2
block_index=$3
cpu_list=$4
env_file=$5
checkpoint=$6
python_bin=$7
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set --
source "$env_file"
set -euo pipefail
export PYTHONPATH="$bundle_dir:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN_DEMO_CPU_LIST="$cpu_list" QWEN_DEMO_WORKER_AUDIT="$result_dir/worker-audit.json"
test "${ASCEND_RT_VISIBLE_DEVICES:-}" = 0
cd "${PYPTO_SERVING_ROOT:?Select the pinned serving checkout}"
mkdir -p "$result_dir"
npu-smi info > "$result_dir/npu-before.txt"
"$python_bin" "$bundle_dir/run_pypto.py" \
  --model "$checkpoint" --compiled-dir "$result_dir/../../compiled/block-$block_index" \
  --output "$result_dir/results.json" --device "${TASK_DEVICE:-0}" \
  --lease-id "$run_id" --block-index "$block_index" --variant pypto
npu-smi info > "$result_dir/npu-after.txt"
