#!/usr/bin/env bash
set -euo pipefail
env_file=$1
python_bin=$2
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set --
source "$env_file"
set -euo pipefail
export PYTHONPATH="$bundle_dir:${PYTHONPATH:-}"
"$python_bin" "$bundle_dir/audit.py"
"$python_bin" -c 'import pypto, simpler, golden, pypto_serving; print("Runtime imports OK")'
