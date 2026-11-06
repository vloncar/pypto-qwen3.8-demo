# Set up and run

## Prerequisites

Use a Linux development environment with an Ascend 910B2 visible to the user,
matching host driver/firmware, CANN 9.0.0, Python 3.12, Git, a C++17 compiler,
CMake and Ninja. The reference host is aarch64. Other Ascend A2/A3 configurations
need independent validation. Driver installation is a host administration task;
these instructions install Python components into a virtual environment.

Allow disk space for the checkpoint (about 30 GB), source builds and generated
programs, plus sufficient host/device memory for weights and caches. A container
needs working Ascend device mappings and enough shared memory for weight staging.
For the automated comparison it must also have your site's `task-submit` device
allocator installed; that service is not included in this repository.

## Get the exact sources

Choose a workspace and clone the companion repository:

```bash
export QWEN_WORKSPACE="$HOME/qwen-pypto"
mkdir -p "$QWEN_WORKSPACE"
git clone https://github.com/vloncar/pypto-qwen3.8-demo.git "$QWEN_WORKSPACE/pypto-qwen3.8-demo"
cd "$QWEN_WORKSPACE/pypto-qwen3.8-demo"
python3 checkout.py "$QWEN_WORKSPACE"
```

Alternatively, existing local checkouts can supply the pinned commits:

```bash
python3 checkout.py "$QWEN_WORKSPACE" \
  --lib-url /path/to/local/pypto-lib \
  --serving-url /path/to/local/pypto-serving
```

`checkout.py` reads the exact commits from `versions.json`, checks them out
without modifying the originals, and initializes the compiler runtime and
serving kernel submodules. It refuses to overwrite existing target directories.
`--compiler-url /path/to/local/pypto` also supports an existing compiler clone.
The detached checkouts are intentional: a later branch update cannot silently
change this benchmark. Place a copy of this companion repo at
`$QWEN_WORKSPACE/pypto-qwen3.8-demo` or adjust the following paths.

## Install the PyPTO environment

Run inside the development container/environment. Source the installed CANN
version first, so the runtime installer can find both device compilers:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
npu-smi info
test -x "$ASCEND_HOME_PATH/bin/ccec"
test -f "$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"

python3.12 -m venv "$QWEN_WORKSPACE/.venv"
source "$QWEN_WORKSPACE/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$QWEN_WORKSPACE/pypto/build-constraints.txt"
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "$QWEN_WORKSPACE/pypto-qwen3.8-demo/requirements-pypto.txt"
python -m pip install --no-build-isolation "$QWEN_WORKSPACE/pypto"
python -m pip install --no-build-isolation "$QWEN_WORKSPACE/pypto/runtime"
python -m pip install --no-deps -e "$QWEN_WORKSPACE/pypto-serving"
```

CPU PyTorch is used for host preparation and sampling; generated PyPTO programs
run the model on the NPU through the Ascend runtime. This installation does not
use a PyTorch CPU implementation of the model for serving.

Install the PTOAS release selected by the compiler checkout, rather than the
latest release. The binary bundle includes its Python runtime:

```bash
export PYPTO_ROOT="$QWEN_WORKSPACE/pypto"
source "$PYPTO_ROOT/toolchain/versions.env"
curl -fL "https://github.com/hw-native-sys/PTOAS/releases/download/$PTOAS_VERSION/ptoas-bin-$(uname -m).tar.gz" \
  -o "$QWEN_WORKSPACE/ptoas.tar.gz"
mkdir -p "$QWEN_WORKSPACE/ptoas-bin"
tar -xzf "$QWEN_WORKSPACE/ptoas.tar.gz" -C "$QWEN_WORKSPACE/ptoas-bin"
export PTOAS_ROOT="$QWEN_WORKSPACE/ptoas-bin"
```

Simpler resolves PTO ISA from its committed `pto_isa.pin` on first compilation.
Do not override it with an unrelated checkout. The expected resolved revisions
are recorded in `versions.json`; first use may require network access. The
reference environment used the equivalent PTOAS 0.67 wheel, not a new assembler
version. See the checked-out pypto-lib `docs/get-started/installation.md` for
compiler prerequisites and platform-specific troubleshooting.

Copy `environment.example.sh` to the ignored `environment.local.sh`, edit its
paths, then source it. `PYPTO_LIB_ROOT` must select the new kernel checkout,
not the older serving submodule. `PYTHONPATH` must not let a separate CANN-provided
PyPTO installation shadow the installed compiler.

```bash
cd "$QWEN_WORKSPACE/pypto-qwen3.8-demo"
cp environment.example.sh environment.local.sh
# Edit environment.local.sh for your paths before sourcing it.
source environment.local.sh
bash preflight.sh "$PWD/environment.local.sh" "$QWEN_WORKSPACE/.venv/bin/python"
python -m pypto_serving.cli.main --help
```

Preflight verifies model/serving source hashes, compiler/runtime checkout
revisions, the installed runtime build stamp, the assembler release and imports.
Keep the environment used to build PyPTO and simpler unchanged while
running. It is not a byte-for-byte validation of the compiler's native binaries.

## Get the checkpoint

Use the same checkpoint for both engines. Download on the host into a directory
mounted read-only in the development container, or copy the complete directory:

```bash
export QWEN_CHECKPOINT=/path/to/models/qwen/checkpoint
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Eco-Tech/Qwen3.8-27B-w8a8",
    revision="ca2937f18b7e399c0ee91c728a4cde9b837aebff",
    local_dir=os.environ["QWEN_CHECKPOINT"],
)
PY
python verify_checkpoint.py "$QWEN_CHECKPOINT"
```

The SHA-256 manifest covers all 19 files used in the comparison, including all
ten quantized weight shards. Both runners repeat this check outside timing.
The host and container paths can differ; their file contents must match.

## Run generation

First obtain an exclusive device allocation using your site's normal method.
Within that allocation, with `environment.local.sh` sourced:

```bash
bash run_offline.sh "$QWEN_CHECKPOINT" "${TASK_DEVICE:-0}" 'The capital of France is'
```

This runs the actual serving CLI with TP=EP=1, concurrency capacity two,
context 4096, page size 128, prefill budget 2048 and eight greedy output tokens.
Every launch gets a new build directory; compilation and weight staging take
place before generation. For a site's `task-submit` queue, put environment
activation and the command above in a local shell script and submit it with:

```bash
task-submit --device auto --ptoas none --timeout 3600 --max-time 0 \
  'bash /absolute/path/to/your-generation.sh'
```

For HTTP serving, use the same CLI options shown in `run_offline.sh`, omit
`--prompt` and `--generate-config`, and add `--host 127.0.0.1 --port 8000`.
Then send a raw completion request:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0.0}'
```

The benchmark below uses streaming engine APIs without HTTP in both engines.
Do not compare an HTTP measurement with its in-process numbers.
