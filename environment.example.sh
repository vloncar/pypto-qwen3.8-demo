# Copy to environment.local.sh and edit these paths inside the development container.
# Source CANN before installing or importing the selected runtime.
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export QWEN_WORKSPACE=/workspace/qwen
source "$QWEN_WORKSPACE/.venv/bin/activate"
export PYPTO_LIB_ROOT="$QWEN_WORKSPACE/pypto-lib"
export PYPTO_SERVING_ROOT="$QWEN_WORKSPACE/pypto-serving"
export PYPTO_ROOT="$QWEN_WORKSPACE/pypto"
export PTOAS_ROOT="$QWEN_WORKSPACE/ptoas-bin"
# Override PYTHONPATH so another PyPTO provided by CANN cannot shadow this install.
export PYTHONPATH="$PYPTO_SERVING_ROOT:$PYPTO_LIB_ROOT:$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
