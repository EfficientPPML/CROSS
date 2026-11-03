#!/usr/bin/env bash
# Prepared command only. Scheduling/TPU lease is owned by the parent agent.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {bert|qwen} NEW_OUTPUT_DIRECTORY EXISTING_PUBLIC_KEY_PARENT" >&2
  exit 2
fi
case "$1" in
  bert) problem=p2_bert ;;
  qwen) problem=p4_qwen ;;
  *) echo 'model must be bert or qwen' >&2; exit 2 ;;
esac
snapshot_root=${SWEEP_ROOT:-/tmp/transformer-collector-outer-failure-20261103/checkout}
python_bin=${SWEEP_PYTHON:-/home/jianming_gatech/miniconda3/envs/jaxite/bin/python}
cpu_affinity=${SWEEP_CPU_AFFINITY:-160-167}
[[ -d "$3" && ! -e "$2" ]]
exec taskset -c "$cpu_affinity" env \
  -u CROSS_SKIP_TOPLEVEL_ROTKEYS -u XLA_FLAGS \
  CROSS_ROOT="$snapshot_root" PYTHONPATH="$snapshot_root:$snapshot_root/demos" \
  PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=tpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CROSS_LEVEL_PARAMS_MAX=1 CROSS_BARRETT_NTT_CONTEXT_CACHE_CAPACITY=2 \
  CROSS_PUBLIC_KEY_DIRECTORY="$3" \
  "$python_bin" "$snapshot_root/cadathlon/validation/check_transformer_suite.py" \
  --model "$1" \
  --implementation "$snapshot_root/cadathlon/agent_pset/$problem/src/${1}_optimized_reference.py" \
  --cases 1 2 3 4 5 6 --warmups 1 --repeats 3 --cleanup-grace-s 30 --output "$2"
