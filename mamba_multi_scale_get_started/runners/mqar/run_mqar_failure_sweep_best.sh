#!/usr/bin/env bash
# Thin wrapper: MQAR vocab failure sweep (see runners/mqar/py/run_mqar_failure_sweep_best.py).
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-8}"

exec "$PYTHON" runners/mqar/py/run_mqar_failure_sweep_best.py --num-gpus "$NUM_GPUS" "$@"
