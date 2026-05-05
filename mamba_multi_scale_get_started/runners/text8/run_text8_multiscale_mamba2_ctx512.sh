#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

echo "Running Text8 MultiScale Mamba2 ctx512 training..."
OUT=outputs/Text8/text8_multiscale_mamba2_ctx512_runner
mkdir -p "$OUT"
"$PYTHON" code/train_lm.py \
  --config configs/Text8/text8_multiscale_mamba2_ctx512.yaml \
  experiment.name=text8_multiscale_mamba2_ctx512_runner \
  logging.output_dir="$OUT" \
  2>&1 | tee "$OUT/run.log"

echo "Text8 training finished."
