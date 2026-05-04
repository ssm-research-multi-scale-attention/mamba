#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

echo "Running MultiScale Mamba2 (gated, 4 layers)..."
OUT=outputs/TinyShakespeare/tiny_shakespeare_multiscale_mamba2_gated_4layers_random_windows
mkdir -p "$OUT"
"$PYTHON" code/train_lm.py \
  --config configs/TinyShakespeare/tiny_shakespeare_multiscale_mamba2.yaml \
  'model.multiscale.fusion=gated' \
  'model.layer_headdims=[32,32,32,32]' \
  experiment.name=tiny_shakespeare_multiscale_mamba2_gated_4layers_random_windows \
  logging.output_dir="$OUT" \
  2>&1 | tee "$OUT/run.log"

echo "Running MultiScale Mamba2 (gated, 12 layers)..."
OUT=outputs/TinyShakespeare/tiny_shakespeare_multiscale_mamba2_gated_12layers_random_windows
mkdir -p "$OUT"
"$PYTHON" code/train_lm.py \
  --config configs/TinyShakespeare/tiny_shakespeare_multiscale_mamba2.yaml \
  'model.multiscale.fusion=gated' \
  'model.layer_headdims=[32,32,32,32,32,32,32,32,32,32,32,32]' \
  experiment.name=tiny_shakespeare_multiscale_mamba2_gated_12layers_random_windows \
  logging.output_dir="$OUT" \
  2>&1 | tee "$OUT/run.log"


echo "All experiments finished."
