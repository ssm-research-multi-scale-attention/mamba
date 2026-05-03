#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

echo "Running Mamba2 medium long context (ctx512, 8 layers)..."
OUT=outputs/TinyShakespeare/tiny_shakespeare_mamba2_medium_ctx512
mkdir -p "$OUT"
"$PYTHON" code/train_lm.py \
  --config configs/TinyShakespeare/tiny_shakespeare_mamba2.yaml \
  experiment.name=tiny_shakespeare_mamba2_medium_ctx512 \
  logging.output_dir="$OUT" \
  data.block_size=512 \
  data.sampling=random_windows \
  data.steps_per_epoch=10000 \
  train.epochs=20 \
  train.lr=0.0003 \
  'model.layer_headdims=[32,32,32,32,32,32,32,32]' \
  2>&1 | tee "$OUT/run.log"

echo "Running Mamba2 large long context (ctx512, d_model=512)..."
OUT=outputs/TinyShakespeare/tiny_shakespeare_mamba2_large_ctx512
mkdir -p "$OUT"
"$PYTHON" code/train_lm.py \
  --config configs/TinyShakespeare/tiny_shakespeare_mamba2.yaml \
  experiment.name=tiny_shakespeare_mamba2_large_ctx512 \
  logging.output_dir="$OUT" \
  data.block_size=512 \
  data.sampling=random_windows \
  data.steps_per_epoch=10000 \
  train.epochs=20 \
  train.lr=0.0003 \
  model.d_model=512 \
  'model.layer_headdims=[64,64,64,64,64,64,64,64]' \
  model.mamba.d_state=64 \
  2>&1 | tee "$OUT/run.log"

echo "Running MultiScale Mamba2 long context (ctx512, gated, stride=4)..."
OUT=outputs/TinyShakespeare/tiny_shakespeare_multiscale_mamba2_ctx512
mkdir -p "$OUT"
"$PYTHON" code/train_lm.py \
  --config configs/TinyShakespeare/tiny_shakespeare_multiscale_mamba2.yaml \
  experiment.name=tiny_shakespeare_multiscale_mamba2_ctx512 \
  logging.output_dir="$OUT" \
  data.block_size=512 \
  data.sampling=random_windows \
  data.steps_per_epoch=10000 \
  train.epochs=20 \
  train.lr=0.0003 \
  model.multiscale.fusion=gated \
  model.multiscale.stride=4 \
  2>&1 | tee "$OUT/run.log"

echo "Heavy TinyShakespeare night experiments finished."
