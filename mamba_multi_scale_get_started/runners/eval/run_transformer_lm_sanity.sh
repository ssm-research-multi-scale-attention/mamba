#!/usr/bin/env bash
# Hyperparameter sanity sweep for TinyShakespeare transformer_lm (param_match scale only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CFG="configs/TinyShakespeare/tiny_shakespeare_transformer_param_match.yaml"
mkdir -p outputs/TransformerLMSanity logs/transformer_lm_sanity

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"

run_one() {
  local lr="$1"
  local do="$2"
  local name="sanity_pm_lr${lr}_do${do}_seed${SEED}"
  local out="outputs/TransformerLMSanity/${name}"
  local log="logs/transformer_lm_sanity/${name}.log"
  echo "======== ${name} (lr=${lr}, dropout=${do}) =========" | tee "$log"
  {
    "$PYTHON" code/train_lm.py \
      --config "$ROOT/$CFG" \
      experiment.name="$name" \
      logging.output_dir="$out" \
      train.lr="$lr" \
      "model.transformer.dropout=$do" \
      train.epochs="$EPOCHS" \
      train.early_stopping.patience="$PATIENCE" \
      experiment.seed="$SEED" \
      loader.batch_size=128 \
      loader.num_workers=0 \
      cuda_device=0 \
      device=cuda:0
  } 2>&1 | tee -a "$log" || echo "FAILED: $name" | tee -a "$log"
}

for lr in 1e-4 3e-4 1e-3; do
  for do in 0.0 0.1; do
    run_one "$lr" "$do"
  done
done

"$PYTHON" code/aggregate_transformer_lm_sanity.py \
  --root "$ROOT/outputs/TransformerLMSanity" \
  --csv-out "$ROOT/outputs/TransformerLMSanity/summary_transformer_lm_sanity.csv"

echo "Summary: $ROOT/outputs/TransformerLMSanity/summary_transformer_lm_sanity.csv"
