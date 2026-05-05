#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

CONFIGS=(
  configs/MQAR/mqar_mamba2_depth4_len128.yaml
  configs/MQAR/mqar_mamba2_depth6_len128.yaml
  configs/MQAR/mqar_ms_gated_stride2_len128.yaml
  configs/MQAR/mqar_ms_attention_stride4_len128.yaml
)

for c in "${CONFIGS[@]}"; do
  echo "========== G1 len=128: $c =========="
  "$PYTHON" code/train_mqar.py --config "$c"
done

echo "========== aggregate =========="
"$PYTHON" code/aggregate_mqar_results.py

SUMMARY="$ROOT/outputs/MQAR/summary_mqar.csv"
echo "summary: $SUMMARY"
