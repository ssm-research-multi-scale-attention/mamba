#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

CONFIGS=(
  configs/MQAR/mqar_mamba2_depth4_len128.yaml
  configs/MQAR/mqar_mamba2_depth4_len256.yaml
  configs/MQAR/mqar_mamba2_depth4_len512.yaml
  configs/MQAR/mqar_mamba2_depth6_len128.yaml
  configs/MQAR/mqar_mamba2_depth6_len256.yaml
  configs/MQAR/mqar_mamba2_depth6_len512.yaml
  configs/MQAR/mqar_ms_gated_stride2_len128.yaml
  configs/MQAR/mqar_ms_gated_stride2_len256.yaml
  configs/MQAR/mqar_ms_gated_stride2_len512.yaml
  configs/MQAR/mqar_ms_attention_stride4_len128.yaml
  configs/MQAR/mqar_ms_attention_stride4_len256.yaml
  configs/MQAR/mqar_ms_attention_stride4_len512.yaml
)

for c in "${CONFIGS[@]}"; do
  echo "========== $c =========="
  "$PYTHON" code/train_mqar.py --config "$c"
done

echo "Aggregating summary..."
"$PYTHON" code/aggregate_mqar_results.py
echo "MQAR len sweep finished."
