#!/usr/bin/env bash
# External sanity: upstream HazyResearch Zoology MQAR (not ArchEval).
# Default Zoology tree: ../../zoology from project root (sibling of mamba/), i.e. repos/zoology.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${ZOOLOGY_ROOT:-}:${PYTHONPATH:-}"

# repos/zoology next to repos/mamba (default if unset)
DEFAULT_ZOOLOGY="$(cd "$ROOT/../.." && pwd)/zoology"
ZOOLOGY_ROOT="${ZOOLOGY_ROOT:-$DEFAULT_ZOOLOGY}"

mkdir -p outputs/ZoologyMQAR
mkdir -p logs/zoology_mqar

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/zoology_mqar/zmq_sanity_${STAMP}.log"

if [[ ! -d "$ZOOLOGY_ROOT" ]]; then
  echo "ERROR: Zoology repo not found at ZOOLOGY_ROOT=$ZOOLOGY_ROOT" | tee "$LOG"
  echo "Clone or point ZOOLOGY_ROOT to your checkout, e.g.:" | tee -a "$LOG"
  echo "  git clone https://github.com/HazyResearch/zoology.git \"$(dirname "$ROOT")/../../zoology\"" | tee -a "$LOG"
  echo "Or: export ZOOLOGY_ROOT=/path/to/repos/zoology" | tee -a "$LOG"
  echo "See docs/zoology_mqar_sanity.md" | tee -a "$LOG"
  exit 1
fi

if ! "$PYTHON" -c "import zoology" 2>/dev/null; then
  echo "NOTE: zoology not importable in current env; adding ZOOLOGY_ROOT to PYTHONPATH." | tee "$LOG"
  export PYTHONPATH="$ZOOLOGY_ROOT:${PYTHONPATH:-}"
fi

if ! "$PYTHON" -c "import zoology, torch" 2>/dev/null; then
  echo "ERROR: Cannot import zoology and/or torch. Activate your conda env (e.g. nvidenisov-other-4) or:" | tee "$LOG"
  echo "  pip install -e \"\$ZOOLOGY_ROOT\"" | tee "$LOG"
  exit 1
fi

echo "========== Zoology MQAR sanity ==========" | tee "$LOG"
echo "ROOT=$ROOT" | tee -a "$LOG"
echo "ZOOLOGY_ROOT=$ZOOLOGY_ROOT" | tee -a "$LOG"

SEEDS="${SEEDS:-42,43,44}"
MODELS="${MODELS:-mha,mamba2}"
REGIMES="${REGIMES:-easy,trans704,trans768}"
EXTRA=()
if [[ "${INCLUDE_HARD:-0}" == "1" ]]; then
  EXTRA+=(--include-hard)
fi

{
  "$PYTHON" code/zoology_mqar_sanity.py \
    --zoology-root "$ZOOLOGY_ROOT" \
    --output-root outputs/ZoologyMQAR \
    --models "$MODELS" \
    --regimes "$REGIMES" \
    --seeds "$SEEDS" \
    "${EXTRA[@]}"
} 2>&1 | tee -a "$LOG"

"$PYTHON" code/aggregate_zoology_mqar.py \
  --output-root outputs/ZoologyMQAR \
  --csv-out outputs/ZoologyMQAR/summary_zoology_mqar.csv \
  --report-out outputs/ZoologyMQAR/report_zoology_mqar.txt

echo "Log: $LOG" | tee -a "$LOG"
echo "Summary: $ROOT/outputs/ZoologyMQAR/summary_zoology_mqar.csv" | tee -a "$LOG"
echo "Report:  $ROOT/outputs/ZoologyMQAR/report_zoology_mqar.txt" | tee -a "$LOG"
