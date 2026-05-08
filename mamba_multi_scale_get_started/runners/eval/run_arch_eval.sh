#!/usr/bin/env bash
# Unified arch eval: Tiny Shakespeare LM + MQAR sweep + benchmark_timing (waves over GPUs).
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
ARCH_REGISTRY="${ARCH_REGISTRY:-configs/EvalRegistry/architectures.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
LM_SEEDS="${LM_SEEDS:-42}"
MQAR_SEEDS="${MQAR_SEEDS:-42,43,44}"
TIMING_REPEATS="${TIMING_REPEATS:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --architectures)
      ARCH_REGISTRY="${2:?}"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="${2:?}"
      shift 2
      ;;
    --lm-seeds)
      LM_SEEDS="${2:?}"
      shift 2
      ;;
    --mqar-seeds)
      MQAR_SEEDS="${2:?}"
      shift 2
      ;;
    --timing-repeats)
      TIMING_REPEATS="${2:?}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    --skip-completed)
      SKIP_COMPLETED=1
      shift 1
      ;;
    -h|--help)
      echo "Usage: NUM_GPUS=4 $0 [--architectures PATH] [--num-gpus N] \\"
      echo "  [--lm-seeds CSV] [--mqar-seeds CSV] [--timing-repeats K] [--dry-run] [--skip-completed]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p logs/arch_eval
mkdir -p outputs/ArchEval

if [[ "$ARCH_REGISTRY" = /* ]]; then
  ARCH_REGISTRY_ABS="$ARCH_REGISTRY"
else
  ARCH_REGISTRY_ABS="$ROOT/$ARCH_REGISTRY"
fi

REG_BASENAME="$(basename "$ARCH_REGISTRY_ABS")"
REG_STEM="${REG_BASENAME%.yaml}"
IS_FULL_REGISTRY=0
if [[ "$REG_BASENAME" == "architectures.yaml" ]]; then
  IS_FULL_REGISTRY=1
fi

export ARCH_EVAL_LM_SEEDS="$LM_SEEDS"
export ARCH_EVAL_MQAR_SEEDS="$MQAR_SEEDS"
export ARCH_EVAL_TIMING_REPEATS="$TIMING_REPEATS"

MANIFEST_OPTS=(
  "--lm-seeds" "$LM_SEEDS"
  "--mqar-seeds" "$MQAR_SEEDS"
  "--timing-repeats" "$TIMING_REPEATS"
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  "$PYTHON" code/arch_eval_emit_manifest.py "$ARCH_REGISTRY_ABS" "${MANIFEST_OPTS[@]}" --dry-run
  exit 0
fi

MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT
"$PYTHON" code/arch_eval_emit_manifest.py "$ARCH_REGISTRY_ABS" "${MANIFEST_OPTS[@]}" >"$MANIFEST" || exit 1

JOBIDX=()

should_skip_lm() {
  [[ "$SKIP_COMPLETED" -eq 1 ]] || return 1
  local dir="$ROOT/outputs/ArchEval/$1"
  "$PYTHON" -c "import sys
from pathlib import Path
ROOT=Path('$ROOT')
sys.path.insert(0, str(ROOT/'code'))
from arch_eval_common import lm_meta_ok
sys.exit(0 if lm_meta_ok(Path('$dir')/'meta_metrics.csv') else 1)"
}

should_skip_mqar() {
  [[ "$SKIP_COMPLETED" -eq 1 ]] || return 1
  local dir="$ROOT/outputs/ArchEval/$1"
  "$PYTHON" -c "import sys
from pathlib import Path
ROOT=Path('$ROOT')
sys.path.insert(0, str(ROOT/'code'))
from arch_eval_common import mqar_meta_ok
sys.exit(0 if mqar_meta_ok(Path('$dir')/'meta_metrics.csv') else 1)"
}

should_skip_timing() {
  [[ "$SKIP_COMPLETED" -eq 1 ]] || return 1
  local csv="$ROOT/$1"
  "$PYTHON" -c "import sys
from pathlib import Path
ROOT=Path('$ROOT')
sys.path.insert(0, str(ROOT/'code'))
from arch_eval_common import timing_csv_ok
sys.exit(0 if timing_csv_ok(Path('$csv'), 1024) else 1)"
}

resolve_timing_csv() {
  local arch="$1"
  local rep="$2"
  if [[ "$TIMING_REPEATS" -le 1 ]]; then
    echo "outputs/ArchEval/timing_${arch}.csv"
  else
    echo "outputs/ArchEval/timing_${arch}_repeat${rep}.csv"
  fi
}

run_lm_wave() {
  local arch="$1"
  local lm_cfg="$2"
  local seed="$3"
  local out_stem="$4"
  local log="$5"

  local out_rel="outputs/ArchEval/${out_stem}"
  if should_skip_lm "$out_stem"; then
    echo "SKIP lm arch=$arch seed=$seed (outputs/ArchEval/${out_stem})" | tee "$log"
    return 0
  fi

  echo "JOB lm $arch seed=$seed out=${out_stem}" | tee "$log"
  "$PYTHON" code/train_lm.py \
    --config "$ROOT/$lm_cfg" \
    experiment.name="archeval_lm_${arch}_seed${seed}" \
    logging.output_dir="$out_rel" \
    data.block_size=1024 \
    train.epochs=10 \
    experiment.seed="$seed" \
    loader.num_workers=0 \
    cuda_device=0 \
    device=cuda:0 \
    >>"$log" 2>&1 || return 1
}

run_mqar_wave() {
  local mqar_cfg="$1"
  local arch="$2"
  local setting="$3"
  local seed="$4"
  local vocab="$5"
  local minq="$6"
  local log="$7"

  local out_stem="mqar_${setting}_${arch}_seed${seed}"

  if should_skip_mqar "$out_stem"; then
    echo "SKIP mqar setting=$setting arch=$arch seed=$seed" | tee -a "$log"
    return 0
  fi

  echo "JOB mqar setting=$setting arch=$arch seed=$seed" | tee -a "$log"
  local min_arg=()
  if [[ "$minq" == "null" ]]; then
    min_arg=(data.min_query_pos=null)
  else
    min_arg=(data.min_query_pos="$minq")
  fi

  "$PYTHON" code/train_mqar.py \
    --config "$ROOT/$mqar_cfg" \
    experiment.name="archeval_mqar_${setting}_${arch}_seed${seed}" \
    logging.output_dir="outputs/ArchEval/$out_stem" \
    experiment.seed="$seed" \
    data.input_seq_len=128 \
    data.num_kv_pairs=16 \
    data.vocab_size="$vocab" \
    "${min_arg[@]}" \
    data.train_examples=20000 \
    data.val_examples=2000 \
    data.test_examples=2000 \
    data.fixed_examples=true \
    data.random_non_queries=true \
    train.lr=0.0003 \
    train.epochs=20 \
    train.early_stopping.patience=4 \
    loader.num_workers=0 \
    loader.pin_memory=false \
    cuda_device=0 \
    device=cuda:0 \
    >>"$log" 2>&1 || return 1
}

run_timing_wave() {
  local bench_cfg="$1"
  local arch="$2"
  local repeat="$3"
  local log="$4"
  local rel_csv="$5"
  local outcsv_rel="$rel_csv"

  if should_skip_timing "$outcsv_rel"; then
    echo "SKIP timing arch=$arch repeat=$repeat → $rel_csv" | tee -a "$log"
    return 0
  fi

  echo "JOB timing arch=$arch repeat=$repeat -> $rel_csv" | tee -a "$log"
  "$PYTHON" code/benchmark_timing.py \
    --configs "$ROOT/$bench_cfg" \
    --block-sizes 1024 \
    --batch-sizes 1 8 16 \
    --output-csv "$ROOT/$rel_csv" \
    --overwrite \
    --set "experiment.name=archeval_timing_${arch}_rep${repeat}" \
    --set "experiment.seed=$((42 + repeat))" \
    --set device=cuda:0 \
    --set cuda_device=0 \
    >>"$log" 2>&1 || return 1
}

# Build JOB list as pipe-delimited strings
JOB_LINES=()
while IFS=$'\t' read -ra F || [[ ${#F[@]} -gt 0 ]]; do
  [[ -z "${F[0]:-}" ]] && continue
  kind="${F[0]}"
  case "$kind" in
    lm)
      JOB_LINES+=("lm|${F[1]}|${F[2]}|${F[3]}|${F[4]}|${F[5]}")
      ;;
    mqar)
      JOB_LINES+=("mqar|${F[1]}|${F[2]}|${F[3]}|${F[4]}|${F[5]}|${F[6]}|${F[7]}")
      ;;
    timing)
      # timing | arch | atype | bench_cfg | repeat
      JOB_LINES+=("timing|${F[1]}|${F[2]}|${F[3]}|${F[4]}")
      ;;
    *)
      echo "unknown manifest kind: $kind" >&2
      exit 1
      ;;
  esac
done <"$MANIFEST"

n="${#JOB_LINES[@]}"
echo "Total arch-eval jobs: $n"

run_wave_shell() {
  local start="$1"
  local end="$2"
  local pids=()
  local failed=0
  local i gpu

  for ((i = start; i < end; i++)); do
    gpu=$((i - start))
    local line="${JOB_LINES[$i]}"
    local log="logs/arch_eval/job_${i}_gpu${gpu}.log"
    {
      echo "========== JOB $((i + 1))/$n =========="
      echo "$line"

      local IFS='|'
      read -ra P <<<"$line"

      export CUDA_VISIBLE_DEVICES="$gpu"

      case "${P[0]}" in
        lm)
          run_lm_wave "${P[1]}" "${P[3]}" "${P[4]}" "${P[5]}" "$log"
          ;;
        mqar)
          run_mqar_wave "${P[3]}" "${P[1]}" "${P[4]}" "${P[5]}" "${P[6]}" "${P[7]}" "$log"
          ;;
        timing)
          rep="${P[4]}"
          rel_csv="$(resolve_timing_csv "${P[1]}" "$rep")"
          run_timing_wave "${P[3]}" "${P[1]}" "$rep" "$log" "$rel_csv"
          ;;
        *)
          echo "bad job kind ${P[0]}" >&2
          exit 1
          ;;
      esac
    } &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "Some arch_eval jobs failed. See logs/arch_eval/job_*.log"
    exit 1
  fi
}

for ((start = 0; start < n; start += NUM_GPUS)); do
  end=$((start + NUM_GPUS))
  ((end > n)) && end=$n
  echo "========== wave: jobs [$start, $end) =========="
  run_wave_shell "$start" "$end"
done

echo "========== aggregate =========="
SUMMARY_PATH="$ROOT/outputs/ArchEval/summary_arch_eval.csv"
AGG_OPTS=()
AGG_OPTS+=(--registry "$ARCH_REGISTRY_ABS")
AGG_OPTS+=(--lm-seeds "$LM_SEEDS")
AGG_OPTS+=(--mqar-seeds "$MQAR_SEEDS")
AGG_OPTS+=(--timing-repeats "$TIMING_REPEATS")

if [[ "$IS_FULL_REGISTRY" -eq 1 ]]; then
  "$PYTHON" code/aggregate_arch_eval.py "${AGG_OPTS[@]}"
else
  OUT_NAME="summary_arch_eval_${REG_STEM}.csv"
  "$PYTHON" code/aggregate_arch_eval.py "${AGG_OPTS[@]}" --output-file "$OUT_NAME"
  SUMMARY_PATH="$ROOT/outputs/ArchEval/$OUT_NAME"
fi

echo "========== analyze =========="
if [[ "$IS_FULL_REGISTRY" -eq 1 ]]; then
  "$PYTHON" code/analyze_arch_eval.py "$SUMMARY_PATH"
  REP="$ROOT/outputs/ArchEval/report_arch_eval.txt"
else
  REPORT_NAME="report_arch_eval_${REG_STEM}.txt"
  SCORECARD_NAME="scorecard_arch_eval_${REG_STEM}.csv"
  "$PYTHON" code/analyze_arch_eval.py \
    "$SUMMARY_PATH" \
    --baseline-summary "$ROOT/outputs/ArchEval/summary_arch_eval_full.csv" \
    --out-scorecard "$ROOT/outputs/ArchEval/$SCORECARD_NAME" \
    --out-report "$ROOT/outputs/ArchEval/$REPORT_NAME"
  REP="$ROOT/outputs/ArchEval/$REPORT_NAME"
fi

if [[ -f "$REP" ]]; then
  echo "========== $REP =========="
  cat "$REP"
else
  echo "Missing report: $REP"
  exit 1
fi
