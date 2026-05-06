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
    -h|--help)
      echo "Usage: NUM_GPUS=4 $0 [--architectures configs/EvalRegistry/architectures.yaml] [--num-gpus N]"
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

MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT
"$PYTHON" code/arch_eval_emit_manifest.py "$ARCH_REGISTRY_ABS" >"$MANIFEST" || exit 1

# JOBIDX=()

# run_lm_wave() {
#   local arch="$1"
#   local lm_cfg="$2"
#   local log="$3"
#   echo "JOB lm $arch" | tee "$log"
#   "$PYTHON" code/train_lm.py \
#     --config "$ROOT/$lm_cfg" \
#     experiment.name="archeval_lm_${arch}" \
#     logging.output_dir="outputs/ArchEval/lm_${arch}" \
#     data.block_size=1024 \
#     train.epochs=10 \
#     experiment.seed=42 \
#     loader.num_workers=0 \
#     cuda_device=0 \
#     device=cuda:0 \
#     >>"$log" 2>&1 || return 1
# }

# run_mqar_wave() {
#   local mqar_cfg="$1"
#   local arch="$2"
#   local setting="$3"
#   local seed="$4"
#   local vocab="$5"
#   local minq="$6"
#   local log="$7"

#   echo "JOB mqar setting=$setting arch=$arch seed=$seed" | tee -a "$log"
#   local min_arg=()
#   if [[ "$minq" == "null" ]]; then
#     min_arg=(data.min_query_pos=null)
#   else
#     min_arg=(data.min_query_pos="$minq")
#   fi

#   "$PYTHON" code/train_mqar.py \
#     --config "$ROOT/$mqar_cfg" \
#     experiment.name="archeval_mqar_${setting}_${arch}_seed${seed}" \
#     logging.output_dir="outputs/ArchEval/mqar_${setting}_${arch}_seed${seed}" \
#     experiment.seed="$seed" \
#     data.input_seq_len=128 \
#     data.num_kv_pairs=16 \
#     data.vocab_size="$vocab" \
#     "${min_arg[@]}" \
#     data.train_examples=20000 \
#     data.val_examples=2000 \
#     data.test_examples=2000 \
#     data.fixed_examples=true \
#     data.random_non_queries=true \
#     train.lr=0.0003 \
#     train.epochs=20 \
#     train.early_stopping.patience=4 \
#     loader.num_workers=0 \
#     loader.pin_memory=false \
#     cuda_device=0 \
#     device=cuda:0 \
#     >>"$log" 2>&1 || return 1
# }

# run_timing_wave() {
#   local bench_cfg="$1"
#   local arch="$2"
#   local log="$3"
#   local outcsv="outputs/ArchEval/timing_${arch}.csv"

#   echo "JOB timing arch=$arch -> $outcsv" | tee -a "$log"
#   "$PYTHON" code/benchmark_timing.py \
#     --configs "$ROOT/$bench_cfg" \
#     --block-sizes 1024 \
#     --batch-sizes 1 8 16 \
#     --output-csv "$ROOT/$outcsv" \
#     --overwrite \
#     --set "experiment.name=archeval_timing_${arch}" \
#     --set device=cuda:0 \
#     --set cuda_device=0 \
#     >>"$log" 2>&1 || return 1
# }

# # Build JOB list as lines: lm|fields...
# JOB_LINES=()
# while IFS=$'\t' read -r kind arch atype c4 c5 c6 c7 c8 || [[ -n "${kind:-}" ]]; do
#   [[ -z "${kind:-}" ]] && continue
#   case "$kind" in
#     lm)
#       JOB_LINES+=("lm|$arch|$atype|$c4")
#       ;;
#     mqar)
#       JOB_LINES+=("mqar|$arch|$atype|$c4|$c5|$c6|$c7|$c8")
#       ;;
#     timing)
#       JOB_LINES+=("timing|$arch|$atype|$c4")
#       ;;
#     *)
#       echo "unknown manifest kind: $kind" >&2
#       exit 1
#       ;;
#   esac
# done <"$MANIFEST"

# n="${#JOB_LINES[@]}"
# echo "Total arch-eval jobs: $n"

# run_wave_shell() {
#   local start="$1"
#   local end="$2"
#   local pids=()
#   local failed=0
#   local i gpu

#   for ((i = start; i < end; i++)); do
#     gpu=$((i - start))
#     local line="${JOB_LINES[$i]}"
#     local log="logs/arch_eval/job_${i}_gpu${gpu}.log"
#     {
#       echo "========== JOB $((i + 1))/$n =========="
#       echo "$line"

#       local IFS='|'
#       read -ra P <<<"$line"

#       export CUDA_VISIBLE_DEVICES="$gpu"

#       case "${P[0]}" in
#         lm)
#           run_lm_wave "${P[1]}" "${P[3]}" "$log"
#           ;;
#         mqar)
#           run_mqar_wave "${P[3]}" "${P[1]}" "${P[4]}" "${P[5]}" "${P[6]}" "${P[7]}" "$log"
#           ;;
#         timing)
#           run_timing_wave "${P[3]}" "${P[1]}" "$log"
#           ;;
#         *)
#           echo "bad job kind ${P[0]}" >&2
#           exit 1
#           ;;
#       esac
#     } &
#     pids+=("$!")
#   done

#   for pid in "${pids[@]}"; do
#     if ! wait "$pid"; then
#       failed=1
#     fi
#   done

#   if [[ "$failed" -ne 0 ]]; then
#     echo "Some arch_eval jobs failed. See logs/arch_eval/job_*.log"
#     exit 1
#   fi
# }

# for ((start = 0; start < n; start += NUM_GPUS)); do
#   end=$((start + NUM_GPUS))
#   ((end > n)) && end=$n
#   echo "========== wave: jobs [$start, $end) =========="
#   run_wave_shell "$start" "$end"
# done

echo "========== aggregate =========="
SUMMARY_PATH="$ROOT/outputs/ArchEval/summary_arch_eval.csv"
if [[ "$IS_FULL_REGISTRY" -eq 1 ]]; then
  "$PYTHON" code/aggregate_arch_eval.py --registry "$ARCH_REGISTRY_ABS"
else
  OUT_NAME="summary_arch_eval_${REG_STEM}.csv"
  "$PYTHON" code/aggregate_arch_eval.py --registry "$ARCH_REGISTRY_ABS" --output-file "$OUT_NAME"
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
