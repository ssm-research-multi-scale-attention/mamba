#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p logs/mqar_final

COMMON_OV=(
  train.lr=0.0003
  train.epochs=20
  train.early_stopping.patience=4

  data.fixed_examples=true
  data.random_non_queries=true
  data.num_kv_pairs=16
  data.train_examples=20000
  data.val_examples=2000
  data.test_examples=2000

  loader.num_workers=0
  loader.pin_memory=false
)

MODELS=(
  "mamba2_depth4:configs/MQAR/mqar_mamba2_depth4_len128.yaml"
  "ms_gated_stride2:configs/MQAR/mqar_ms_gated_stride2_len128.yaml"
  "ms_attention_stride4:configs/MQAR/mqar_ms_attention_stride4_len128.yaml"
)

SEEDS=(42 43 44)

# format:
# group|input_seq_len|min_query_pos|vocab_size
SETTINGS=(
  "easy|128|null|512"
  "transition704|128|null|704"
  "transition768|128|null|768"
)

JOBS=()

for setting in "${SETTINGS[@]}"; do
  IFS="|" read -r group L minq vocab <<< "$setting"

  for model_spec in "${MODELS[@]}"; do
    IFS=":" read -r model_name config <<< "$model_spec"

    for seed in "${SEEDS[@]}"; do
      exp_name="mqar_final_${group}_${model_name}_L${L}_min${minq}_kv16_vocab${vocab}_train20000_seed${seed}"
      JOBS+=("${exp_name}|${config}|${model_name}|${seed}|${L}|${minq}|${vocab}")
    done
  done
done

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local failed=0

  for ((i=start; i<end; i++)); do
    IFS="|" read -r exp_name config model_name seed L minq vocab <<< "${JOBS[$i]}"
    local gpu=$(( i - start ))
    local log="logs/mqar_final/${exp_name}.gpu${gpu}.log"

    echo "========== launch $exp_name on physical GPU $gpu =========="

    OV=(
      experiment.name="$exp_name"
      logging.output_dir="outputs/MQAR/${exp_name}"
      experiment.seed="$seed"

      data.input_seq_len="$L"
      data.vocab_size="$vocab"

      cuda_device=0
      device=cuda:0
    )

    if [[ "$minq" == "null" ]]; then
      OV+=(data.min_query_pos=null)
    else
      OV+=(data.min_query_pos="$minq")
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" code/train_mqar.py \
      --config "$config" \
      "${COMMON_OV[@]}" \
      "${OV[@]}" \
      > "$log" 2>&1 &

    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "Some MQAR final jobs failed. Check logs/mqar_final/*.log"
    exit 1
  fi
}

n="${#JOBS[@]}"
echo "Total MQAR final jobs: $n"

for ((start=0; start<n; start+=NUM_GPUS)); do
  end=$(( start + NUM_GPUS ))
  if (( end > n )); then
    end="$n"
  fi

  echo "========== wave: jobs [$start, $end) =========="
  run_wave "$start" "$end"
done

echo "========== aggregate =========="
"$PYTHON" code/aggregate_mqar_results.py --output-file summary_mqar_final.csv

echo "========== analyze =========="
"$PYTHON" code/analyze_mqar_sweep.py \
  --summary outputs/MQAR/summary_mqar_final.csv \
  --out-analyzed outputs/MQAR/summary_mqar_final_analyzed.csv \
  --report outputs/MQAR/mqar_final_report.txt

echo "summary:  $ROOT/outputs/MQAR/summary_mqar_final.csv"
echo "analyzed: $ROOT/outputs/MQAR/summary_mqar_final_analyzed.csv"
echo "report:   $ROOT/outputs/MQAR/mqar_final_report.txt"