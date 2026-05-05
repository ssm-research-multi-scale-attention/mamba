#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-8}"

OV_BASE=(
  data.input_seq_len=128
  data.num_kv_pairs=16
  data.vocab_size=640
  data.train_examples=20000
  data.val_examples=2000
  data.test_examples=2000
  data.fixed_examples=true
  data.random_non_queries=true

  train.epochs=20
  train.lr=0.0003
  train.early_stopping.patience=4

  loader.num_workers=0
  loader.pin_memory=false
)

JOBS=(
  "mamba2_depth4:configs/MQAR/mqar_mamba2_depth4_len128.yaml:42"
  "mamba2_depth4:configs/MQAR/mqar_mamba2_depth4_len128.yaml:43"
  "ms_attention_stride4:configs/MQAR/mqar_ms_attention_stride4_len128.yaml:42"
  "ms_attention_stride4:configs/MQAR/mqar_ms_attention_stride4_len128.yaml:43"
)

mkdir -p logs/mqar_parallel

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local failed=0

  for ((i=start; i<end; i++)); do
    IFS=":" read -r model_name config seed <<< "${JOBS[$i]}"
    local gpu=$(( i - start ))

    local exp_name="mqar_hard_rerun_${model_name}_kv16_vocab_640_lr3e4_seed${seed}"
    local log="logs/mqar_parallel/${exp_name}.gpu${gpu}.log"

    echo "========== launch $exp_name on physical GPU $gpu =========="

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" code/train_mqar.py \
      --config "$config" \
      experiment.name="$exp_name" \
      logging.output_dir="outputs/MQAR/${exp_name}" \
      experiment.seed="$seed" \
      cuda_device=0 \
      device=cuda:0 \
      "${OV_BASE[@]}" \
      > "$log" 2>&1 &

    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "Some MQAR hard rerun jobs failed. Check logs/mqar_parallel/*.log"
    exit 1
  fi
}

n="${#JOBS[@]}"

for ((start=0; start<n; start+=NUM_GPUS)); do
  end=$(( start + NUM_GPUS ))
  if (( end > n )); then
    end="$n"
  fi

  echo "========== wave: jobs [$start, $end) =========="
  run_wave "$start" "$end"
done

SUMMARY="$ROOT/outputs/MQAR/summary_mqar_hard_rerun_vocab_640.csv"

echo "========== aggregate =========="
"$PYTHON" code/aggregate_mqar_results.py --output-file "$SUMMARY"

echo "summary: $SUMMARY"