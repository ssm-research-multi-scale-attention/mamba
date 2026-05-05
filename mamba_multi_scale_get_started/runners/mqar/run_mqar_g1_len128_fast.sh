#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-8}"

OV=(
  data.train_examples=20000
  data.val_examples=2000
  data.test_examples=2000

  # easier MQAR than original len128/kv16/vocab1024
  data.input_seq_len=128
  data.num_kv_pairs=16
  data.vocab_size=512
  data.fixed_examples=true

  train.epochs=20
  train.early_stopping.patience=4

  loader.num_workers=0
  loader.pin_memory=false
)

CONFIGS=(
  configs/MQAR/mqar_mamba2_depth4_len128.yaml
  configs/MQAR/mqar_mamba2_depth6_len128.yaml
  configs/MQAR/mqar_ms_gated_stride2_len128.yaml
  configs/MQAR/mqar_ms_attention_stride4_len128.yaml
)

mkdir -p logs/mqar_parallel

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local failed=0

  for ((i=start; i<end; i++)); do
    local c="${CONFIGS[$i]}"
    local gpu=$(( i - start ))
    local base
    base="$(basename "$c" .yaml)"
    local name="${base}_easy_kv16_vocab_512"
    local log="logs/mqar_parallel/${name}.gpu${gpu}.log"

    echo "========== launch $name on physical GPU $gpu =========="

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" code/train_mqar.py \
      --config "$c" \
      experiment.name="$name" \
      logging.output_dir="outputs/MQAR/${name}" \
      cuda_device=0 \
      device=cuda:0 \
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
    echo "Some MQAR jobs failed. Check logs/mqar_parallel/*.log"
    exit 1
  fi
}

n="${#CONFIGS[@]}"

for ((start=0; start<n; start+=NUM_GPUS)); do
  end=$(( start + NUM_GPUS ))
  if (( end > n )); then
    end="$n"
  fi

  echo "========== wave: configs [$start, $end) =========="
  run_wave "$start" "$end"
done

SUMMARY="$ROOT/outputs/MQAR/summary_mqar_kv_16_vocab_512.csv"

echo "========== aggregate =========="
"$PYTHON" code/aggregate_mqar_results.py --output-file "$SUMMARY"

echo "summary: $SUMMARY"