#!/usr/bin/env bash
# Delayed MQAR: L=256, queries only at t>=64 (kv16, vocab512, seed 42).
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-1}"

mkdir -p logs/mqar_delayed_len_sweep

RUNS=(
  "mqar_delayed_l256_min64_mamba2_depth4_kv16_vocab512_seed42:configs/MQAR/mqar_mamba2_depth4_len128.yaml"
  "mqar_delayed_l256_min64_ms_gated_stride2_kv16_vocab512_seed42:configs/MQAR/mqar_ms_gated_stride2_len128.yaml"
  "mqar_delayed_l256_min64_ms_attention_stride4_kv16_vocab512_seed42:configs/MQAR/mqar_ms_attention_stride4_len128.yaml"
)

OV=(
  experiment.seed=42
  data.input_seq_len=256
  data.min_query_pos=64
  data.num_kv_pairs=16
  data.vocab_size=512
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

run_wave() {
  local start="$1"
  local end="$2"
  local pids=()
  local failed=0
  local i

  for ((i = start; i < end; i++)); do
    local gpu=$((i - start))
    local spec="${RUNS[$i]}"
    local name="${spec%%:*}"
    local c="${spec#*:}"
    local log="logs/mqar_delayed_len_sweep/${name}.gpu${gpu}.log"

    echo "========== [$i] $name  (physical GPU $gpu) =========="

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" code/train_mqar.py \
      --config "$c" \
      experiment.name="$name" \
      logging.output_dir="outputs/MQAR/${name}" \
      cuda_device=0 \
      device=cuda:0 \
      "${OV[@]}" \
      >"$log" 2>&1 &

    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "Some delayed MQAR jobs failed. See logs/mqar_delayed_len_sweep/*.log"
    exit 1
  fi
}

n="${#RUNS[@]}"
for ((start = 0; start < n; start += NUM_GPUS)); do
  end=$((start + NUM_GPUS))
  if ((end > n)); then
    end=$n
  fi
  echo "========== wave: runs [$start, $end) =========="
  run_wave "$start" "$end"
done

SUMMARY="$ROOT/outputs/MQAR/summary_mqar_delayed.csv"
echo "========== aggregate → $SUMMARY =========="
"$PYTHON" code/aggregate_mqar_results.py --output-file "$(basename "$SUMMARY")"
echo "summary: $SUMMARY"
