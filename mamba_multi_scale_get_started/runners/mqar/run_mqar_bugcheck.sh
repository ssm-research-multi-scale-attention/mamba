#!/usr/bin/env bash
# G1-style diagnostic: 3 models × 2 LRs × 2 seeds = 12 runs; MQAR kv16 vocab512 len128.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-1}"

mkdir -p logs/mqar_bugcheck

MODEL_SPECS=(
  "mamba2_depth4:configs/MQAR/mqar_mamba2_depth4_len128.yaml"
  "mamba2_depth6:configs/MQAR/mqar_mamba2_depth6_len128.yaml"
  "ms_attention_stride4:configs/MQAR/mqar_ms_attention_stride4_len128.yaml"
)

LRS=(0.0003 0.0001)
LR_SLUGS=(lr3e4 lr1e4)
SEEDS=(42 43)

RUN_NAMES=()
RUN_CFGS=()
RUN_LR=()
RUN_SEED=()

for spec in "${MODEL_SPECS[@]}"; do
  key="${spec%%:*}"
  cfg="${spec#*:}"
  for li in 0 1; do
    lr="${LRS[$li]}"
    slug="${LR_SLUGS[$li]}"
    for seed in "${SEEDS[@]}"; do
      name="mqar_bugcheck_${key}_kv16_vocab512_${slug}_seed${seed}"
      RUN_NAMES+=("$name")
      RUN_CFGS+=("$cfg")
      RUN_LR+=("$lr")
      RUN_SEED+=("$seed")
    done
  done
done

OV_BASE=(
  data.input_seq_len=128
  data.num_kv_pairs=16
  data.vocab_size=512
  data.train_examples=20000
  data.val_examples=2000
  data.test_examples=2000
  data.fixed_examples=true
  train.epochs=20
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
    local name="${RUN_NAMES[$i]}"
    local c="${RUN_CFGS[$i]}"
    local lr="${RUN_LR[$i]}"
    local seed="${RUN_SEED[$i]}"
    local log="logs/mqar_bugcheck/${name}.gpu${gpu}.log"

    echo "========== [$i] $name  (physical GPU $gpu) =========="

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" code/train_mqar.py \
      --config "$c" \
      experiment.name="$name" \
      logging.output_dir="outputs/MQAR/${name}" \
      cuda_device=0 \
      device=cuda:0 \
      experiment.seed="$seed" \
      train.lr="$lr" \
      "${OV_BASE[@]}" \
      >"$log" 2>&1 &

    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "Some bugcheck jobs failed. See logs/mqar_bugcheck/*.log"
    exit 1
  fi
}

n="${#RUN_NAMES[@]}"
for ((start = 0; start < n; start += NUM_GPUS)); do
  end=$((start + NUM_GPUS))
  if ((end > n)); then
    end=$n
  fi
  echo "========== wave: runs [$start, $end) =========="
  run_wave "$start" "$end"
done

SUMMARY="$ROOT/outputs/MQAR/summary_mqar_bugcheck.csv"
echo "========== aggregate → $SUMMARY =========="
"$PYTHON" code/aggregate_mqar_results.py --output-file "$(basename "$SUMMARY")"
echo "summary: $SUMMARY"
