#!/usr/bin/env bash
# MQAR difficulty grid: Groups A/B/C/D × 3 backbones (51 jobs), waves over NUM_GPUS.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-1}"

MODEL_TAG=(mamba2_depth4 ms_gated_stride2 ms_attention_stride4)
MODEL_CFG=(
  configs/MQAR/mqar_mamba2_depth4_len128.yaml
  configs/MQAR/mqar_ms_gated_stride2_len128.yaml
  configs/MQAR/mqar_ms_attention_stride4_len128.yaml
)

mkdir -p logs/mqar_difficulty_sweep

# Parallel arrays: job name | config path | extra Hydra overrides (pipe-separated)
JN=()
JC=()
JE=()

add_job() {
  JN+=("$1")
  JC+=("$2")
  JE+=("$3")
}

# Group A: length, no delay
for L in 128 160 192 224 256; do
  for mi in 0 1 2; do
    add_job \
      "mqar_sweep_A_${MODEL_TAG[$mi]}_L${L}_minNone_kv16_vocab512_train20000_seed42" \
      "${MODEL_CFG[$mi]}" \
      "data.input_seq_len=${L}|data.min_query_pos=null|data.num_kv_pairs=16|data.vocab_size=512|data.train_examples=20000"
  done
done

# Group B: delay at L=256
for MQP in 48 64 96 128; do
  for mi in 0 1 2; do
    add_job \
      "mqar_sweep_B_${MODEL_TAG[$mi]}_L256_min${MQP}_kv16_vocab512_train20000_seed42" \
      "${MODEL_CFG[$mi]}" \
      "data.input_seq_len=256|data.min_query_pos=${MQP}|data.num_kv_pairs=16|data.vocab_size=512|data.train_examples=20000"
  done
done

# Group C: vocab at L=128
for VOC in 512 576 640 704 768; do
  for mi in 0 1 2; do
    add_job \
      "mqar_sweep_C_${MODEL_TAG[$mi]}_L128_minNone_kv16_vocab${VOC}_train20000_seed42" \
      "${MODEL_CFG[$mi]}" \
      "data.input_seq_len=128|data.min_query_pos=null|data.num_kv_pairs=16|data.vocab_size=${VOC}|data.train_examples=20000"
  done
done

# Group D: less train data
for L in 128 160 192; do
  for mi in 0 1 2; do
    add_job \
      "mqar_sweep_D_${MODEL_TAG[$mi]}_L${L}_minNone_kv16_vocab512_train5000_seed42" \
      "${MODEL_CFG[$mi]}" \
      "data.input_seq_len=${L}|data.min_query_pos=null|data.num_kv_pairs=16|data.vocab_size=512|data.train_examples=5000"
  done
done

COMMON=(
  experiment.seed=42
  train.lr=0.0003
  train.epochs=20
  train.early_stopping.patience=4
  data.fixed_examples=true
  data.random_non_queries=true
  data.val_examples=2000
  data.test_examples=2000
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
    local name="${JN[$i]}"
    local c="${JC[$i]}"
    local extra="${JE[$i]}"
    local log="logs/mqar_difficulty_sweep/${name}.gpu${gpu}.log"

    IFS='|' read -r -a EXA <<< "$extra"

    echo "========== [$i/${#JN[@]}] $name  (physical GPU $gpu) =========="

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" code/train_mqar.py \
      --config "$c" \
      experiment.name="$name" \
      logging.output_dir="outputs/MQAR/${name}" \
      cuda_device=0 \
      device=cuda:0 \
      "${COMMON[@]}" \
      "${EXA[@]}" \
      >"$log" 2>&1 &

    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "Some difficulty sweep jobs failed. See logs/mqar_difficulty_sweep/*.log"
    exit 1
  fi
}

n="${#JN[@]}"
for ((start = 0; start < n; start += NUM_GPUS)); do
  end=$((start + NUM_GPUS))
  if ((end > n)); then
    end=$n
  fi
  echo "========== wave: jobs [$start, $end) of $n =========="
  run_wave "$start" "$end"
done

SUMMARY="$ROOT/outputs/MQAR/summary_mqar_difficulty_sweep.csv"
echo "========== aggregate → $SUMMARY =========="
"$PYTHON" code/aggregate_mqar_results.py --output-file "$(basename "$SUMMARY")"
echo "summary: $SUMMARY"
"$PYTHON" code/analyze_mqar_sweep.py
echo "Analyzed → outputs/MQAR/summary_mqar_difficulty_sweep_analyzed.csv"
echo "Report   → outputs/MQAR/mqar_sweep_report.txt"
