echo "Running Mamba2 (12 layers)..."
OUT=outputs/TinyShakespeare/tiny_shakespeare_mamba2_12layers_random_windows
mkdir -p "$OUT"
"$PYTHON" code/train_lm.py \
  --config configs/TinyShakespeare/tiny_shakespeare_mamba2.yaml \
  'model.layer_headdims=[32,32,32,32,32,32,32,32,32,32,32,32]' \
  experiment.name=tiny_shakespeare_mamba2_12layers_random_windows \
  logging.output_dir="$OUT" \
  2>&1 | tee "$OUT/run.log"