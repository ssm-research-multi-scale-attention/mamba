#!/usr/bin/env bash
# Same pipeline; LSTM via model.backbone (configs/lstm_baseline.yaml).
python mamba2_heads_imdb.py --config configs/lstm_baseline.yaml >logs/run_lstm_baseline.log
