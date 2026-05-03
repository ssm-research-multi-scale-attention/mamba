#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python code/train_lm.py --config configs/TinyShakespeare/tiny_shakespeare_lstm.yaml "$@"
