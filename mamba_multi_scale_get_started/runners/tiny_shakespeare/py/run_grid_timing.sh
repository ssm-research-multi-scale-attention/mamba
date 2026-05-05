#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON="${PYTHON:-python}"
"$PYTHON" run_grid_timing.py --grid-config grid_configs/D_timing.yaml
