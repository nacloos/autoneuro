#!/bin/bash
set -euo pipefail

# Run training and capture METRIC lines
UV_CACHE_DIR=/storage/nacloos/.uv uv run python train.py 2>/dev/null
