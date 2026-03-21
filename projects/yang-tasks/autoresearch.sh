#!/bin/bash
set -euo pipefail

# Run training and capture METRIC lines
python train.py 2>/dev/null
