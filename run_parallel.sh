#!/bin/bash
# Run a single experiment in a worktree copy on a specific GPU
# Usage: bash run_parallel.sh <gpu_id> <worktree_dir> <description>
set -e

GPU_ID=$1
WORKTREE=$2
DESC=$3

cd "$WORKTREE"
CUDA_VISIBLE_DEVICES=$GPU_ID uv run python evaluate.py --description "$DESC"
