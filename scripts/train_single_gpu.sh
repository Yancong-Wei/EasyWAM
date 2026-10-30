#!/usr/bin/env bash
# Low-level ZeRO-1 launcher on one GPU. Extra args are Hydra overrides.
# For the A800 CPU-offload baseline, prefer scripts/robodojo_pipeline.py.
# Example:
#   bash scripts/train_single_gpu.sh task=robodojo_easywam_mot_wan22_a800
set -euo pipefail

unset OMP_NUM_THREADS || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

exec bash scripts/train_zero1.sh "$@"
