#!/bin/bash

TIME=${1:-"04:00:00"}
GPUS=${2:-"1"}
MEM=${3:-"64G"}
CPUS=${4:-"8"}

echo "🚀 Requesting GPU allocation..."

srun \
  --partition=gen-h100 \
  --gres=gpu:$GPUS \
  --time=$TIME \
  --mem=$MEM \
  --cpus-per-task=$CPUS \
  --pty bash -i -c '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate pyg
    
    echo ""
    echo "✓ Connected to GPU node: $(hostname)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
    
    exec bash -i
  '
