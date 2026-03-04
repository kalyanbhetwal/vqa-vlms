#!/bin/bash
# One-time script to upgrade vLLM in your environment
# Run this interactively or as a separate job

set -e

echo "Upgrading vLLM environment..."

# Load CUDA module
module load cuda/12.1.1

# Set CUDA_HOME for compilation
export CUDA_HOME=/opt/local/stow/cuda-12.1.1
export PATH=$CUDA_HOME/bin:$PATH

# Activate virtual environment
source /fs/nexus-scratch/bhetwal/vllm-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install numpy first
pip install numpy

# Upgrade vLLM (this will take several minutes)
echo "Upgrading vLLM (this may take 10-15 minutes)..."
pip install --upgrade vllm

# Upgrade transformers
pip install --upgrade transformers

echo "Upgrade complete. You can now run run_qwen_gpu.sh"
