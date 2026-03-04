#!/bin/bash
# Download Qwen2.5-7B-Instruct model to cluster storage

set -e

MODEL_DIR="/fs/nexus-scratch/bhetwal/models/Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

echo "Downloading Qwen2.5-7B-Instruct model..."
echo "Target directory: $MODEL_DIR"

# Create model directory
mkdir -p "$MODEL_DIR"

# Activate your virtual environment
source /fs/nexus-scratch/bhetwal/vllm-env/bin/activate

# Download model using huggingface-cli
pip install -U "huggingface_hub[cli]"

# Download the model
huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False

echo "Model downloaded successfully to $MODEL_DIR"
