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

# Install huggingface_hub if needed
pip install -U huggingface_hub

# Download the model using Python
python3 << EOF
from huggingface_hub import snapshot_download

model_name = "$MODEL_NAME"
local_dir = "$MODEL_DIR"

print(f"Downloading {model_name} to {local_dir}...")
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Download complete")
EOF

echo "Model downloaded successfully to $MODEL_DIR"
