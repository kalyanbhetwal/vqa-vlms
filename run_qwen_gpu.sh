#!/bin/bash
#SBATCH --job-name=vllm_qwen
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-metzler
#SBATCH --qos=vulcan-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=qwen/server_%j.out
#SBATCH --error=qwen/server_%j.err

set -e

# Create output directory if needed
mkdir -p qwen

# Load CUDA module
module load cuda/12.1.1

# Activate virtual environment
source /fs/nexus-scratch/bhetwal/vllm-env/bin/activate

echo "Starting vLLM server with Qwen2.5-7B-Instruct"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $SLURMD_NODENAME"

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model /fs/nexus-scratch/bhetwal/models/Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code &

VLLM_PID=$!

# Wait for server to start
echo "Waiting for vLLM server to start..."
sleep 30

# Check if server is running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM server failed to start"
    exit 1
fi

echo "vLLM server started successfully"

# Server is now running
echo "VQA server is ready at http://localhost:8000"
echo "Server will run until job time limit (3 hours)"
echo "To use: python scripts/vqa_inference.py --mode api --image IMAGE.jpg --question 'Your question'"

# Wait for server to finish (keeps job alive)
wait $VLLM_PID
