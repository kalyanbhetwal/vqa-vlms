#!/bin/bash
#SBATCH --job-name=mhaldetect_eval
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-metzler
#SBATCH --qos=vulcan-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/mhaldetect_%j.out
#SBATCH --error=results/mhaldetect_%j.err

set -e

# Create output directory
mkdir -p results

# Load CUDA module
module load cuda/12.1.1

# Activate virtual environment
source /fs/nexus-scratch/bhetwal/vllm-env/bin/activate

# Upgrade transformers to version compatible with vLLM 0.11.2
#pip install 'transformers>=4.47.0' --quiet

# Dataset paths
MHALDETECT_DATASET="/fs/nexus-scratch/bhetwal/data/mhaldetect/val_raw.json"
COCO_IMAGES="/fs/nexus-scratch/bhetwal/data/coco/val2014"
MODEL_PATH="/fs/nexus-scratch/bhetwal/models/Qwen/Qwen2.5-7B-Instruct"

echo "Starting M-HalDetect evaluation pipeline"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Node: $SLURMD_NODENAME"

# Start vLLM server in background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code &

VLLM_PID=$!

# Wait for server to start
echo "Waiting for vLLM server to start..."
sleep 45

# Check if server is running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM server failed to start"
    exit 1
fi

# Test server is responding
echo "Testing server connection..."
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready"
        break
    fi
    echo "Waiting for server... (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Server failed to become ready"
    kill $VLLM_PID
    exit 1
fi

# Run M-HalDetect evaluation
echo ""
echo "Starting M-HalDetect evaluation..."
echo "Dataset: $MHALDETECT_DATASET"
echo "COCO images: $COCO_IMAGES"
echo ""

python scripts/evaluate_mhaldetect.py \
    --dataset "$MHALDETECT_DATASET" \
    --coco-images "$COCO_IMAGES" \
    --api-url "http://localhost:8000/v1" \
    --output "results/mhaldetect_evaluation_${SLURM_JOB_ID}.json" \
    --temperature 0.7 \
    --max-tokens 512

EVAL_EXIT_CODE=$?

# Cleanup
echo ""
echo "Shutting down vLLM server..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null || true

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Evaluation complete!"
    echo "Results saved to: results/mhaldetect_evaluation_${SLURM_JOB_ID}.json"
else
    echo ""
    echo "Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi
