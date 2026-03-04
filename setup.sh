#!/bin/bash
# Quick setup script for VQA with Qwen2-VL

set -e

echo "🚀 Setting up VQA with Qwen2-VL and vLLM"
echo "========================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install YAML support
pip install pyyaml

# Create examples directory if needed
mkdir -p examples

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the vLLM server: python scripts/start_vllm_server.py"
echo "3. Run inference: python scripts/vqa_inference.py --image IMAGE_PATH --question QUESTION"
echo ""
echo "Note: First run will download the model (~15GB for 7B variant)"
