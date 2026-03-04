#!/bin/bash
# Setup script for M-HalDetect dataset

set -e

DATA_DIR="/fs/nexus-scratch/bhetwal/data"
MHALDETECT_DIR="$DATA_DIR/mhaldetect"
COCO_DIR="$DATA_DIR/coco"

echo "Setting up M-HalDetect dataset..."
echo "Data directory: $DATA_DIR"

# Create directories
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Clone M-HalDetect repository
if [ ! -d "mhal-detect" ]; then
    echo "Cloning M-HalDetect repository..."
    git clone https://github.com/hendryx-scale/mhal-detect.git
    mv mhal-detect "$MHALDETECT_DIR"
else
    echo "M-HalDetect repository already exists"
fi

# Download COCO 2014 validation images
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

if [ ! -f "val2014.zip" ] && [ ! -d "val2014" ]; then
    echo "Downloading COCO 2014 validation images (6.2GB)..."
    wget http://images.cocodataset.org/zips/val2014.zip

    echo "Extracting images..."
    unzip val2014.zip

    echo "Cleaning up zip file..."
    rm val2014.zip
elif [ -d "val2014" ]; then
    echo "COCO 2014 validation images already exist"
else
    echo "Extracting COCO images..."
    unzip val2014.zip
fi

echo ""
echo "Setup complete!"
echo ""
echo "Dataset locations:"
echo "  M-HalDetect: $MHALDETECT_DIR"
echo "  COCO images: $COCO_DIR/val2014"
echo ""
echo "Dataset files available:"
echo "  - train_raw.json (training set)"
echo "  - val_raw.json (validation set)"
echo ""
echo "To run evaluation:"
echo "  python scripts/evaluate_mhaldetect.py \\"
echo "    --dataset $MHALDETECT_DIR/val_raw.json \\"
echo "    --coco-images $COCO_DIR/val2014"
echo ""
