# M-HalDetect Evaluation Setup

Integration of M-HalDetect hallucination detection dataset with Qwen2-VL.

## Overview

M-HalDetect is a comprehensive multimodal hallucination detection dataset from AAAI 2024 with 16k fine-grained annotations on VQA examples. This setup allows you to evaluate Qwen2-VL for hallucinations.

## Quick Start

### 1. Setup Dataset (one-time)

```bash
bash setup_mhaldetect.sh
```

This will:
- Clone M-HalDetect repository
- Download COCO 2014 validation images (6.2GB)
- Set up directory structure at `/fs/nexus-scratch/bhetwal/data/`

### 2. Run Evaluation

```bash
sbatch evaluate_mhaldetect_gpu.sh
```

This will:
- Start vLLM server with Qwen2.5-7B-Instruct
- Run M-HalDetect evaluation
- Save results to `results/mhaldetect_evaluation_<job_id>.json`
- Automatically cleanup

## Files Created

### Scripts

- `scripts/evaluate_mhaldetect.py` - Main evaluation script
- `setup_mhaldetect.sh` - Dataset download and setup
- `evaluate_mhaldetect_gpu.sh` - SLURM job for GPU evaluation

### Outputs

Results are saved in `results/` directory with:
- Summary metrics (avg hallucination rate, class distribution)
- Per-sample predictions and ground truth
- Hallucination classification breakdown

## Evaluation Metrics

The evaluation provides:

1. **Hallucination Rate**: Percentage of inaccurate segments in responses
2. **Class Distribution**:
   - Accurate segments
   - Inaccurate objects
   - Inaccurate attributes
   - Inaccurate relations
   - Other inaccuracies

3. **Per-Sample Analysis**: Detailed predictions vs ground truth

## Manual Evaluation

For interactive testing:

```bash
# Start server
sbatch run_qwen_gpu.sh

# In another terminal, run evaluation
python scripts/evaluate_mhaldetect.py \
    --dataset /fs/nexus-scratch/bhetwal/data/mhaldetect/val_raw.json \
    --coco-images /fs/nexus-scratch/bhetwal/data/coco/val2014 \
    --max-samples 100 \
    --output results/test_eval.json
```

## Dataset Structure

```
/fs/nexus-scratch/bhetwal/data/
├── mhaldetect/
│   ├── train_raw.json        # M-HalDetect training set
│   ├── val_raw.json          # M-HalDetect validation set
│   └── README.md             # Dataset documentation
└── coco/
    └── val2014/              # COCO 2014 validation images
```

## Requirements

- COCO 2014 validation images (~6.2GB)
- M-HalDetect dataset (~few MB)
- GPU with sufficient VRAM (32GB+ recommended)
- vLLM environment with transformers>=4.47.0

## Troubleshooting

**Server fails to start:**
- Check transformers version: `pip install 'transformers>=4.47.0'`
- Verify model path exists: `/fs/nexus-scratch/bhetwal/models/Qwen/Qwen2.5-7B-Instruct`

**Images not found:**
- Ensure COCO images are in correct directory
- Check image ID format matches COCO_val2014_XXXXXXXXXXXX.jpg

**Out of memory:**
- Reduce `--gpu-memory-utilization` to 0.75
- Reduce `--max-tokens` in evaluation

## Reference

Paper: "Detecting and Preventing Hallucinations in Large Vision Language Models"
Authors: Anisha Gunjal, Jihan Yin, Erhan Bas
Venue: AAAI 2024
ArXiv: https://arxiv.org/abs/2308.06394
