#!/usr/bin/env python3
"""
Start vLLM server for Qwen2-VL model
"""

import argparse
import yaml
from pathlib import Path


def start_server(config_path: str):
    """Start vLLM server with configuration from YAML file"""

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    vllm_config = config['vllm_config']
    model_config = config['model_config']
    model_name = config['model_name']

    # Build vLLM command
    cmd_parts = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        f"--model {model_name}",
        f"--tensor-parallel-size {vllm_config['tensor_parallel_size']}",
        f"--gpu-memory-utilization {vllm_config['gpu_memory_utilization']}",
        f"--max-model-len {vllm_config['max_model_len']}",
        f"--max-num-seqs {vllm_config['max_num_seqs']}",
        f"--dtype {vllm_config['dtype']}",
        f"--host {vllm_config['host']}",
        f"--port {vllm_config['port']}",
    ]

    if model_config['trust_remote_code']:
        cmd_parts.append("--trust-remote-code")

    cmd = " ".join(cmd_parts)

    print(f"Starting vLLM server with command:")
    print(cmd)
    print("\nServer will be available at:")
    print(f"http://{vllm_config['host']}:{vllm_config['port']}/v1")

    import os
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start vLLM server for Qwen2-VL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen2_vl_config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()
    start_server(args.config)
