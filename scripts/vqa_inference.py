#!/usr/bin/env python3
"""
VQA inference using Qwen2-VL with vLLM backend
Supports both local vLLM engine and API server
"""

import argparse
import base64
import json
from pathlib import Path
from typing import List, Dict, Union
import requests
from PIL import Image
from io import BytesIO


class VQAInference:
    """Visual Question Answering using Qwen2-VL"""

    def __init__(self, mode: str = "api", api_url: str = "http://localhost:8000/v1", model_name: str = None):
        """
        Initialize VQA inference engine

        Args:
            mode: "api" for vLLM server or "local" for direct vLLM engine
            api_url: URL of vLLM API server (for api mode)
            model_name: Model name/path (for local mode)
        """
        self.mode = mode
        self.api_url = api_url

        if mode == "local":
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_name or "Qwen/Qwen2-VL-7B-Instruct",
                trust_remote_code=True,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
            )
            self.SamplingParams = SamplingParams

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def ask_question_api(self, image_path: str, question: str, **kwargs) -> str:
        """
        Ask a question about an image using vLLM API server

        Args:
            image_path: Path to image file
            question: Question to ask about the image
            **kwargs: Additional generation parameters

        Returns:
            Answer string
        """
        # Encode image
        base64_image = self.encode_image(image_path)

        # Prepare request
        headers = {"Content-Type": "application/json"}

        # Qwen2-VL uses special tokens for images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

        payload = {
            "model": "/fs/nexus-scratch/bhetwal/models/Qwen/Qwen2.5-7B-Instruct",  # Match the loaded model path
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Make request
        response = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def ask_question_local(self, image_path: str, question: str, **kwargs) -> str:
        """
        Ask a question about an image using local vLLM engine

        Args:
            image_path: Path to image file
            question: Question to ask about the image
            **kwargs: Additional generation parameters

        Returns:
            Answer string
        """
        from qwen_vl_utils import process_vision_info

        # Prepare messages in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare sampling parameters
        sampling_params = self.SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512),
            repetition_penalty=kwargs.get("repetition_penalty", 1.05),
        )

        # Generate
        outputs = self.llm.generate(
            {
                "prompt": messages,
                "multi_modal_data": {
                    "image": image_inputs[0] if image_inputs else None,
                },
            },
            sampling_params=sampling_params,
        )

        return outputs[0].outputs[0].text

    def ask_question(self, image_path: str, question: str, **kwargs) -> str:
        """
        Ask a question about an image

        Args:
            image_path: Path to image file
            question: Question to ask about the image
            **kwargs: Additional generation parameters

        Returns:
            Answer string
        """
        if self.mode == "api":
            return self.ask_question_api(image_path, question, **kwargs)
        else:
            return self.ask_question_local(image_path, question, **kwargs)

    def batch_vqa(self, qa_pairs: List[Dict[str, str]], **kwargs) -> List[str]:
        """
        Process multiple VQA pairs

        Args:
            qa_pairs: List of dicts with 'image' and 'question' keys
            **kwargs: Additional generation parameters

        Returns:
            List of answers
        """
        answers = []
        for pair in qa_pairs:
            answer = self.ask_question(pair["image"], pair["question"], **kwargs)
            answers.append(answer)
        return answers


def main():
    parser = argparse.ArgumentParser(description="VQA inference with Qwen2-VL")
    parser.add_argument("--mode", type=str, default="api", choices=["api", "local"],
                        help="Inference mode: 'api' for vLLM server, 'local' for direct engine")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM API server URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Model name/path (for local mode)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image file")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask about the image")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate")

    args = parser.parse_args()

    # Initialize VQA engine
    vqa = VQAInference(mode=args.mode, api_url=args.api_url, model_name=args.model)

    # Ask question
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    print("Generating answer...\n")

    answer = vqa.ask_question(
        args.image,
        args.question,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
