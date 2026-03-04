#!/usr/bin/env python3
"""
Simple example of VQA using Qwen2-VL
"""

import sys
sys.path.append('..')

from scripts.vqa_inference import VQAInference


def main():
    # Initialize VQA engine (using API server)
    vqa = VQAInference(mode="api", api_url="http://localhost:8000/v1")

    # Single question example
    image_path = "examples/sample_image.jpg"
    question = "What objects are in this image?"

    print("Single VQA Example")
    print("=" * 50)
    print(f"Question: {question}")

    answer = vqa.ask_question(image_path, question)
    print(f"Answer: {answer}\n")

    # Multiple questions on same image
    questions = [
        "What is the main subject of this image?",
        "What colors are prominent in this image?",
        "Is this image taken indoors or outdoors?",
    ]

    print("Multiple Questions Example")
    print("=" * 50)

    for q in questions:
        answer = vqa.ask_question(image_path, q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    # Batch processing
    qa_pairs = [
        {"image": "examples/image1.jpg", "question": "What is in this image?"},
        {"image": "examples/image2.jpg", "question": "Describe the scene."},
        {"image": "examples/image3.jpg", "question": "Count the objects."},
    ]

    print("Batch Processing Example")
    print("=" * 50)

    answers = vqa.batch_vqa(qa_pairs)
    for pair, answer in zip(qa_pairs, answers):
        print(f"Image: {pair['image']}")
        print(f"Q: {pair['question']}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
