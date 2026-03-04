#!/usr/bin/env python3
"""
Evaluate Qwen2-VL on M-HalDetect dataset for hallucination detection
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from vqa_inference import VQAInference


class MHalDetectEvaluator:
    """Evaluator for M-HalDetect hallucination detection dataset"""

    def __init__(
        self,
        dataset_path: str,
        coco_images_path: str,
        api_url: str = "http://localhost:8000/v1"
    ):
        """
        Initialize M-HalDetect evaluator

        Args:
            dataset_path: Path to M-HalDetect dataset JSON file
            coco_images_path: Path to COCO 2014 validation images directory
            api_url: URL of vLLM API server
        """
        self.dataset_path = Path(dataset_path)
        self.coco_images_path = Path(coco_images_path)
        self.vqa = VQAInference(mode="api", api_url=api_url)

        # Load dataset
        print(f"Loading M-HalDetect dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} examples")

    def get_image_path(self, image_id: int) -> Path:
        """Get full path to COCO image from image ID"""
        # COCO 2014 validation images are named like COCO_val2014_000000XXXXXX.jpg
        image_filename = f"COCO_val2014_{image_id:012d}.jpg"
        return self.coco_images_path / image_filename

    def classify_segments(self, segments: List[Dict]) -> Dict[str, int]:
        """
        Classify segments by hallucination type

        Args:
            segments: List of segment annotations with 'class' labels

        Returns:
            Dictionary with counts for each class
        """
        class_counts = {
            'accurate': 0,
            'inaccurate_object': 0,
            'inaccurate_attribute': 0,
            'inaccurate_relation': 0,
            'inaccurate_other': 0
        }

        for seg in segments:
            seg_class = seg.get('class', 'unknown').lower()
            if seg_class == 'accurate':
                class_counts['accurate'] += 1
            elif 'object' in seg_class or 'entity' in seg_class:
                class_counts['inaccurate_object'] += 1
            elif 'attribute' in seg_class or 'description' in seg_class:
                class_counts['inaccurate_attribute'] += 1
            elif 'relation' in seg_class:
                class_counts['inaccurate_relation'] += 1
            elif 'inaccurate' in seg_class:
                class_counts['inaccurate_other'] += 1

        return class_counts

    def compute_hallucination_rate(self, segments: List[Dict]) -> float:
        """
        Compute hallucination rate from segment annotations

        Args:
            segments: List of segment annotations

        Returns:
            Hallucination rate (percentage of inaccurate segments)
        """
        if not segments:
            return 0.0

        total = len(segments)
        inaccurate = sum(1 for seg in segments
                        if 'inaccurate' in seg.get('class', '').lower())

        return (inaccurate / total) * 100

    def evaluate_sample(self, sample: Dict, **kwargs) -> Dict:
        """
        Evaluate a single sample from M-HalDetect

        Args:
            sample: Single M-HalDetect sample with 'image_id', 'question', 'segments'
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with prediction and evaluation metrics
        """
        image_id = sample['image_id']
        question = sample['question']
        image_path = self.get_image_path(image_id)

        if not image_path.exists():
            return {
                'image_id': image_id,
                'error': f'Image not found: {image_path}',
                'skipped': True
            }

        # Get model prediction
        try:
            prediction = self.vqa.ask_question(
                str(image_path),
                question,
                **kwargs
            )
        except Exception as e:
            return {
                'image_id': image_id,
                'error': str(e),
                'skipped': True
            }

        # Compute metrics from ground truth
        segments = sample.get('segments', [])
        class_counts = self.classify_segments(segments)
        hallucination_rate = self.compute_hallucination_rate(segments)

        return {
            'image_id': image_id,
            'question': question,
            'prediction': prediction,
            'ground_truth_segments': segments,
            'class_counts': class_counts,
            'hallucination_rate': hallucination_rate,
            'skipped': False
        }

    def evaluate_dataset(
        self,
        max_samples: int = None,
        output_file: str = None,
        **kwargs
    ) -> Dict:
        """
        Evaluate entire M-HalDetect dataset

        Args:
            max_samples: Maximum number of samples to evaluate (None for all)
            output_file: Path to save detailed results JSON
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with aggregate metrics
        """
        samples = self.data[:max_samples] if max_samples else self.data
        results = []

        print(f"Evaluating {len(samples)} samples...")

        for sample in tqdm(samples, desc="Evaluating"):
            result = self.evaluate_sample(sample, **kwargs)
            results.append(result)

        # Compute aggregate metrics
        valid_results = [r for r in results if not r.get('skipped', False)]

        total_samples = len(valid_results)
        avg_hallucination_rate = np.mean([
            r['hallucination_rate'] for r in valid_results
        ])

        # Aggregate class counts
        total_class_counts = {
            'accurate': 0,
            'inaccurate_object': 0,
            'inaccurate_attribute': 0,
            'inaccurate_relation': 0,
            'inaccurate_other': 0
        }

        for result in valid_results:
            for key in total_class_counts:
                total_class_counts[key] += result['class_counts'].get(key, 0)

        # Compute percentages
        total_segments = sum(total_class_counts.values())
        class_percentages = {
            key: (count / total_segments * 100) if total_segments > 0 else 0
            for key, count in total_class_counts.items()
        }

        summary = {
            'total_samples': total_samples,
            'skipped_samples': len(results) - total_samples,
            'avg_hallucination_rate': avg_hallucination_rate,
            'total_class_counts': total_class_counts,
            'class_percentages': class_percentages,
            'total_segments': total_segments
        }

        # Save detailed results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'results': results
                }, f, indent=2)
            print(f"\nDetailed results saved to {output_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2-VL on M-HalDetect dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to M-HalDetect dataset JSON file"
    )
    parser.add_argument(
        "--coco-images",
        type=str,
        required=True,
        help="Path to COCO 2014 validation images directory"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API server URL"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mhaldetect_evaluation.json",
        help="Path to save detailed results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MHalDetectEvaluator(
        dataset_path=args.dataset,
        coco_images_path=args.coco_images,
        api_url=args.api_url
    )

    # Run evaluation
    summary = evaluator.evaluate_dataset(
        max_samples=args.max_samples,
        output_file=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Print summary
    print("\n" + "="*60)
    print("M-HALDETECT EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples evaluated: {summary['total_samples']}")
    print(f"Skipped samples: {summary['skipped_samples']}")
    print(f"\nAverage hallucination rate: {summary['avg_hallucination_rate']:.2f}%")
    print(f"\nSegment class distribution:")
    print(f"  Total segments: {summary['total_segments']}")
    for class_name, percentage in summary['class_percentages'].items():
        count = summary['total_class_counts'][class_name]
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
