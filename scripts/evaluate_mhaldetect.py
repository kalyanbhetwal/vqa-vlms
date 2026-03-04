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

    def get_image_path(self, image_filename: str) -> Path:
        """Get full path to COCO image from filename"""
        return self.coco_images_path / image_filename

    def classify_annotations(self, annotations: List[Dict]) -> Dict[str, int]:
        """
        Classify annotations by label type

        Args:
            annotations: List of annotations with 'label' field

        Returns:
            Dictionary with counts for each label type
        """
        label_counts = {
            'accurate': 0,
            'inaccurate': 0,
            'analysis': 0
        }

        for annot in annotations:
            label = annot.get('label', 'UNKNOWN').upper()
            if label == 'ACCURATE':
                label_counts['accurate'] += 1
            elif label == 'INACCURATE':
                label_counts['inaccurate'] += 1
            elif label == 'ANALYSIS':
                label_counts['analysis'] += 1

        return label_counts

    def compute_hallucination_rate(self, annotations: List[Dict]) -> float:
        """
        Compute hallucination rate from annotations

        Args:
            annotations: List of annotations with 'label' field

        Returns:
            Hallucination rate (percentage of inaccurate annotations, excluding ANALYSIS)
        """
        if not annotations:
            return 0.0

        # Exclude ANALYSIS labels from calculation
        content_annotations = [a for a in annotations
                              if a.get('label', '').upper() != 'ANALYSIS']

        if not content_annotations:
            return 0.0

        total = len(content_annotations)
        inaccurate = sum(1 for a in content_annotations
                        if a.get('label', '').upper() == 'INACCURATE')

        return (inaccurate / total) * 100

    def evaluate_sample(self, sample: Dict, **kwargs) -> Dict:
        """
        Evaluate a single sample from M-HalDetect

        Args:
            sample: Single M-HalDetect sample with 'image', 'question', 'annotations'
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with prediction and evaluation metrics
        """
        image_filename = sample['image']
        question = sample['question']
        # Remove <image> token from question
        question_text = question.replace('<image>\n', '').replace('<image>', '')
        image_path = self.get_image_path(image_filename)

        if not image_path.exists():
            return {
                'image': image_filename,
                'error': f'Image not found: {image_path}',
                'skipped': True
            }

        # Get model prediction
        try:
            prediction = self.vqa.ask_question(
                str(image_path),
                question_text,
                **kwargs
            )
        except Exception as e:
            return {
                'image': image_filename,
                'error': str(e),
                'skipped': True
            }

        # Compute metrics from ground truth
        annotations = sample.get('annotations', [])
        ground_truth_response = sample.get('response', '')
        label_counts = self.classify_annotations(annotations)
        hallucination_rate = self.compute_hallucination_rate(annotations)

        return {
            'image': image_filename,
            'question': question_text,
            'prediction': prediction,
            'ground_truth_response': ground_truth_response,
            'ground_truth_annotations': annotations,
            'label_counts': label_counts,
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

        # Aggregate label counts
        total_label_counts = {
            'accurate': 0,
            'inaccurate': 0,
            'analysis': 0
        }

        for result in valid_results:
            for key in total_label_counts:
                total_label_counts[key] += result['label_counts'].get(key, 0)

        # Compute percentages
        total_annotations = sum(total_label_counts.values())
        label_percentages = {
            key: (count / total_annotations * 100) if total_annotations > 0 else 0
            for key, count in total_label_counts.items()
        }

        summary = {
            'total_samples': total_samples,
            'skipped_samples': len(results) - total_samples,
            'avg_hallucination_rate': avg_hallucination_rate,
            'total_label_counts': total_label_counts,
            'label_percentages': label_percentages,
            'total_annotations': total_annotations
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
    print(f"\nAnnotation label distribution:")
    print(f"  Total annotations: {summary['total_annotations']}")
    for label_name, percentage in summary['label_percentages'].items():
        count = summary['total_label_counts'][label_name]
        print(f"  {label_name}: {count} ({percentage:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
