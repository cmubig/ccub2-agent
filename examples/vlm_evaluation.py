#!/usr/bin/env python3
"""
VLM Evaluation Example: Evaluate existing images for cultural accuracy.

This example shows how to use the VLM Cultural Detector standalone
to evaluate any image without generating or editing.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.vlm_detector import create_vlm_detector


def evaluate_image(
    image_path: str,
    prompt: str,
    country: str = "korea",
    category: str = "general",
):
    """
    Evaluate a single image for cultural accuracy.

    Args:
        image_path: Path to image file
        prompt: Original prompt that should describe the image
        country: Target country for cultural evaluation
        category: Category for context (optional)

    Returns:
        Evaluation results (scores and issues)
    """
    print("=" * 80)
    print("VLM Cultural Evaluation")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Country: {country}")
    print(f"Category: {category}")
    print()

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"✓ Image loaded: {image.size[0]}x{image.size[1]}")
    print()

    # Initialize VLM detector
    print("Initializing VLM detector...")
    vlm = create_vlm_detector(
        country=country,
        vlm_model="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
    )
    print("✓ VLM detector initialized")
    print()

    # Evaluate image
    print("Evaluating cultural accuracy...")
    print()
    cultural_score, prompt_score, issues = vlm.score_cultural_quality(
        image=image,
        prompt=prompt,
        country=country,
        category=category,
    )

    # Display results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Cultural Score: {cultural_score}/10")
    print("  → Authenticity of cultural elements")
    print()
    print(f"Prompt Score: {prompt_score}/10")
    print("  → Alignment with original prompt")
    print()

    if cultural_score >= 8.0:
        print("✓ Cultural accuracy is ACCEPTABLE (score >= 8)")
    else:
        print("✗ Cultural accuracy NEEDS IMPROVEMENT (score < 8)")
    print()

    if issues:
        print("Detected Issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No specific issues detected.")
    print()

    print("=" * 80)

    return {
        "cultural_score": cultural_score,
        "prompt_score": prompt_score,
        "issues": issues,
    }


def batch_evaluate(
    image_paths: list,
    prompts: list,
    country: str = "korea",
    category: str = "general",
):
    """
    Evaluate multiple images in batch.

    Args:
        image_paths: List of paths to image files
        prompts: List of prompts corresponding to each image
        country: Target country
        category: Category for context

    Returns:
        List of evaluation results
    """
    assert len(image_paths) == len(prompts), "Number of images and prompts must match"

    print("=" * 80)
    print(f"BATCH EVALUATION: {len(image_paths)} images")
    print("=" * 80)
    print()

    # Initialize VLM detector once (reuse for all images)
    vlm = create_vlm_detector(
        country=country,
        vlm_model="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
    )

    results = []
    for idx, (img_path, prompt) in enumerate(zip(image_paths, prompts), 1):
        print(f"\n[{idx}/{len(image_paths)}] Evaluating: {img_path}")
        print(f"Prompt: {prompt}")
        print("-" * 80)

        image = Image.open(img_path).convert("RGB")
        cultural_score, prompt_score, issues = vlm.score_cultural_quality(
            image=image,
            prompt=prompt,
            country=country,
            category=category,
        )

        result = {
            "image_path": img_path,
            "prompt": prompt,
            "cultural_score": cultural_score,
            "prompt_score": prompt_score,
            "issues": issues,
        }
        results.append(result)

        print(f"Cultural: {cultural_score}/10 | Prompt: {prompt_score}/10")
        if issues:
            print(f"Issues: {len(issues)} detected")

    # Summary
    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)
    avg_cultural = sum(r["cultural_score"] for r in results) / len(results)
    avg_prompt = sum(r["prompt_score"] for r in results) / len(results)
    print(f"Average Cultural Score: {avg_cultural:.2f}/10")
    print(f"Average Prompt Score: {avg_prompt:.2f}/10")
    print(f"Images with score >= 8: {sum(1 for r in results if r['cultural_score'] >= 8)}/{len(results)}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images for cultural accuracy")
    parser.add_argument("--image-path", type=str, help="Path to image file")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Original prompt describing the image",
    )
    parser.add_argument("--country", type=str, default="korea", help="Target country")
    parser.add_argument(
        "--category",
        type=str,
        default="general",
        help="Category (traditional_clothing, food, architecture, etc.)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: evaluate all images in directory",
    )

    args = parser.parse_args()

    if args.batch:
        # Batch mode: evaluate all images in directory
        image_dir = Path(args.image_path or ".")
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

        if not image_paths:
            print(f"No images found in {image_dir}")
            sys.exit(1)

        # Use filename as prompt if not provided
        prompts = [args.prompt or f"Image: {p.stem}" for p in image_paths]

        batch_evaluate(
            image_paths=[str(p) for p in image_paths],
            prompts=prompts,
            country=args.country,
            category=args.category,
        )
    else:
        # Single image mode
        if not args.image_path:
            print("Error: --image-path required for single image evaluation")
            sys.exit(1)

        if not args.prompt:
            print("Warning: No prompt provided. Using filename as prompt.")
            args.prompt = Path(args.image_path).stem

        evaluate_image(
            image_path=args.image_path,
            prompt=args.prompt,
            country=args.country,
            category=args.category,
        )
