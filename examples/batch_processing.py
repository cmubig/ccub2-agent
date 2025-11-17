#!/usr/bin/env python3
"""
Batch Processing Example: Process multiple prompts efficiently.

This example shows how to generate culturally-accurate images for
multiple prompts in batch, reusing initialized models for efficiency.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.vlm_detector import create_vlm_detector
from ccub2_agent.adapters.image_editing_adapter import create_adapter
from PIL import Image


def batch_generate(
    prompts: List[str],
    country: str = "korea",
    category: str = "general",
    t2i_model: str = "sd35",
    i2i_model: str = "qwen",
    max_iterations: int = 3,
    output_dir: str = "output/batch",
):
    """
    Generate culturally-accurate images for multiple prompts.

    Args:
        prompts: List of text prompts
        country: Target country
        category: Category for all prompts
        t2i_model: Text-to-Image model
        i2i_model: Image-to-Image editing model
        max_iterations: Maximum refinement iterations per image
        output_dir: Output directory for images

    Returns:
        List of results with metadata
    """
    print("=" * 80)
    print(f"BATCH PROCESSING: {len(prompts)} prompts")
    print("=" * 80)
    print(f"Country: {country}")
    print(f"Category: {category}")
    print(f"T2I Model: {t2i_model}")
    print(f"I2I Model: {i2i_model}")
    print(f"Max Iterations: {max_iterations}")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize models ONCE (reuse for all prompts)
    print("Initializing models (one-time setup)...")
    vlm = create_vlm_detector(
        country=country,
        vlm_model="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
    )
    t2i_adapter = create_adapter(model_type=t2i_model, quantization="4bit")
    i2i_adapter = create_adapter(model_type=i2i_model, quantization="4bit")
    print("✓ Models initialized")
    print()

    # Process each prompt
    results = []
    for idx, prompt in enumerate(tqdm(prompts, desc="Processing"), 1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(prompts)}] {prompt}")
        print("=" * 80)

        try:
            # Generate initial image
            print("Generating initial image...")
            image = t2i_adapter.generate(prompt=prompt, width=1024, height=1024)

            # Evaluate
            cultural_score, prompt_score, issues = vlm.score_cultural_quality(
                image=image,
                prompt=prompt,
                country=country,
                category=category,
            )
            print(f"Initial score: Cultural {cultural_score}/10, Prompt {prompt_score}/10")

            # Iterative refinement
            iteration = 0
            while cultural_score < 8.0 and iteration < max_iterations:
                iteration += 1
                print(f"\nIteration {iteration}: Refining...")

                # Retrieve references
                reference_paths = vlm.clip_rag.search_by_image(
                    query_image=image,
                    category=category,
                    top_k=3,
                )
                reference_images = [Image.open(p) for p in reference_paths]

                # Generate editing prompt
                editing_prompt = f"""Edit to improve cultural accuracy:
Issues: {', '.join(issues) if issues else 'General improvements'}
Target: {country} {category}"""

                # Edit image
                image = i2i_adapter.edit(
                    image=image,
                    prompt=editing_prompt,
                    reference_images=reference_images,
                )

                # Re-evaluate
                cultural_score, prompt_score, issues = vlm.score_cultural_quality(
                    image=image,
                    prompt=prompt,
                    country=country,
                    category=category,
                )
                print(f"Iteration {iteration} score: Cultural {cultural_score}/10")

            # Save result
            filename = f"{timestamp}_prompt{idx:03d}.png"
            output_file = output_path / filename
            image.save(output_file)

            result = {
                "prompt": prompt,
                "output_file": str(output_file),
                "cultural_score": cultural_score,
                "prompt_score": prompt_score,
                "iterations": iteration,
                "issues": issues,
                "status": "success",
            }
            results.append(result)

            print(f"✓ Saved to: {filename}")
            print(f"Final score: Cultural {cultural_score}/10 ({iteration} iterations)")

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "prompt": prompt,
                "output_file": None,
                "status": "error",
                "error": str(e),
            })

    # Save metadata
    metadata_file = output_path / f"{timestamp}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "country": country,
                "category": category,
                "t2i_model": t2i_model,
                "i2i_model": i2i_model,
                "max_iterations": max_iterations,
            },
            "results": results,
            "summary": {
                "total": len(prompts),
                "success": sum(1 for r in results if r["status"] == "success"),
                "error": sum(1 for r in results if r["status"] == "error"),
                "avg_cultural_score": sum(
                    r.get("cultural_score", 0) for r in results if r["status"] == "success"
                ) / max(sum(1 for r in results if r["status"] == "success"), 1),
                "avg_iterations": sum(
                    r.get("iterations", 0) for r in results if r["status"] == "success"
                ) / max(sum(1 for r in results if r["status"] == "success"), 1),
            }
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    success = sum(1 for r in results if r["status"] == "success")
    print(f"Processed: {len(prompts)} prompts")
    print(f"Success: {success}")
    print(f"Errors: {len(prompts) - success}")
    if success > 0:
        avg_score = sum(
            r.get("cultural_score", 0) for r in results if r["status"] == "success"
        ) / success
        avg_iters = sum(
            r.get("iterations", 0) for r in results if r["status"] == "success"
        ) / success
        print(f"Average Cultural Score: {avg_score:.2f}/10")
        print(f"Average Iterations: {avg_iters:.1f}")
    print(f"\nOutput: {output_dir}")
    print(f"Metadata: {metadata_file}")
    print("=" * 80)

    return results


def load_prompts_from_file(file_path: str) -> List[str]:
    """
    Load prompts from text file (one prompt per line).

    Args:
        file_path: Path to text file

    Returns:
        List of prompts
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return prompts


def load_prompts_from_json(file_path: str) -> List[Dict]:
    """
    Load prompts from JSON file with additional metadata.

    Expected format:
    [
        {"prompt": "...", "category": "...", "country": "..."},
        ...
    ]

    Args:
        file_path: Path to JSON file

    Returns:
        List of prompt dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process multiple prompts")
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file with prompts (txt or json)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Prompts as command-line arguments",
    )
    parser.add_argument("--country", type=str, default="korea", help="Target country")
    parser.add_argument(
        "--category",
        type=str,
        default="general",
        help="Category",
    )
    parser.add_argument(
        "--t2i-model",
        type=str,
        default="sd35",
        help="T2I model (sd35, flux, sdxl, gemini)",
    )
    parser.add_argument(
        "--i2i-model",
        type=str,
        default="qwen",
        help="I2I model (qwen, flux, sdxl, sd35)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max refinement iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/batch",
        help="Output directory",
    )

    args = parser.parse_args()

    # Load prompts
    if args.prompts_file:
        file_path = Path(args.prompts_file)
        if file_path.suffix == ".json":
            # JSON format with metadata
            data = load_prompts_from_json(args.prompts_file)
            # Process with different settings per prompt
            print("JSON batch processing not yet implemented in this example")
            print("Using first prompt only as demo:")
            prompts = [data[0]["prompt"]]
        else:
            # Simple text file
            prompts = load_prompts_from_file(args.prompts_file)
    elif args.prompts:
        prompts = args.prompts
    else:
        # Demo prompts
        print("No prompts provided. Using demo prompts...")
        prompts = [
            "A person wearing traditional Korean hanbok",
            "Traditional Korean bibimbap in a stone bowl",
            "Korean palace architecture with traditional roof tiles",
            "Korean traditional fan dance performance",
        ]

    # Run batch processing
    batch_generate(
        prompts=prompts,
        country=args.country,
        category=args.category,
        t2i_model=args.t2i_model,
        i2i_model=args.i2i_model,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
    )
