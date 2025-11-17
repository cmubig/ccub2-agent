#!/usr/bin/env python3
"""
Basic Usage Example: Generate a culturally-accurate image.

This example demonstrates the simplest way to use CCUB2-Agent:
1. Generate initial image with T2I model
2. Evaluate cultural accuracy with VLM
3. Iteratively refine until cultural score >= 8/10
"""

import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.vlm_detector import create_vlm_detector
from ccub2_agent.adapters.image_editing_adapter import create_adapter
from PIL import Image


def generate_culturally_accurate_image(
    prompt: str,
    country: str = "korea",
    category: str = "traditional_clothing",
    t2i_model: str = "sd35",
    i2i_model: str = "qwen",
    max_iterations: int = 5,
    output_path: str = "output/basic_example.png",
):
    """
    Generate a culturally-accurate image with automatic refinement.

    Args:
        prompt: Text description of desired image
        country: Target country for cultural context
        category: Category (traditional_clothing, food, architecture, etc.)
        t2i_model: Text-to-Image model (sd35, flux, sdxl, gemini)
        i2i_model: Image-to-Image editing model (qwen, flux, sdxl, sd35)
        max_iterations: Maximum refinement iterations
        output_path: Where to save the final image

    Returns:
        Final image and metadata
    """
    print("=" * 80)
    print("CCUB2-Agent: Basic Usage Example")
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Country: {country}")
    print(f"Category: {category}")
    print(f"T2I Model: {t2i_model}")
    print(f"I2I Model: {i2i_model}")
    print()

    # Initialize components
    print("Initializing VLM detector and adapters...")
    vlm = create_vlm_detector(
        country=country,
        vlm_model="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
    )
    t2i_adapter = create_adapter(model_type=t2i_model, quantization="4bit")
    i2i_adapter = create_adapter(model_type=i2i_model, quantization="4bit")
    print("✓ Initialization complete")
    print()

    # Step 0: Generate initial image
    print("-" * 80)
    print("Step 0: Generating initial image with T2I model...")
    print("-" * 80)
    image = t2i_adapter.generate(prompt=prompt, width=1024, height=1024)
    print("✓ Initial image generated")
    print()

    # Evaluate initial image
    print("Evaluating cultural accuracy...")
    cultural_score, prompt_score, issues = vlm.score_cultural_quality(
        image=image,
        prompt=prompt,
        country=country,
        category=category,
    )
    print(f"Cultural Score: {cultural_score}/10")
    print(f"Prompt Score: {prompt_score}/10")
    if issues:
        print("Detected Issues:")
        for issue in issues:
            print(f"  - {issue}")
    print()

    # Iterative refinement
    iteration = 0
    while cultural_score < 8.0 and iteration < max_iterations:
        iteration += 1
        print("-" * 80)
        print(f"Step {iteration}: Refining image...")
        print("-" * 80)

        # Retrieve reference images
        print("Retrieving reference images from CLIP RAG...")
        reference_paths = vlm.clip_rag.search_by_image(
            query_image=image,
            category=category,
            top_k=3,
        )
        reference_images = [Image.open(p) for p in reference_paths]
        print(f"✓ Retrieved {len(reference_images)} reference images")

        # Generate editing prompt
        editing_prompt = f"""Edit the image to improve cultural accuracy:
- Address issues: {', '.join(issues) if issues else 'General improvements'}
- Maintain overall composition and subject
- Use reference images for authentic cultural details
Target: {country} {category}"""

        # Edit image
        print("Editing image with I2I model...")
        image = i2i_adapter.edit(
            image=image,
            prompt=editing_prompt,
            reference_images=reference_images,
        )
        print("✓ Image edited")
        print()

        # Re-evaluate
        print("Re-evaluating cultural accuracy...")
        cultural_score, prompt_score, issues = vlm.score_cultural_quality(
            image=image,
            prompt=prompt,
            country=country,
            category=category,
        )
        print(f"Cultural Score: {cultural_score}/10")
        print(f"Prompt Score: {prompt_score}/10")
        if issues:
            print("Remaining Issues:")
            for issue in issues:
                print(f"  - {issue}")
        print()

        if cultural_score >= 8.0:
            print("✓ Cultural accuracy threshold reached!")
            break

    # Save final image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print("=" * 80)
    print(f"✓ Final image saved to: {output_path}")
    print(f"Final Cultural Score: {cultural_score}/10")
    print(f"Iterations: {iteration}")
    print("=" * 80)

    return image, {
        "cultural_score": cultural_score,
        "prompt_score": prompt_score,
        "iterations": iteration,
        "issues": issues,
    }


if __name__ == "__main__":
    # Example 1: Korean traditional clothing
    print("Example 1: Korean Traditional Clothing")
    generate_culturally_accurate_image(
        prompt="A person wearing traditional Korean hanbok",
        country="korea",
        category="traditional_clothing",
        output_path="output/example_hanbok.png",
    )

    # Example 2: Korean food
    print("\n\nExample 2: Korean Food")
    generate_culturally_accurate_image(
        prompt="Traditional Korean bibimbap in a stone bowl",
        country="korea",
        category="food",
        output_path="output/example_bibimbap.png",
    )

    print("\n✓ All examples completed successfully!")
