#!/usr/bin/env python3
"""
Test Model-Agnostic Image Editing

Tests the full workflow with different I2I models:
1. Generate image with cultural problems
2. VLM detects issues
3. Select best reference image (1장)
4. Edit with reference
5. Re-evaluate improvement

Supports: Qwen, SDXL, Flux
"""

import argparse
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.vlm_detector import create_vlm_detector
from ccub2_agent.modules.reference_selector import create_reference_selector
from ccub2_agent.adapters import create_adapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_full_workflow(
    prompt: str,
    model_type: str,
    country: str,
    category: str,
    vlm_detector,
    reference_selector,
    output_dir: Path,
    t2i_model: str = "sdxl",
):
    """
    Test full workflow with one model.

    Args:
        prompt: Generation prompt
        model_type: 'qwen', 'sdxl', or 'flux'
        country: Target country
        category: Image category
        vlm_detector: VLM detector instance
        reference_selector: Reference selector instance
        output_dir: Output directory
    """
    import gc
    import torch

    logger.info("="*80)
    logger.info(f"TESTING: {model_type.upper()}")
    logger.info("="*80)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Country: {country}, Category: {category}")
    logger.info("")

    # Free VLM and CLIP memory before loading image generator
    logger.info("0. Freeing VLM and CLIP memory temporarily...")
    vlm_obj = vlm_detector.vlm
    clip_obj = vlm_detector.clip_rag
    vlm_detector.vlm = None
    vlm_detector.clip_rag = None
    del vlm_obj
    del clip_obj
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✓ Memory freed")
    logger.info("")

    # Create adapter
    logger.info(f"1. Initializing {model_type} adapter (T2I: {t2i_model})...")
    adapter = create_adapter(model_type=model_type, t2i_model=t2i_model, device="auto")
    logger.info(f"✓ {model_type} adapter ready")
    logger.info("")

    # Generate initial image
    logger.info("2. Generating initial image...")
    try:
        initial_image = adapter.generate(prompt=prompt, width=1024, height=1024, seed=42)

        # Save initial image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_path = output_dir / f"{model_type}_{timestamp}_initial.png"
        initial_image.save(initial_path)
        logger.info(f"✓ Initial image saved: {initial_path}")
    except Exception as e:
        logger.error(f"✗ Image generation failed: {e}")
        return None

    logger.info("")

    # Free adapter memory before reloading VLM
    logger.info("Freeing adapter memory...")
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✓ Adapter memory freed")
    logger.info("")

    # Reload VLM and CLIP for detection
    logger.info("Reloading VLM and CLIP...")
    from metric.cultural_metric.enhanced_cultural_metric_pipeline import EnhancedVLMClient
    from ccub2_agent.modules.clip_image_rag import CLIPImageRAG
    from pathlib import Path

    vlm_detector.vlm = EnhancedVLMClient(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
    )

    clip_index_dir = Path("data/clip_index") / country
    if clip_index_dir.exists():
        vlm_detector.clip_rag = CLIPImageRAG(
            index_dir=clip_index_dir,
            model_name="openai/clip-vit-base-patch32",
            device="cuda",
        )
        reference_selector.clip_rag = vlm_detector.clip_rag

    logger.info("✓ VLM and CLIP reloaded")
    logger.info("")

    # Detect issues
    logger.info("3. Detecting cultural issues...")
    try:
        issues = vlm_detector.detect(
            image_path=initial_path,
            prompt=prompt,
            country=country,
            editing_prompt=None,
            category=category,
        )

        logger.info(f"✓ Detected {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            logger.info(f"   {i}. [{issue['type']}] {issue['description']} (severity: {issue['severity']}/10)")
    except Exception as e:
        logger.error(f"✗ Issue detection failed: {e}")
        return None

    logger.info("")

    # Get cultural scores
    logger.info("4. Evaluating initial quality...")
    try:
        initial_cultural, initial_prompt = vlm_detector.score_cultural_quality(
            image_path=initial_path,
            prompt=prompt,
            country=country,
        )
        logger.info(f"✓ Initial scores - Cultural: {initial_cultural}/5, Prompt: {initial_prompt}/5")
    except Exception as e:
        logger.error(f"✗ Scoring failed: {e}")
        return None

    logger.info("")

    # Select reference image
    logger.info("5. Selecting best reference image (1장)...")
    try:
        reference = reference_selector.select_best_reference(
            query_image=initial_path,
            issues=issues,
            category=category,
            k=10,
        )

        if reference:
            logger.info(f"✓ Selected: {Path(reference['image_path']).name}")
            logger.info(f"   Similarity: {reference['similarity']:.1%}")
            logger.info(f"   Reason: {reference['reason']}")
        else:
            logger.warning("⚠ No reference image found")
            reference = None
    except Exception as e:
        logger.error(f"✗ Reference selection failed: {e}")
        reference = None

    logger.info("")

    # Generate editing instruction with MODEL-SPECIFIC optimization
    logger.info("6. Generating model-specific editing instruction...")

    # Build universal instruction first
    universal_instruction = f"Improve the cultural accuracy of the {category} in this {country} image."

    if issues:
        universal_instruction += f" Fix these issues: "
        for issue in issues[:3]:
            universal_instruction += f"{issue['description']}. "

    # Create editing context
    from ccub2_agent.modules.prompt_adapter import get_prompt_adapter, EditingContext

    context = EditingContext(
        original_prompt=prompt,
        detected_issues=issues,
        cultural_elements=cultural_context if cultural_context else "",
        reference_images=[reference['image_path']] if reference else None,
        country=country,
        category=category,
        preserve_identity=True
    )

    # Get model-specific optimized prompt!
    adapter = get_prompt_adapter()
    instruction = adapter.adapt(universal_instruction, model_type, context)

    logger.info(f"Universal instruction: {universal_instruction}")
    logger.info(f"Model-specific ({model_type}) instruction:\n{instruction}")
    logger.info("")

    # Free VLM and CLIP memory before editing
    logger.info("Freeing VLM and CLIP memory for editing...")
    vlm_obj = vlm_detector.vlm
    clip_obj = vlm_detector.clip_rag
    vlm_detector.vlm = None
    vlm_detector.clip_rag = None
    del vlm_obj
    del clip_obj
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✓ VLM and CLIP memory freed")
    logger.info("")

    # Reload adapter for editing
    logger.info("Reloading adapter for editing...")
    adapter = create_adapter(model_type=model_type, t2i_model=t2i_model, device="auto")
    logger.info("✓ Adapter reloaded")
    logger.info("")

    # Edit image
    logger.info("7. Editing image with reference...")
    try:
        edited_image = adapter.edit(
            image=initial_path,
            instruction=instruction,
            reference_image=Path(reference['image_path']) if reference else None,
            strength=0.8,
            seed=42,
        )

        edited_path = output_dir / f"{model_type}_{timestamp}_edited.png"
        edited_image.save(edited_path)
        logger.info(f"✓ Edited image saved: {edited_path}")
    except Exception as e:
        logger.error(f"✗ Image editing failed: {e}")
        return None

    logger.info("")

    # Free adapter memory before re-evaluation
    logger.info("Freeing adapter memory for re-evaluation...")
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✓ Adapter memory freed")
    logger.info("")

    # Reload VLM for re-evaluation (no CLIP needed for scoring)
    logger.info("Reloading VLM for re-evaluation...")
    from metric.cultural_metric.enhanced_cultural_metric_pipeline import EnhancedVLMClient
    vlm_detector.vlm = EnhancedVLMClient(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
    )
    logger.info("✓ VLM reloaded")
    logger.info("")

    # Re-evaluate
    logger.info("8. Re-evaluating edited image...")
    try:
        final_cultural, final_prompt = vlm_detector.score_cultural_quality(
            image_path=edited_path,
            prompt=prompt,
            country=country,
        )

        logger.info(f"✓ Final scores - Cultural: {final_cultural}/5, Prompt: {final_prompt}/5")
        logger.info(f"   Improvement: Cultural +{final_cultural - initial_cultural}, Prompt +{final_prompt - initial_prompt}")
    except Exception as e:
        logger.error(f"✗ Re-evaluation failed: {e}")
        return None

    logger.info("")

    # Summary
    result = {
        'model': model_type,
        'prompt': prompt,
        'initial_cultural': initial_cultural,
        'initial_prompt': initial_prompt,
        'final_cultural': final_cultural,
        'final_prompt': final_prompt,
        'cultural_improvement': final_cultural - initial_cultural,
        'prompt_improvement': final_prompt - initial_prompt,
        'issues_detected': len(issues),
        'reference_used': reference is not None,
        'initial_image': str(initial_path),
        'edited_image': str(edited_path),
    }

    logger.info("="*80)
    logger.info("RESULT SUMMARY")
    logger.info("="*80)
    logger.info(f"Model: {model_type}")
    logger.info(f"Cultural: {initial_cultural} → {final_cultural} ({'+' if result['cultural_improvement'] >= 0 else ''}{result['cultural_improvement']})")
    logger.info(f"Prompt: {initial_prompt} → {final_prompt} ({'+' if result['prompt_improvement'] >= 0 else ''}{result['prompt_improvement']})")
    logger.info(f"Issues detected: {len(issues)}")
    logger.info(f"Reference used: {'✓' if reference else '✗'}")
    logger.info("="*80)
    logger.info("")

    return result


def interactive_mode():
    """Interactive CLI mode for user-friendly configuration."""
    print("")
    print("="*80)
    print("CCUB2 AGENT - MODEL-AGNOSTIC IMAGE EDITING")
    print("="*80)
    print("")
    print("Welcome! Let's configure your image editing workflow.")
    print("")

    # 1. T2I Model Selection
    print("─" * 80)
    print("STEP 1: Select Text-to-Image (T2I) Model")
    print("─" * 80)
    print("Which model should generate the initial image?")
    print("")
    print("  1. SDXL (Stable Diffusion XL) - Fast, balanced")
    print("  2. FLUX - High quality, slower")
    print("")
    while True:
        choice = input("Enter your choice [1-2] (default: 1): ").strip() or "1"
        if choice in ['1', '2']:
            t2i_model = 'sdxl' if choice == '1' else 'flux'
            break
        print("Invalid choice. Please enter 1 or 2.")
    print(f"✓ Selected T2I model: {t2i_model.upper()}")
    print("")

    # 2. I2I Model Selection
    print("─" * 80)
    print("STEP 2: Select Image-to-Image (I2I) Model")
    print("─" * 80)
    print("Which model should edit the image for cultural accuracy?")
    print("")
    print("  1. Qwen Image Edit - Text rendering, detailed")
    print("  2. SDXL - Balanced, versatile")
    print("  3. FLUX Kontext - Context preservation")
    print("  4. ALL - Test all models (takes longer)")
    print("")
    while True:
        choice = input("Enter your choice [1-4] (default: 1): ").strip() or "1"
        if choice in ['1', '2', '3', '4']:
            model_map = {'1': 'qwen', '2': 'sdxl', '3': 'flux', '4': 'all'}
            model = model_map[choice]
            break
        print("Invalid choice. Please enter 1-4.")
    print(f"✓ Selected I2I model: {model.upper()}")
    print("")

    # 3. Country Selection (dynamic based on available data)
    print("─" * 80)
    print("STEP 3: Select Target Country")
    print("─" * 80)
    print("Which country's cultural authenticity should we aim for?")
    print("")

    # Detect available countries from contributions
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "01_setup"))
    from detect_available_countries import detect_available_countries, get_country_display_name

    contributions_csv = PROJECT_ROOT / "data" / "_contributions.csv"
    available_countries = detect_available_countries(contributions_csv)

    if available_countries:
        # Show top 10 countries + Other option
        country_list = sorted(available_countries.items(), key=lambda x: -x[1])[:10]
        for i, (country, count) in enumerate(country_list, 1):
            display = get_country_display_name(country)
            print(f"  {i}. {display:<25} ({count} contributions)")
        print(f"  {len(country_list) + 1}. Other (enter manually)")
        print("")

        max_choice = len(country_list) + 1
        while True:
            choice = input(f"Enter your choice [1-{max_choice}] (default: 1): ").strip() or "1"
            try:
                choice_num = int(choice)
                if choice_num == max_choice:
                    country = input("Enter country name: ").strip().lower()
                    if country:
                        print(f"⚠️  Warning: '{country}' may not have data in contributions.csv")
                        break
                    print("Country name cannot be empty.")
                elif 1 <= choice_num <= len(country_list):
                    country = country_list[choice_num - 1][0]
                    break
                else:
                    print(f"Invalid choice. Please enter 1-{max_choice}.")
            except ValueError:
                print(f"Invalid input. Please enter a number 1-{max_choice}.")
    else:
        # Fallback if detection fails
        print("  (No countries detected in contributions.csv)")
        country = input("Enter country name: ").strip().lower()
        while not country:
            print("  Country name is required!")
            country = input("Enter country name: ").strip().lower()

    print(f"✓ Selected country: {get_country_display_name(country)}")
    print("")

    # 4. Category Selection
    print("─" * 80)
    print("STEP 4: Select Image Category")
    print("─" * 80)
    print("What category of image are you generating?")
    print("")
    print("  1. Traditional Clothing")
    print("  2. Food")
    print("  3. Architecture")
    print("  4. General / Other")
    print("")
    while True:
        choice = input("Enter your choice [1-4] (default: 4): ").strip() or "4"
        if choice in ['1', '2', '3', '4']:
            category_map = {'1': 'traditional_clothing', '2': 'food', '3': 'architecture', '4': 'general'}
            category = category_map[choice]
            break
        print("Invalid choice. Please enter 1-4.")
    print(f"✓ Selected category: {category.replace('_', ' ').title()}")
    print("")

    # 5. Prompt Input
    print("─" * 80)
    print("STEP 5: Enter Generation Prompt")
    print("─" * 80)
    print("Enter the prompt for image generation.")
    print("Example: \"A person wearing traditional clothing in a cultural setting\"")
    print("")
    while True:
        prompt = input("Prompt: ").strip()
        if prompt:
            break
        print("Prompt cannot be empty. Please enter a valid prompt.")
    print(f"✓ Prompt: {prompt}")
    print("")

    # Summary
    print("="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"  T2I Model:  {t2i_model.upper()}")
    print(f"  I2I Model:  {model.upper()}")
    print(f"  Country:    {country.capitalize()}")
    print(f"  Category:   {category.replace('_', ' ').title()}")
    print(f"  Prompt:     {prompt}")
    print("="*80)
    print("")

    confirm = input("Proceed with this configuration? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y' and confirm != 'yes':
        print("Cancelled.")
        sys.exit(0)

    print("")
    print("Starting workflow...")
    print("")

    return {
        'prompt': prompt,
        'model': model,
        't2i_model': t2i_model,
        'country': country,
        'category': category,
        'output_dir': PROJECT_ROOT / "results" / "model_agnostic_tests"
    }


def check_initialization(country: str, data_dir: Path, interactive: bool = True) -> bool:
    """Check if dataset is initialized and offer to initialize if missing."""
    required_paths = [
        data_dir / "country_packs" / country / "approved_dataset_enhanced.json",
        data_dir / "cultural_knowledge" / f"{country}_knowledge.json",
        data_dir / "cultural_index" / country / "faiss.index",
    ]

    missing = [p for p in required_paths if not p.exists()]

    if not missing:
        return True

    # Dataset is not initialized
    print("")
    print("="*80)
    print("⚠️  DATASET NOT INITIALIZED")
    print("="*80)
    print("")
    print("The following required files are missing:")
    for path in missing:
        print(f"  ❌ {path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path}")
    print("")
    print("This is normal for first-time setup.")
    print("")

    if interactive:
        print("Would you like to initialize the dataset now?")
        print("(This will download images, process data, and build indices - takes ~2-5 hours)")
        print("")
        choice = input("Initialize now? [Y/n]: ").strip().lower()

        if not choice or choice in ['y', 'yes']:
            print("")
            print("Starting initialization...")
            print("")

            # Run init script
            import subprocess
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "init_dataset.py"),
                "--country", country,
                "--data-dir", str(data_dir)
            ]

            try:
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    print("")
                    print("✅ Initialization complete! Starting workflow...")
                    print("")
                    return True
                else:
                    print("")
                    print("❌ Initialization failed. Please check errors above.")
                    return False
            except Exception as e:
                print(f"❌ Error running initialization: {e}")
                return False
        else:
            print("")
            print("You can initialize manually later by running:")
            print(f"  python scripts/init_dataset.py --country {country}")
            print("")
            return False
    else:
        # Non-interactive mode: just show instruction
        print("Please run the initialization script first:")
        print("")
        print(f"  python scripts/init_dataset.py --country {country}")
        print("")
        print("="*80)
        print("")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test model-agnostic image editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first-time users)
  python scripts/04_testing/test_model_agnostic_editing.py

  # Command-line mode
  python scripts/04_testing/test_model_agnostic_editing.py \\
    --prompt "A person in traditional clothing" \\
    --model qwen \\
    --t2i-model sdxl \\
    --country korea \\
    --category traditional_clothing
        """
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Image generation prompt'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['qwen', 'sdxl', 'flux', 'all'],
        help='I2I model to test (or "all" for all models)'
    )
    parser.add_argument(
        '--t2i-model',
        type=str,
        choices=['sdxl', 'flux'],
        help='T2I model for initial image generation'
    )
    parser.add_argument(
        '--country',
        type=str,
        help='Target country'
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Image category'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / "results" / "model_agnostic_tests",
        help='Output directory'
    )

    args = parser.parse_args()

    # Check if running in interactive mode (no required args provided)
    if not args.prompt:
        config = interactive_mode()
        args.prompt = config['prompt']
        args.model = config['model']
        args.t2i_model = config['t2i_model']
        args.country = config['country']
        args.category = config['category']
        args.output_dir = config['output_dir']
        args._from_cli = False
    else:
        # Validate required args in CLI mode
        if not args.model:
            args.model = 'qwen'
        if not args.t2i_model:
            args.t2i_model = 'sdxl'
        if not args.country:
            args.country = 'korea'
        if not args.category:
            args.category = 'general'
        args._from_cli = True

    logger.info("="*80)
    logger.info("MODEL-AGNOSTIC IMAGE EDITING TEST")
    logger.info("="*80)
    logger.info("")

    # Check if dataset is initialized
    data_dir = PROJECT_ROOT / "data"
    # Interactive mode only offers auto-initialization if prompt was not provided via CLI
    is_interactive = not hasattr(args, '_from_cli') or not args._from_cli
    if not check_initialization(args.country, data_dir, interactive=is_interactive):
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    logger.info(">>> Initializing components...")

    text_index_dir = PROJECT_ROOT / "data" / "cultural_index" / args.country
    clip_index_dir = PROJECT_ROOT / "data" / "clip_index" / args.country

    vlm_detector = create_vlm_detector(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        index_dir=text_index_dir if text_index_dir.exists() else None,
        clip_index_dir=clip_index_dir if clip_index_dir.exists() else None,
        load_in_4bit=True,
        debug=True,
    )

    reference_selector = create_reference_selector(
        clip_rag=vlm_detector.clip_rag,
        quality_weight=0.2,
    )

    logger.info("✓ Components initialized")
    logger.info("")

    # Test models
    models = ['qwen', 'sdxl', 'flux'] if args.model == 'all' else [args.model]

    results = []
    for model_type in models:
        try:
            result = test_full_workflow(
                prompt=args.prompt,
                model_type=model_type,
                country=args.country,
                category=args.category,
                vlm_detector=vlm_detector,
                reference_selector=reference_selector,
                output_dir=args.output_dir,
                t2i_model=args.t2i_model,
            )

            if result:
                results.append(result)

        except Exception as e:
            logger.error(f"Test failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    if len(results) > 1:
        logger.info("")
        logger.info("="*80)
        logger.info("COMPARISON ACROSS MODELS")
        logger.info("="*80)

        for result in results:
            logger.info(f"\n{result['model'].upper()}:")
            logger.info(f"  Cultural improvement: {result['cultural_improvement']:+d}")
            logger.info(f"  Prompt improvement: {result['prompt_improvement']:+d}")
            logger.info(f"  Final cultural score: {result['final_cultural']}/5")

        logger.info("")
        logger.info("="*80)


if __name__ == "__main__":
    main()
