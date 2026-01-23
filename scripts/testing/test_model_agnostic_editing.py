#!/usr/bin/env python3
"""
Test Model-Agnostic Image Editing

Tests the full workflow with different I2I models:
1. Generate image with cultural problems
2. VLM detects issues
3. Select best reference image (single image)
4. Edit with reference
5. Re-evaluate improvement

Supports: Qwen (I2I), Qwen-Image (T2I), SDXL, Flux, SD3.5 Medium, Gemini (Nano Banana)
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
        model_type: 'qwen', 'sdxl', 'flux', 'sd35', or 'gemini'
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

    # Free VLM and CLIP memory before loading image generator (skip for API-based models)
    if model_type not in ['gemini'] and t2i_model not in ['gemini', 'qwen-t2i']:
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
    else:
        if 'gemini' in [model_type, t2i_model]:
            logger.info("0. Using API-based model (Gemini) - skipping memory management")
        else:
            logger.info("0. Loading Qwen-Image T2I model...")
        logger.info("")

    # Create adapter
    logger.info(f"1. Initializing {model_type} adapter (T2I: {t2i_model})...")
    adapter = create_adapter(model_type=model_type, t2i_model=t2i_model, device="auto")
    logger.info(f"✓ {model_type} adapter ready")
    logger.info("")

    # Generate initial image with REALISTIC prompt
    logger.info("2. Generating initial image...")

    # IMPORTANT: Add photorealistic keywords for better quality
    realistic_prompt = f"photorealistic professional photograph, {prompt}, realistic skin texture, detailed fabric, natural lighting, 8k uhd, high quality"

    try:
        initial_image = adapter.generate(prompt=realistic_prompt, width=1024, height=1024, seed=42)

        # Save initial image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_path = output_dir / f"t2i-{t2i_model}_i2i-{model_type}_{timestamp}_initial.png"
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

    # Get initial cultural scores (before any editing)
    logger.info("3. Evaluating initial quality...")
    try:
        initial_cultural, initial_prompt = vlm_detector.score_cultural_quality(
            image_path=initial_path,
            prompt=prompt,
            country=country,
        )
        logger.info(f"✓ Initial scores - Cultural: {initial_cultural}/10, Prompt: {initial_prompt}/10")
    except Exception as e:
        logger.error(f"✗ Scoring failed: {e}")
        return None

    logger.info("")

    # ITERATIVE REFINEMENT LOOP
    max_iterations = 3
    current_image = initial_path
    current_cultural = initial_cultural
    current_prompt_score = initial_prompt
    iteration_history = []
    previous_instruction = None  # Track previous editing instruction for context

    for iteration in range(1, max_iterations + 1):
        logger.info("="*80)
        logger.info(f"ITERATION {iteration}/{max_iterations}")
        logger.info("="*80)
        logger.info("")

        # Detect issues on current image
        logger.info(f"4. Detecting cultural issues (iteration {iteration})...")
        try:
            issues = vlm_detector.detect(
                image_path=current_image,
                prompt=prompt,
                country=country,
                editing_prompt=previous_instruction,  # Pass previous editing context
                category=category,
            )

            if issues:
                logger.info(f"✓ Detected {len(issues)} issues:")
                for i, issue in enumerate(issues[:5], 1):
                    logger.info(f"   {i}. {issue['description']}")
            else:
                logger.info("✓ No cultural issues detected!")
        except Exception as e:
            logger.error(f"✗ Issue detection failed: {e}")
            issues = []

        logger.info("")

        # Check early exit conditions
        # Exit ONLY if no issues detected (ignore score threshold)
        # We want to continue improving until VLM finds no issues
        if not issues:
            logger.info(f"✓ No issues detected! Current scores: Cultural {current_cultural}/10, Prompt {current_prompt_score}/10")
            logger.info("   Stopping iteration")
            logger.info("")
            break

        # Log current status
        logger.info(f"   Current scores: Cultural {current_cultural}/10, Prompt {current_prompt_score}/10")
        logger.info(f"   Continuing iteration to fix remaining {len(issues)} issues")
        logger.info("")

        # Select reference image
        logger.info(f"5. Selecting best reference image (iteration {iteration})...")
        try:
            reference = reference_selector.select_best_reference(
                query_image=current_image,
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
        logger.info(f"6. Generating model-specific editing instruction (iteration {iteration})...")

        # Build universal instruction first
        universal_instruction = f"Improve the cultural accuracy of the {category} in this {country} image."

        if issues:
            universal_instruction += f" Fix these issues: "
            for issue in issues[:3]:
                universal_instruction += f"{issue['description']}. "

        # Create editing context
        from ccub2_agent.modules.prompt_adapter import get_prompt_adapter, EditingContext

        # Get cultural context from reference if available
        cultural_context = ""
        if reference and 'description' in reference:
            cultural_context = reference['description']

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
        prompt_adapter = get_prompt_adapter()
        instruction = prompt_adapter.adapt(universal_instruction, model_type, context)

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

        # Edit image with RAG-based cultural guidance
        logger.info(f"7. Editing image with RAG cultural guidance (iteration {iteration})...")

        # Extract metadata for dynamic cultural context
        reference_metadata = reference.get('metadata') if reference else None
        if reference_metadata:
            logger.info(f"   Using RAG metadata with description: {reference_metadata.get('description_enhanced', 'N/A')[:80]}...")

        try:
            edited_image = adapter.edit(
                image=current_image,
                instruction=instruction,
                reference_image=Path(reference['image_path']) if reference else None,
                reference_metadata=reference_metadata,  # ← PASS RAG METADATA!
                strength=0.95,  # High strength for aggressive editing
                seed=42,
            )

            edited_path = output_dir / f"t2i-{t2i_model}_i2i-{model_type}_{timestamp}_iter{iteration}.png"
            edited_image.save(edited_path)
            logger.info(f"✓ Edited image saved: {edited_path}")
        except Exception as e:
            logger.error(f"✗ Image editing failed: {e}")
            break

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
        logger.info(f"8. Re-evaluating edited image (iteration {iteration})...")
        try:
            new_cultural, new_prompt = vlm_detector.score_cultural_quality(
                image_path=edited_path,
                prompt=prompt,
                country=country,
            )

            logger.info(f"✓ Iteration {iteration} scores - Cultural: {new_cultural}/10, Prompt: {new_prompt}/10")
            logger.info(f"   Change from previous: Cultural {new_cultural - current_cultural:+d}, Prompt {new_prompt - current_prompt_score:+d}")
            logger.info(f"   Change from initial: Cultural {new_cultural - initial_cultural:+d}, Prompt {new_prompt - initial_prompt:+d}")
        except Exception as e:
            logger.error(f"✗ Re-evaluation failed: {e}")
            break

        logger.info("")

        # Save iteration history
        iteration_history.append({
            'iteration': iteration,
            'image_path': str(edited_path),
            'cultural_score': new_cultural,
            'prompt_score': new_prompt,
            'issues_detected': len(issues),
            'reference_used': reference is not None,
            'cultural_change': new_cultural - current_cultural,
            'prompt_change': new_prompt - current_prompt_score,
        })

        # Check termination conditions
        # Only stop if BOTH scores don't improve (stuck)
        # Allow iteration to continue if at least one score improves
        if new_cultural < current_cultural and new_prompt < current_prompt_score:
            logger.info(f"⚠ Both scores decreased (Cultural: {new_cultural}/{current_cultural}, Prompt: {new_prompt}/{current_prompt_score})")
            logger.info("   Stopping iteration - quality degraded")
            logger.info("")
            # Don't use the worse version
            break

        # Update current state for next iteration
        current_image = edited_path
        current_cultural = new_cultural
        current_prompt_score = new_prompt
        previous_instruction = instruction  # Save instruction for next iteration's context

        # Reload CLIP for next iteration (needed for reference selection)
        if iteration < max_iterations:
            logger.info("Reloading CLIP for next iteration...")
            from ccub2_agent.modules.clip_image_rag import CLIPImageRAG
            clip_index_dir = Path("data/clip_index") / country
            if clip_index_dir.exists():
                vlm_detector.clip_rag = CLIPImageRAG(
                    index_dir=clip_index_dir,
                    model_name="openai/clip-vit-base-patch32",
                    device="cuda",
                )
                reference_selector.clip_rag = vlm_detector.clip_rag
            logger.info("✓ CLIP reloaded")
            logger.info("")

    # Final evaluation if we didn't just evaluate
    if not iteration_history or iteration_history[-1]['image_path'] != str(current_image):
        final_cultural = current_cultural
        final_prompt = current_prompt_score
    else:
        final_cultural = iteration_history[-1]['cultural_score']
        final_prompt = iteration_history[-1]['prompt_score']

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
        'iterations': len(iteration_history),
        'iteration_history': iteration_history,
        'initial_image': str(initial_path),
        'final_image': str(current_image),
    }

    logger.info("="*80)
    logger.info("RESULT SUMMARY")
    logger.info("="*80)
    logger.info(f"Model: {model_type}")
    logger.info(f"Iterations: {len(iteration_history)}/{max_iterations}")
    logger.info(f"Cultural: {initial_cultural} → {final_cultural} ({'+' if result['cultural_improvement'] >= 0 else ''}{result['cultural_improvement']})")
    logger.info(f"Prompt: {initial_prompt} → {final_prompt} ({'+' if result['prompt_improvement'] >= 0 else ''}{result['prompt_improvement']})")

    if iteration_history:
        logger.info("")
        logger.info("Iteration History:")
        for iter_data in iteration_history:
            logger.info(f"  Iter {iter_data['iteration']}: Cultural={iter_data['cultural_score']}/10 ({iter_data['cultural_change']:+d}), "
                       f"Prompt={iter_data['prompt_score']}/10 ({iter_data['prompt_change']:+d}), "
                       f"Issues={iter_data['issues_detected']}")

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
    print("  3. SD3.5 Medium - Excellent prompt understanding")
    print("  4. Qwen-Image - Text rendering specialist (Chinese/English)")
    print("  5. Gemini (Nano Banana) - API-based, requires GOOGLE_API_KEY")
    print("")
    while True:
        choice = input("Enter your choice [1-5] (default: 1): ").strip() or "1"
        if choice in ['1', '2', '3', '4', '5']:
            model_map = {'1': 'sdxl', '2': 'flux', '3': 'sd35', '4': 'qwen-t2i', '5': 'gemini'}
            t2i_model = model_map[choice]
            break
        print("Invalid choice. Please enter 1-5.")
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
    print("  4. SD3.5 Medium - Natural language understanding")
    print("  5. Gemini (Nano Banana) - Conversational editing, API-based")
    print("  6. ALL - Test all models (takes longer)")
    print("")
    while True:
        choice = input("Enter your choice [1-6] (default: 1): ").strip() or "1"
        if choice in ['1', '2', '3', '4', '5', '6']:
            model_map = {'1': 'qwen', '2': 'sdxl', '3': 'flux', '4': 'sd35', '5': 'gemini', '6': 'all'}
            model = model_map[choice]
            break
        print("Invalid choice. Please enter 1-6.")
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
        choices=['qwen', 'sdxl', 'flux', 'sd35', 'gemini', 'all'],
        help='I2I model to test (or "all" for all models)'
    )
    parser.add_argument(
        '--t2i-model',
        type=str,
        choices=['sdxl', 'flux', 'sd35', 'gemini', 'qwen-t2i'],
        help='T2I model for initial image generation (qwen-t2i for text rendering, gemini requires API key)'
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
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path.home() / "ccub2-agent-data",
        help='Data directory (default: ~/ccub2-agent-data)'
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
            args.t2i_model = 'sd35'
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
    data_dir = args.data_dir
    logger.info(f"Using data directory: {data_dir}")
    # Interactive mode only offers auto-initialization if prompt was not provided via CLI
    is_interactive = not hasattr(args, '_from_cli') or not args._from_cli
    if not check_initialization(args.country, data_dir, interactive=is_interactive):
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    logger.info(">>> Initializing components...")

    text_index_dir = data_dir / "cultural_index" / args.country
    clip_index_dir = data_dir / "clip_index" / args.country

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
    models = ['qwen', 'sdxl', 'flux', 'sd35', 'gemini'] if args.model == 'all' else [args.model]

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
            logger.info(f"  Iterations: {result['iterations']}/3")
            logger.info(f"  Cultural improvement: {result['cultural_improvement']:+d}")
            logger.info(f"  Prompt improvement: {result['prompt_improvement']:+d}")
            logger.info(f"  Final cultural score: {result['final_cultural']}/10")

        logger.info("")
        logger.info("="*80)


if __name__ == "__main__":
    main()
