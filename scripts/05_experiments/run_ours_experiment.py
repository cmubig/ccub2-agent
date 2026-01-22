#!/usr/bin/env python3
"""
OURS Experiment Script

Runs our full pipeline (VLM Detector → I2I Editor) on bias baseline images
with 1 iteration to generate "ours" results for comparison with closed-source models.

Usage:
    python run_ours_experiment.py --country china
    python run_ours_experiment.py --country korea
    python run_ours_experiment.py --country all
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = PROJECT_ROOT / "base_experimental"
OUTPUT_DIR = PROJECT_ROOT / "base_experimental" / "ours"
CSV_PATH = BASE_DIR / "edit-prompt.csv"


def load_prompts_from_csv() -> List[Dict[str, str]]:
    """Load edit prompts from CSV file."""
    prompts = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                'prompt': row['I2I_prompt'],
                'base_path': row['base_path'],
                'output_path': row['output_path'].replace('{model_name}', 'ours')
            })
    return prompts


def generate_korea_prompts() -> List[Dict[str, str]]:
    """Generate prompts for Korea images (not in CSV)."""
    korea_dir = BASE_DIR / "Korea"
    prompts = []

    # Category mappings for prompt generation
    category_prompts = {
        'architecture_house': 'traditional house',
        'architecture_landmark': 'landmark',
        'art_dance': 'traditional dance',
        'art_painting': 'traditional painting',
        'art_sculpture': 'traditional sculpture',
        'event_festival': 'traditional festival',
        'event_funeral': 'funeral',
        'event_religious_ritual': 'religious ritual',
        'event_wedding': 'traditional wedding',
        'fashion_accessories': 'traditional accessories',
        'fashion_clothing': 'traditional clothing',
        'fashion_makeup': 'traditional makeup',
        'food_dessert': 'traditional dessert',
        'food_main_dish': 'traditional main dish',
        'food_snack': 'traditional snack',
        'people_president': 'president',
        'people_soldier': 'soldier',
        'wildlife_animal_national': 'national animal',
        'wildlife_plant_national': 'national plant',
    }

    for img_file in korea_dir.glob("*.png"):
        filename = img_file.name
        # Parse filename: flux_korea_food_snack_traditional.png
        parts = filename.replace('.png', '').split('_')

        # Extract category (e.g., food_snack or architecture_house)
        if len(parts) >= 4:
            # Find category parts (everything after korea and before last modifier)
            category_parts = parts[2:-1]  # e.g., ['food', 'snack'] or ['architecture', 'house']
            category = '_'.join(category_parts)

            # Get prompt template
            prompt_desc = category_prompts.get(category, category.replace('_', ' '))
            prompt = f"Change the image to represent {prompt_desc} in Korea."

            prompts.append({
                'prompt': prompt,
                'base_path': f"korea/{filename}",
                'output_path': f"ours/{filename}"
            })

    return prompts


def extract_country_from_path(base_path: str) -> str:
    """Extract country from base_path."""
    if 'china' in base_path.lower():
        return 'china'
    elif 'korea' in base_path.lower():
        return 'korea'
    return 'unknown'


def extract_category_from_filename(filename: str) -> str:
    """Extract category from filename for RAG search."""
    # flux_china_architecture_house_traditional.png -> architecture
    parts = filename.replace('.png', '').split('_')
    if len(parts) >= 3:
        return parts[2]  # e.g., 'architecture', 'food', 'fashion'
    return 'general'


def run_experiment(
    country_filter: str = "all",
    i2i_model: str = "qwen",
    max_images: Optional[int] = None,
    skip_existing: bool = True
):
    """
    Run the OURS experiment.

    Args:
        country_filter: 'china', 'korea', or 'all'
        i2i_model: I2I model to use ('qwen', 'flux', 'sd35')
        max_images: Maximum images to process (for testing)
        skip_existing: Skip already processed images
    """
    logger.info("=" * 60)
    logger.info("OURS EXPERIMENT - Cultural Bias Mitigation Pipeline")
    logger.info("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts_from_csv()

    # Add Korea prompts (not in CSV)
    if country_filter in ['korea', 'all']:
        korea_prompts = generate_korea_prompts()
        prompts.extend(korea_prompts)
        logger.info(f"Added {len(korea_prompts)} Korea prompts")

    # Filter by country
    if country_filter != 'all':
        prompts = [p for p in prompts if country_filter in p['base_path'].lower()]

    logger.info(f"Total images to process: {len(prompts)}")

    if max_images:
        prompts = prompts[:max_images]
        logger.info(f"Limited to {max_images} images for testing")

    # Initialize components
    logger.info("Initializing pipeline components...")

    from ccub2_agent.modules.vlm_detector import VLMCulturalDetector
    from ccub2_agent.adapters.image_editing_adapter import create_adapter
    from ccub2_agent.modules.prompt_adapter import UniversalPromptAdapter, EditingContext
    from ccub2_agent.modules.clip_image_rag import CLIPImageRAG

    # Initialize VLM Detector
    logger.info("Loading VLM Detector...")
    vlm_detector = VLMCulturalDetector(load_in_4bit=True)

    # Initialize I2I Adapter
    logger.info(f"Loading I2I Adapter: {i2i_model}")
    i2i_adapter = create_adapter(model_type=i2i_model, t2i_model="sd35")

    # Initialize Prompt Adapter
    prompt_adapter = UniversalPromptAdapter()

    # Results tracking
    results = []

    # Process each image
    for idx, item in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{len(prompts)}] Processing: {item['base_path']}")
        logger.info(f"{'='*60}")

        try:
            # Resolve paths - handle case sensitivity
            base_path_str = item['base_path']
            # Try different case combinations
            possible_paths = [
                BASE_DIR / base_path_str,
                BASE_DIR / base_path_str.replace('china/', 'China/'),
                BASE_DIR / base_path_str.replace('korea/', 'Korea/'),
            ]

            base_path = None
            for p in possible_paths:
                if p.exists():
                    base_path = p
                    break

            if not base_path:
                logger.error(f"Base image not found: {item['base_path']}")
                continue

            output_path = OUTPUT_DIR / Path(item['output_path']).name

            # Skip if exists
            if skip_existing and output_path.exists():
                logger.info(f"Skipping (already exists): {output_path.name}")
                continue

            # Extract metadata
            country = extract_country_from_path(item['base_path'])
            category = extract_category_from_filename(base_path.name)
            base_prompt = item['prompt']

            logger.info(f"Country: {country}, Category: {category}")
            logger.info(f"Base prompt: {base_prompt}")

            # Load image
            image = Image.open(base_path).convert('RGB')

            # Step 1: VLM Detection
            logger.info("\n[Step 1] VLM Cultural Detection...")
            issues = vlm_detector.detect(
                image_path=base_path,
                prompt=base_prompt,
                country=country,
                category=category
            )

            # Calculate initial score based on issues
            if issues:
                avg_severity = sum(issue.get('severity', 5) for issue in issues) / len(issues)
                initial_score = max(1, 10 - avg_severity)
            else:
                initial_score = 9.0

            logger.info(f"Initial Cultural Score: {initial_score:.1f}/10")
            logger.info(f"Issues detected: {len(issues)}")
            for issue in issues[:3]:
                logger.info(f"  - {issue.get('description', str(issue))}")

            # Step 2: Generate Adapted Prompt
            logger.info("\n[Step 2] Prompt Adaptation...")

            # Build editing context from VLM issues
            context = EditingContext(
                country=country,
                category=category,
                cultural_issues=[issue.get('description', str(issue)) for issue in issues[:5]],
                target_elements=[],  # Will be extracted by adapter
                preserve_elements=["composition", "pose", "background", "lighting"]
            )

            adapted_prompt = prompt_adapter.adapt(
                universal_instruction=base_prompt,
                model_type=i2i_model,
                context=context
            )
            logger.info(f"Adapted prompt: {adapted_prompt[:200]}...")

            # Step 3: I2I Editing (1 iteration only)
            logger.info(f"\n[Step 3] I2I Editing with {i2i_model}...")

            # Initialize CLIP RAG for this country
            clip_rag = None
            reference_image = None
            reference_metadata = None

            try:
                clip_index_path = PROJECT_ROOT / "data" / "clip_index" / country
                if clip_index_path.exists():
                    clip_rag = CLIPImageRAG(index_path=str(clip_index_path))

                    # Search for reference image
                    search_results = clip_rag.search(
                        query_image=image,
                        category=category,
                        top_k=3
                    )

                    if search_results:
                        # Use top result as reference
                        top_result = search_results[0]
                        ref_path = top_result.get('path')
                        if ref_path and Path(ref_path).exists():
                            reference_image = Image.open(ref_path).convert('RGB')
                            reference_metadata = top_result.get('metadata', {})
                            logger.info(f"Using reference: {Path(ref_path).name} (similarity: {top_result.get('score', 0):.2%})")
            except Exception as e:
                logger.warning(f"CLIP RAG not available: {e}")

            # Run I2I editing
            edited_image = i2i_adapter.edit(
                image=image,
                instruction=adapted_prompt,
                reference_image=reference_image,
                reference_metadata=reference_metadata,
                strength=0.35,
                num_inference_steps=40,
                seed=42
            )

            # Save result
            edited_image.save(output_path)
            logger.info(f"Saved: {output_path}")

            # Step 4: Re-evaluate with VLM
            logger.info("\n[Step 4] Final VLM Evaluation...")
            final_result = vlm_detector.analyze(
                image=edited_image,
                prompt=base_prompt,
                country=country
            )

            final_score = final_result.get('cultural_score', 5.0)
            final_issues = final_result.get('issues', [])

            improvement = final_score - initial_score
            logger.info(f"Final Cultural Score: {final_score:.1f}/10 ({'+' if improvement >= 0 else ''}{improvement:.1f})")
            logger.info(f"Remaining issues: {len(final_issues)}")

            # Record result
            results.append({
                'image': base_path.name,
                'country': country,
                'category': category,
                'initial_score': initial_score,
                'final_score': final_score,
                'improvement': improvement,
                'initial_issues': len(issues),
                'final_issues': len(final_issues),
                'output_path': str(output_path)
            })

            # Clear GPU memory
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {item['base_path']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)

    if results:
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        avg_initial = sum(r['initial_score'] for r in results) / len(results)
        avg_final = sum(r['final_score'] for r in results) / len(results)

        logger.info(f"Processed: {len(results)} images")
        logger.info(f"Average Initial Score: {avg_initial:.2f}/10")
        logger.info(f"Average Final Score: {avg_final:.2f}/10")
        logger.info(f"Average Improvement: {'+' if avg_improvement >= 0 else ''}{avg_improvement:.2f}")

        # Save detailed results
        results_path = OUTPUT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'experiment': 'ours',
                'i2i_model': i2i_model,
                'country_filter': country_filter,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_images': len(results),
                    'avg_initial_score': avg_initial,
                    'avg_final_score': avg_final,
                    'avg_improvement': avg_improvement
                },
                'results': results
            }, f, indent=2)
        logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run OURS experiment")
    parser.add_argument('--country', type=str, default='all',
                        choices=['china', 'korea', 'all'],
                        help='Country to process')
    parser.add_argument('--i2i-model', type=str, default='qwen',
                        choices=['qwen', 'flux', 'sd35'],
                        help='I2I model to use')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum images to process (for testing)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Process all images even if output exists')

    args = parser.parse_args()

    run_experiment(
        country_filter=args.country,
        i2i_model=args.i2i_model,
        max_images=args.max_images,
        skip_existing=not args.no_skip
    )


if __name__ == "__main__":
    main()
