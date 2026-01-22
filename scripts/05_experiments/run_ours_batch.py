#!/usr/bin/env python3
"""
Simple Batch I2I Editing Script for OURS Experiment

Reads CSV prompts and applies Qwen-Image-Edit-2509 to each base image.
No VLM detection - just direct I2I editing with the provided prompts.

Usage:
    python run_ours_batch.py
"""

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / f'ours_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = PROJECT_ROOT / "base_experimental"
OUTPUT_DIR = BASE_DIR / "ours"
CHINA_CSV = BASE_DIR / "edit-prompt.csv"
KOREA_CSV = BASE_DIR / "edit-prompt-kor.csv"


def load_all_prompts():
    """Load prompts from both CSV files."""
    prompts = []

    # Load China prompts
    if CHINA_CSV.exists():
        with open(CHINA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({
                    'prompt': row['I2I_prompt'],
                    'base_path': row['base_path'],
                    'output_name': Path(row['base_path']).name,
                    'country': 'china'
                })
        logger.info(f"Loaded {len(prompts)} China prompts")

    # Load Korea prompts
    korea_count = 0
    if KOREA_CSV.exists():
        with open(KOREA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({
                    'prompt': row['I2I_prompt'],
                    'base_path': row['base_path'],
                    'output_name': Path(row['base_path']).name,
                    'country': 'korea'
                })
                korea_count += 1
        logger.info(f"Loaded {korea_count} Korea prompts")

    return prompts


def resolve_image_path(base_path_str: str) -> Path:
    """Resolve image path handling case sensitivity."""
    # Try different case combinations
    possible_paths = [
        BASE_DIR / base_path_str,
        BASE_DIR / base_path_str.replace('china/', 'China/'),
        BASE_DIR / base_path_str.replace('korea/', 'Korea/'),
        BASE_DIR / base_path_str.replace('China/', 'china/'),
        BASE_DIR / base_path_str.replace('Korea/', 'korea/'),
    ]

    for p in possible_paths:
        if p.exists():
            return p
    return None


def main():
    logger.info("=" * 70)
    logger.info("OURS BATCH EXPERIMENT - Qwen-Image-Edit-2509")
    logger.info("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

    # Load all prompts
    prompts = load_all_prompts()
    logger.info(f"Total images to process: {len(prompts)}")

    # Initialize I2I adapter
    logger.info("\nLoading Qwen-Image-Edit-2509...")
    from ccub2_agent.adapters.image_editing_adapter import create_adapter

    i2i_adapter = create_adapter(model_type='qwen', t2i_model='sd35')
    logger.info("Model loaded successfully!\n")

    # Process each image
    success_count = 0
    skip_count = 0
    error_count = 0

    start_time = time.time()

    for idx, item in enumerate(prompts):
        output_path = OUTPUT_DIR / item['output_name']

        # Skip if already exists
        if output_path.exists():
            logger.info(f"[{idx+1}/{len(prompts)}] SKIP (exists): {item['output_name']}")
            skip_count += 1
            continue

        logger.info(f"\n[{idx+1}/{len(prompts)}] Processing: {item['output_name']}")
        logger.info(f"  Prompt: {item['prompt']}")

        try:
            # Resolve input path
            input_path = resolve_image_path(item['base_path'])
            if not input_path:
                logger.error(f"  ERROR: Image not found: {item['base_path']}")
                error_count += 1
                continue

            # Load image
            image = Image.open(input_path).convert('RGB')

            # Apply I2I editing
            edited_image = i2i_adapter.edit(
                image=image,
                instruction=item['prompt'],
                reference_image=None,
                reference_metadata=None,
                strength=0.35,
                num_inference_steps=40,
                seed=42
            )

            # Save result
            edited_image.save(output_path)
            logger.info(f"  SAVED: {output_path.name}")
            success_count += 1

            # Clear GPU memory periodically
            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            error_count += 1
            continue

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total: {len(prompts)} | Success: {success_count} | Skipped: {skip_count} | Errors: {error_count}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
