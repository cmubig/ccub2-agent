#!/usr/bin/env python3
"""
Stable cultural knowledge extraction for all countries.
Runs sequentially with proper GPU cleanup between countries.
Includes resume capability and detailed logging.
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / f"stable_extraction_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

COUNTRIES = [
    'korea', 'china', 'japan', 'usa', 'nigeria',
    'mexico', 'kenya', 'italy', 'france', 'germany'
]


def get_extraction_status(country: str) -> dict:
    """Get current extraction status for a country."""
    knowledge_path = PROJECT_ROOT / "data" / "cultural_knowledge" / f"{country}_knowledge.json"
    dataset_path = PROJECT_ROOT / "data" / "country_packs" / country / "approved_dataset.json"

    status = {
        'country': country,
        'dataset_exists': dataset_path.exists(),
        'knowledge_exists': knowledge_path.exists(),
        'total_items': 0,
        'extracted_count': 0,
        'extraction_rate': 0.0
    }

    if dataset_path.exists():
        with open(dataset_path) as f:
            data = json.load(f)
            status['total_items'] = len(data.get('items', []))

    if knowledge_path.exists():
        with open(knowledge_path) as f:
            data = json.load(f)
            status['extracted_count'] = data.get('extracted_count', 0)

    if status['total_items'] > 0:
        status['extraction_rate'] = status['extracted_count'] / status['total_items']

    return status


def extract_country(country: str, gpu_id: int = 0, resume: bool = True) -> bool:
    """
    Extract cultural knowledge for one country.

    Args:
        country: Country name
        gpu_id: GPU ID to use
        resume: Whether to resume from existing progress

    Returns:
        True if successful, False otherwise
    """
    country_pack_dir = PROJECT_ROOT / "data" / "country_packs" / country
    output_path = PROJECT_ROOT / "data" / "cultural_knowledge" / f"{country}_knowledge.json"

    # Check dataset exists
    dataset_path = country_pack_dir / "approved_dataset.json"
    if not dataset_path.exists():
        logger.warning(f"Skipping {country}: dataset not found at {dataset_path}")
        return False

    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "02_data_processing" / "extract_cultural_knowledge.py"),
        "--data-dir", str(country_pack_dir),
        "--output", str(output_path),
        "--load-in-4bit",  # Use 4-bit quantization to save GPU memory
    ]

    if resume:
        cmd.append("--resume")

    # Set environment
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    logger.info(f"{'='*80}")
    logger.info(f"Starting extraction: {country.upper()} on GPU {gpu_id}")
    logger.info(f"{'='*80}")

    try:
        # Run extraction
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=7200  # 2 hour timeout per country
        )

        # Log output
        logger.info(f"Output for {country}:")
        for line in proc.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")

        success = proc.returncode == 0

        if success:
            logger.info(f"✓ {country.upper()} completed successfully")
        else:
            logger.error(f"✗ {country.upper()} failed with return code {proc.returncode}")

        # Clear GPU cache
        logger.info(f"Clearing GPU {gpu_id} cache...")
        clear_cmd = [
            sys.executable, "-c",
            "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
        ]
        subprocess.run(clear_cmd, env=env)

        # Wait before next country
        time.sleep(5)

        return success

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {country.upper()} timed out after 2 hours")
        return False
    except Exception as e:
        logger.error(f"✗ {country.upper()} failed with exception: {e}")
        return False


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("STABLE CULTURAL KNOWLEDGE EXTRACTION")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*80)

    # Get initial status
    logger.info("\nInitial status:")
    logger.info("-"*80)
    statuses = []
    for country in COUNTRIES:
        status = get_extraction_status(country)
        statuses.append(status)
        logger.info(
            f"{country:10s}: {status['extracted_count']:3d}/{status['total_items']:3d} "
            f"({status['extraction_rate']*100:5.1f}%)"
        )

    logger.info("-"*80)

    # Ask which countries to process
    countries_to_process = []
    for status in statuses:
        if not status['dataset_exists']:
            continue
        if status['extraction_rate'] < 0.9:  # Less than 90% complete
            countries_to_process.append(status['country'])

    if not countries_to_process:
        logger.info("\n✓ All countries are already complete (>90% extraction rate)")
        return

    logger.info(f"\nCountries to process: {', '.join(countries_to_process)}")
    logger.info(f"Total: {len(countries_to_process)} countries")
    logger.info("="*80)

    # Process each country sequentially
    results = {}

    for i, country in enumerate(countries_to_process, 1):
        logger.info(f"\n[{i}/{len(countries_to_process)}] Processing {country.upper()}...")

        success = extract_country(country, gpu_id=0, resume=True)
        results[country] = success

        # Show progress
        status = get_extraction_status(country)
        logger.info(
            f"Result: {status['extracted_count']}/{status['total_items']} "
            f"({status['extraction_rate']*100:.1f}%)"
        )

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)

    for country in COUNTRIES:
        status = get_extraction_status(country)
        success_marker = "✓" if status['extraction_rate'] > 0.8 else "✗"
        logger.info(
            f"{success_marker} {country:10s}: {status['extracted_count']:3d}/{status['total_items']:3d} "
            f"({status['extraction_rate']*100:5.1f}%)"
        )

    logger.info("="*80)

    # Count successes
    successful = sum(1 for s in statuses if get_extraction_status(s['country'])['extraction_rate'] > 0.8)
    logger.info(f"\n✓ Successfully completed: {successful}/{len(COUNTRIES)} countries")
    logger.info(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
