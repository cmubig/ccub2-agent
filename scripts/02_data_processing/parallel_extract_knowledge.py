#!/usr/bin/env python3
"""
Extract cultural knowledge for multiple countries in parallel using both GPUs.
Runs 2 countries at a time (one per GPU).
"""
import subprocess
import sys
import os
from pathlib import Path
import logging
import time

PROJECT_ROOT = Path(__file__).parent.parent
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

countries = ['korea', 'china', 'japan', 'usa', 'nigeria', 'general', 'mexico', 'kenya', 'italy', 'france', 'germany']

def run_country(country: str, gpu_id: int):
    """Run extraction for one country on specified GPU."""
    country_pack_dir = PROJECT_ROOT / "data" / "country_packs" / country
    output_path = PROJECT_ROOT / "data" / "cultural_knowledge" / f"{country}_knowledge.json"

    dataset_path = country_pack_dir / "approved_dataset_enhanced.json"
    if not dataset_path.exists():
        logger.warning(f"Skipping {country}: enhanced dataset not found")
        return None

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "02_data_processing" / "extract_cultural_knowledge.py"),
        "--data-dir", str(country_pack_dir),
        "--output", str(output_path),
        "--resume"
    ]

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    logger.info(f"Starting {country.upper()} on GPU {gpu_id}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return proc

# Process countries 2 at a time
i = 0
while i < len(countries):
    # Start up to 2 countries in parallel
    procs = []
    for j in range(2):
        if i + j < len(countries):
            country = countries[i + j]
            gpu_id = j  # GPU 0 for first, GPU 1 for second
            proc = run_country(country, gpu_id)
            if proc:
                procs.append((country, proc))

    # Wait for both to complete
    for country, proc in procs:
        proc.wait()
        if proc.returncode == 0:
            logger.info(f"✓ {country.upper()} completed")
        else:
            logger.error(f"✗ {country.upper()} failed")

    i += 2

logger.info("=" * 80)
logger.info("All countries processed!")
