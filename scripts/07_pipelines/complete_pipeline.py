#!/usr/bin/env python3
"""
Complete the remaining pipeline steps for all countries:
1. Extract cultural knowledge
2. Build CLIP image index
3. Integrate knowledge to RAG
"""

import subprocess
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_cultural_knowledge(country: str) -> bool:
    """Extract cultural knowledge for a country."""
    logger.info(f"[{country.upper()}] Step 1: Extracting cultural knowledge...")

    country_pack_dir = PROJECT_ROOT / "data" / "country_packs" / country
    dataset_path = country_pack_dir / "approved_dataset_enhanced.json"
    output_path = PROJECT_ROOT / "data" / "cultural_knowledge" / f"{country}_knowledge.json"

    if not dataset_path.exists():
        logger.warning(f"Enhanced dataset not found: {dataset_path}")
        return False

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "02_data_processing" / "extract_cultural_knowledge.py"),
        "--data-dir", str(country_pack_dir),
        "--output", str(output_path),
        "--resume"  # Resume from existing progress
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ {country}: Cultural knowledge extracted")
        return True
    except Exception as e:
        logger.error(f"✗ {country}: Cultural knowledge extraction failed - {e}")
        return False


def build_clip_index(country: str) -> bool:
    """Build CLIP image index for a country."""
    logger.info(f"[{country.upper()}] Step 2: Building CLIP image index...")

    country_pack_dir = PROJECT_ROOT / "data" / "country_packs" / country
    dataset_path = country_pack_dir / "approved_dataset_enhanced.json"
    images_dir = country_pack_dir / "images"
    output_dir = PROJECT_ROOT / "data" / "clip_index" / country

    if not dataset_path.exists():
        logger.warning(f"Enhanced dataset not found: {dataset_path}")
        return False

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return False

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "03_indexing" / "build_clip_image_index.py"),
        "--country", country,
        "--images-dir", str(images_dir),
        "--dataset", str(dataset_path),
        "--output-dir", str(output_dir)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ {country}: CLIP index built")
        return True
    except Exception as e:
        logger.error(f"✗ {country}: CLIP index building failed - {e}")
        return False


def integrate_to_rag(country: str) -> bool:
    """Integrate knowledge to RAG for a country."""
    logger.info(f"[{country.upper()}] Step 3: Integrating knowledge to RAG...")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "03_indexing" / "integrate_knowledge_to_rag.py"),
        "--country", country
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ {country}: Knowledge integrated to RAG")
        return True
    except Exception as e:
        logger.error(f"✗ {country}: RAG integration failed - {e}")
        return False


def process_country(country: str) -> bool:
    """Process all remaining steps for a country."""
    logger.info("=" * 80)
    logger.info(f"Processing {country.upper()}")
    logger.info("=" * 80)

    # Step 1: Extract cultural knowledge
    if not extract_cultural_knowledge(country):
        return False

    # Step 2: Build CLIP index
    if not build_clip_index(country):
        return False

    # Step 3: Integrate to RAG
    if not integrate_to_rag(country):
        return False

    logger.info(f"✅ {country.upper()} pipeline completed successfully!")
    return True


def main():
    countries = ['korea', 'china', 'japan', 'usa', 'nigeria', 'general', 'mexico', 'kenya', 'italy', 'france', 'germany']

    logger.info("=" * 80)
    logger.info("COMPLETING PIPELINE FOR ALL COUNTRIES")
    logger.info("=" * 80)
    logger.info(f"Countries to process: {len(countries)}")
    logger.info("")

    results = []
    for i, country in enumerate(countries, 1):
        logger.info(f"\n[{i}/{len(countries)}] Starting {country}...")
        success = process_country(country)
        results.append((country, success))
        print()

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETION SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for _, success in results if success)
    logger.info(f"✅ Success: {success_count}/{len(countries)}")

    failed = [country for country, success in results if not success]
    if failed:
        logger.warning(f"❌ Failed: {', '.join(failed)}")

    logger.info("=" * 80)

    return success_count == len(countries)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
