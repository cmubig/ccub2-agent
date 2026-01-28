#!/usr/bin/env python3
"""
Batch initialize multiple countries without user interaction.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def init_country(country: str, data_dir: Path, skip_images: bool = False):
    """Initialize a single country dataset."""

    logger.info(f"=" * 80)
    logger.info(f"Initializing {country.upper()}")
    logger.info(f"=" * 80)

    country_pack_dir = data_dir / "country_packs" / country
    country_pack_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert contributions to dataset (using Firebase)
    logger.info(f"▶ Step 1: Creating approved_dataset.json for {country}...")
    try:
        from init_dataset import convert_contributions_to_dataset
        new_items = convert_contributions_to_dataset(
            PROJECT_ROOT / "data" / "_contributions.csv",
            country_pack_dir / "approved_dataset.json",
            country,
            use_firebase=True
        )
        logger.info(f"✓ Added {new_items} new items to dataset")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return False

    if new_items == 0:
        logger.info(f"No new data for {country}, checking if dataset exists...")
        if not (country_pack_dir / "approved_dataset.json").exists():
            logger.warning(f"No dataset found for {country}, skipping")
            return False

    # Step 2: Download images
    if not skip_images:
        logger.info(f"▶ Step 2: Downloading images for {country}...")
        try:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "05_utils" / "download_country_images.py"),
                "--country", country,
                "--output-dir", str(country_pack_dir / "images")
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"✓ Images downloaded")
        except Exception as e:
            logger.warning(f"Image download failed: {e}")

    # Step 3: Enhance captions
    logger.info(f"▶ Step 3: Enhancing captions for {country}...")
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data_processing" / "enhance_captions.py"),
            "--dataset", str(country_pack_dir / "approved_dataset.json"),
            "--images-dir", str(country_pack_dir / "images"),
            "--output", str(country_pack_dir / "approved_dataset_enhanced.json"),
            "--country", country
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"✓ Captions enhanced")
    except Exception as e:
        logger.error(f"Caption enhancement failed: {e}")
        return False

    # Step 4: Extract cultural knowledge
    logger.info(f"▶ Step 4: Extracting cultural knowledge for {country}...")
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "data_processing" / "extract_cultural_knowledge.py"),
            "--country", country,
            "--dataset", str(country_pack_dir / "approved_dataset_enhanced.json"),
            "--output", str(data_dir / "cultural_knowledge" / f"{country}_knowledge.json")
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"✓ Cultural knowledge extracted")
    except Exception as e:
        logger.error(f"Knowledge extraction failed: {e}")
        return False

    # Step 5: Build CLIP index
    logger.info(f"▶ Step 5: Building CLIP image index for {country}...")
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "indexing" / "build_clip_image_index.py"),
            "--country", country,
            "--images-dir", str(country_pack_dir / "images"),
            "--dataset", str(country_pack_dir / "approved_dataset_enhanced.json"),
            "--output-dir", str(data_dir / "clip_index" / country)
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"✓ CLIP index built")
    except Exception as e:
        logger.error(f"CLIP index building failed: {e}")
        return False

    # Step 6: Integrate knowledge to RAG
    logger.info(f"▶ Step 6: Integrating knowledge to RAG for {country}...")
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "indexing" / "integrate_knowledge_to_rag.py"),
            "--country", country
        ]
        subprocess.run(cmd, check=True)
        logger.info(f"✓ Knowledge integrated to RAG")
    except Exception as e:
        logger.error(f"RAG integration failed: {e}")
        return False

    logger.info(f"✅ {country.upper()} initialization completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch initialize multiple countries")
    parser.add_argument(
        '--countries',
        type=str,
        nargs='+',
        required=True,
        help='List of countries to initialize (e.g., china japan usa)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=PROJECT_ROOT / "data",
        help='Data directory'
    )
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='Skip image download'
    )

    args = parser.parse_args()

    logger.info(f"Batch initializing {len(args.countries)} countries:")
    for country in args.countries:
        logger.info(f"  - {country}")

    success_count = 0
    failed_countries = []

    for country in args.countries:
        success = init_country(country, args.data_dir, args.skip_images)
        if success:
            success_count += 1
        else:
            failed_countries.append(country)
        print()

    logger.info("=" * 80)
    logger.info(f"Batch initialization completed:")
    logger.info(f"  ✅ Success: {success_count}/{len(args.countries)}")
    if failed_countries:
        logger.info(f"  ❌ Failed: {', '.join(failed_countries)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
