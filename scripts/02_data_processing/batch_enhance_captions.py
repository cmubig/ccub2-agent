#!/usr/bin/env python3
"""
Batch enhance captions for multiple countries sequentially.
"""

import subprocess
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def enhance_captions_for_country(country: str) -> bool:
    """Enhance captions for a single country."""
    logger.info("=" * 80)
    logger.info(f"Enhancing captions for {country.upper()}")
    logger.info("=" * 80)

    country_pack_dir = PROJECT_ROOT / "data" / "country_packs" / country
    dataset_path = country_pack_dir / "approved_dataset.json"
    images_dir = country_pack_dir / "images"
    output_path = country_pack_dir / "approved_dataset_enhanced.json"

    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}")
        return False

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return False

    # Check if already enhanced
    if output_path.exists():
        logger.info(f"Enhanced dataset already exists, will update with new items")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "02_data_processing" / "enhance_captions.py"),
        "--dataset", str(dataset_path),
        "--images-dir", str(images_dir),
        "--output", str(output_path),
        "--country", country
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        logger.info(f"✓ {country} captions enhanced successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {country} caption enhancement failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {country} caption enhancement error: {e}")
        return False


def main():
    countries = ['china', 'japan', 'usa', 'nigeria', 'general', 'mexico', 'kenya', 'italy', 'france', 'germany']

    logger.info(f"Starting batch caption enhancement for {len(countries)} countries")
    logger.info("This will run SEQUENTIALLY to avoid GPU memory issues")
    logger.info("=" * 80)
    print()

    results = []
    for i, country in enumerate(countries, 1):
        logger.info(f"\n[{i}/{len(countries)}] Processing {country}...")
        success = enhance_captions_for_country(country)
        results.append((country, success))
        print()

    logger.info("=" * 80)
    logger.info("Caption enhancement summary:")
    success_count = sum(1 for _, success in results if success)
    logger.info(f"  ✓ Success: {success_count}/{len(countries)}")

    failed = [country for country, success in results if not success]
    if failed:
        logger.warning(f"  ✗ Failed: {', '.join(failed)}")

    logger.info("=" * 80)

    return success_count == len(countries)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
