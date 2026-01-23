#!/usr/bin/env python3
"""
Batch download images for multiple countries in parallel.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_country_images(country: str) -> tuple[str, bool, str]:
    """Download images for a single country."""
    logger.info(f"Starting download for {country}...")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "05_utils" / "download_country_images.py"),
        "--country", country
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per country
        )

        if result.returncode == 0:
            logger.info(f"✓ {country} download completed")
            return (country, True, result.stdout)
        else:
            logger.error(f"✗ {country} download failed")
            return (country, False, result.stderr)

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {country} download timed out")
        return (country, False, "Timeout")
    except Exception as e:
        logger.error(f"✗ {country} download error: {e}")
        return (country, False, str(e))


def main():
    countries = ['china', 'japan', 'usa', 'nigeria', 'general', 'mexico', 'kenya', 'italy', 'france', 'germany']

    logger.info(f"Starting batch download for {len(countries)} countries")
    logger.info("=" * 80)

    # Use ThreadPoolExecutor for parallel downloads (max 5 at a time)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_country_images, country): country for country in countries}

        results = []
        for future in as_completed(futures):
            country, success, output = future.result()
            results.append((country, success))

            # Log summary
            if success:
                # Count downloaded images from output
                lines = output.split('\n')
                for line in lines:
                    if 'Downloaded' in line or 'images' in line.lower():
                        logger.info(f"  {country}: {line.strip()}")

    logger.info("=" * 80)
    logger.info("Download summary:")
    success_count = sum(1 for _, success in results if success)
    logger.info(f"  ✓ Success: {success_count}/{len(countries)}")

    failed = [country for country, success in results if not success]
    if failed:
        logger.warning(f"  ✗ Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
