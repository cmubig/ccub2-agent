#!/usr/bin/env python3
"""
Download test images from Firebase Storage for E2E testing.

Usage:
    python scripts/download_test_images.py --country korea --limit 5
"""

import argparse
import csv
import logging
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_test_images(
    country: str = "korea",
    category: str = None,
    limit: int = 5,
    output_dir: Path = None
):
    """Download approved contribution images for testing."""

    contributions_file = PROJECT_ROOT / "data" / "_contributions.csv"
    if not contributions_file.exists():
        raise FileNotFoundError(f"Contributions file not found: {contributions_file}")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "test_images" / country
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read contributions
    downloaded = []
    with open(contributions_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(downloaded) >= limit:
                break

            # Filter by status (approved preferred)
            status = row.get('reviewStatus', '')
            if status not in ['approved', 'pending']:
                continue

            # Filter by category if specified
            if category and row.get('category', '').lower() != category.lower():
                continue

            image_url = row.get('imageURL', '')
            if not image_url:
                continue

            # Extract filename from URL or use ID
            file_id = row.get('__id__', f'image_{len(downloaded)}')
            cat = row.get('category', 'unknown').lower().replace(' ', '_')
            filename = f"{cat}_{file_id}.jpg"
            output_path = output_dir / filename

            if output_path.exists():
                logger.info(f"Already exists: {filename}")
                downloaded.append({
                    "path": str(output_path),
                    "category": cat,
                    "description": row.get('description', '')
                })
                continue

            # Download image
            try:
                logger.info(f"Downloading: {filename}")
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()

                with open(output_path, 'wb') as img_file:
                    img_file.write(response.content)

                downloaded.append({
                    "path": str(output_path),
                    "category": cat,
                    "description": row.get('description', '')
                })
                logger.info(f"  -> Saved to {output_path}")

            except Exception as e:
                logger.error(f"  -> Failed: {e}")

    logger.info(f"\nDownloaded {len(downloaded)} images to {output_dir}")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download test images")
    parser.add_argument("--country", "-c", default="korea")
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", "-n", type=int, default=5)
    parser.add_argument("--output-dir", "-o", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    downloaded = download_test_images(
        country=args.country,
        category=args.category,
        limit=args.limit,
        output_dir=output_dir
    )

    if downloaded:
        print("\nTest images ready:")
        for img in downloaded:
            print(f"  {img['path']}")
        print(f"\nRun E2E test with:")
        print(f"  python scripts/test_e2e_loop.py --image {downloaded[0]['path']} --country {args.country}")


if __name__ == "__main__":
    main()
