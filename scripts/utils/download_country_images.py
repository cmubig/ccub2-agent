#!/usr/bin/env python3
"""
Download all Country Pack images from Firebase Storage.
"""

import argparse
import json
import requests
from pathlib import Path
from tqdm import tqdm
import time

def download_image(url: str, save_path: Path, timeout: int = 15) -> bool:
    """Download image from URL."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"  ✗ Failed {save_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download country pack images from Firebase")
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., korea, japan, china)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for images (default: PROJECT_ROOT/data/country_packs/{country}/images)'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        help='Dataset JSON path (default: PROJECT_ROOT/data/country_packs/{country}/approved_dataset.json)'
    )
    args = parser.parse_args()

    # Set default paths
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    base_dir = PROJECT_ROOT / f"data/country_packs/{args.country}"
    dataset_path = args.dataset or (base_dir / "approved_dataset.json")
    images_dir = args.output_dir or (base_dir / "images")

    print("="*70)
    print(f"DOWNLOADING {args.country.upper()} IMAGES")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {images_dir}")
    print()

    # Load dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = data['items']
    print(f"Total items: {len(items)}")

    # Count by category
    categories = {}
    for item in items:
        cat = item.get('category', 'uncategorized')
        categories[cat] = categories.get(cat, 0) + 1

    print("Categories:", dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)))
    print()

    # Check existing images
    existing = set()
    if images_dir.exists():
        for f in images_dir.rglob("*.*"):
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.gif'}:
                existing.add(f.name)

    print(f"Already downloaded: {len(existing)} images")
    print()

    # Download missing images
    downloaded = 0
    failed = 0
    skipped = 0

    for item in tqdm(items, desc="Downloading"):
        image_url = item.get('image_url')
        category = item.get('category', 'uncategorized')
        contribution_id = item.get('contribution_id')

        if not image_url:
            continue

        # Generate image_path if not present
        image_path = item.get('image_path')
        if not image_path:
            # Extract extension from URL
            ext = '.jpg'
            if '.png' in image_url.lower():
                ext = '.png'
            elif '.webp' in image_url.lower():
                ext = '.webp'

            # Generate filename
            if contribution_id:
                filename = f"{contribution_id}{ext}"
            else:
                filename = f"{item.get('id', f'img_{downloaded}')}{ext}"

            image_path = f"images/{category}/{filename}"

        # Target path
        target_path = images_dir / Path(image_path).relative_to('images') if 'images/' in image_path else images_dir / image_path

        # Skip if exists
        if target_path.exists():
            skipped += 1
            continue

        # Download
        if download_image(image_url, target_path):
            downloaded += 1
        else:
            failed += 1

        # Rate limit
        time.sleep(0.05)

    print()
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total images: {len(existing) + downloaded}")
    print()


if __name__ == "__main__":
    main()
