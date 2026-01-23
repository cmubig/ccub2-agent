#!/usr/bin/env python3
"""
Download images from Firebase Storage (incremental update)
Usage: python scripts/download_images.py --country korea [--max-images 100]
"""

import json
import requests
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm
import time


def load_dataset(country_dir: Path) -> Dict:
    """Load approved_dataset.json"""
    dataset_path = country_dir / "approved_dataset.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_dataset(country_dir: Path, dataset: Dict):
    """Save updated approved_dataset.json"""
    dataset_path = country_dir / "approved_dataset.json"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def download_image(url: str, save_path: Path, timeout: int = 10) -> bool:
    """
    Download image from URL.

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        # Save image
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def get_image_extension(url: str) -> str:
    """Extract file extension from URL"""
    # Try to get from URL path
    if '.jpg' in url.lower() or '.jpeg' in url.lower():
        return '.jpg'
    elif '.png' in url.lower():
        return '.png'
    elif '.gif' in url.lower():
        return '.gif'
    elif '.webp' in url.lower():
        return '.webp'
    else:
        return '.jpg'  # Default


def download_country_images(
    country: str,
    data_dir: Path,
    max_images: int = None,
    skip_existing: bool = True
):
    """
    Download images for a country pack.

    Args:
        country: Country name
        data_dir: Data directory path
        max_images: Maximum number of images to download (None = all)
        skip_existing: Skip already downloaded images
    """
    country_dir = data_dir / "country_packs" / country
    images_dir = country_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading images for: {country.upper()}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_dataset(country_dir)
    items = dataset.get('items', [])

    print(f"Total items: {len(items)}")

    # Filter items that need download
    to_download = []
    for item in items:
        # Skip if no image URL
        if not item.get('image_url'):
            continue

        # Check if already downloaded
        if skip_existing and item.get('image_path'):
            image_path = country_dir / item['image_path']
            if image_path.exists():
                continue

        to_download.append(item)

    print(f"To download: {len(to_download)}")

    if max_images:
        to_download = to_download[:max_images]
        print(f"Limited to: {max_images}")

    if not to_download:
        print("\n✓ All images already downloaded!")
        return

    # Download images
    print(f"\nDownloading...\n")

    success_count = 0
    failed_count = 0

    for item in tqdm(to_download, desc="Progress"):
        image_url = item['image_url']
        contrib_id = item.get('contribution_id', item['id'])
        category = item['category']

        # Create category directory
        category_dir = images_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        ext = get_image_extension(image_url)
        filename = f"{contrib_id}{ext}"
        save_path = category_dir / filename

        # Relative path for JSON
        relative_path = f"images/{category}/{filename}"

        # Download
        if download_image(image_url, save_path):
            # Update item with local path
            item['image_path'] = relative_path
            success_count += 1
        else:
            failed_count += 1

        # Rate limiting
        time.sleep(0.1)

    # Save updated dataset
    save_dataset(country_dir, dataset)

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"{'='*60}")
    print(f"Success: {success_count}")
    print(f"Failed:  {failed_count}")
    print(f"Total:   {len(to_download)}")
    print(f"\nImages saved to: {images_dir}")
    print(f"Dataset updated: {country_dir / 'approved_dataset.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Download images from Firebase Storage'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., korea, japan, china)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to download (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Data directory path (default: auto-detect)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download existing images'
    )

    args = parser.parse_args()

    # Auto-detect data directory
    if args.data_dir is None:
        # Check if running on server or local
        server_data_path = Path.home() / "ccub2-agent" / "data"
        local_path = Path(__file__).parent.parent / "data"

        if server_data_path.exists():
            args.data_dir = server_data_path
        else:
            args.data_dir = local_path

    print(f"Data directory: {args.data_dir}")

    download_country_images(
        country=args.country,
        data_dir=args.data_dir,
        max_images=args.max_images,
        skip_existing=not args.force
    )


if __name__ == "__main__":
    main()
