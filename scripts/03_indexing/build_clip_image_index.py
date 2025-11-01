#!/usr/bin/env python3
"""
Build CLIP-based FAISS index from Country Pack images.

This script:
1. Loads images from Country Pack
2. Encodes them using CLIP
3. Builds FAISS index for similarity search
4. Saves index + metadata to disk

Usage:
    python scripts/build_clip_image_index.py --country korea --images-dir ~/ccub2-agent-data/country_packs/korea/images/
"""

import argparse
from pathlib import Path
import json
import logging
import sys
from typing import List, Dict, Any

import numpy as np
import faiss
from tqdm import tqdm

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.clip_image_rag import CLIPImageRAG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_images(images_dir: Path, dataset_path: Path = None) -> List[Dict[str, Any]]:
    """
    Find all images in the directory, organized by category.

    Args:
        images_dir: Root images directory with category subdirectories
        dataset_path: Optional path to approved_dataset.json with descriptions

    Returns:
        List of image metadata dicts with descriptions if available
    """
    image_list = []

    # Load dataset descriptions if available
    descriptions = {}
    if dataset_path and dataset_path.exists():
        logger.info(f"Loading descriptions from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data.get('items', [])
            for item in items:
                # Map by image_path or filename
                img_path = item.get('image_path', '')
                if img_path:
                    filename = Path(img_path).name
                    descriptions[filename] = {
                        'description': item.get('description', ''),
                        'description_enhanced': item.get('description_enhanced', ''),
                        'description_lang': item.get('description_lang', 'ko'),
                    }
        logger.info(f"Loaded descriptions for {len(descriptions)} images")

    # Image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    # Check if images_dir has subdirectories (categories)
    subdirs = [d for d in images_dir.iterdir() if d.is_dir()]

    if subdirs:
        # Organized by category
        for category_dir in subdirs:
            category = category_dir.name
            logger.info(f"Scanning category: {category}")

            for img_path in category_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    meta = {
                        'image_path': str(img_path),
                        'category': category,
                        'filename': img_path.name,
                    }

                    # Add description if available
                    desc_data = descriptions.get(img_path.name, {})
                    if desc_data:
                        meta.update(desc_data)

                    image_list.append(meta)
    else:
        # Flat directory
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                meta = {
                    'image_path': str(img_path),
                    'category': 'general',
                    'filename': img_path.name,
                }

                # Add description if available
                desc_data = descriptions.get(img_path.name, {})
                if desc_data:
                    meta.update(desc_data)

                image_list.append(meta)

    logger.info(f"Found {len(image_list)} images")
    return image_list


def build_clip_index(
    country: str,
    images_dir: Path,
    output_dir: Path,
    dataset_path: Path = None,
    model_name: str = "openai/clip-vit-base-patch32",
):
    """
    Build CLIP FAISS index from images.

    Args:
        country: Country name
        images_dir: Directory containing images
        output_dir: Output directory for index
        dataset_path: Optional path to approved_dataset.json with descriptions
        model_name: CLIP model name
    """
    logger.info("="*60)
    logger.info("Building CLIP Image Index")
    logger.info("="*60)
    logger.info(f"Country: {country}")
    logger.info(f"Images dir: {images_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Dataset: {dataset_path or 'None'}")
    logger.info(f"CLIP model: {model_name}")
    logger.info("")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images (with descriptions if available)
    logger.info("Scanning for images...")
    image_list = find_images(images_dir, dataset_path)

    if not image_list:
        logger.error("No images found!")
        return

    # Initialize CLIP (without index, we'll build it)
    logger.info("Initializing CLIP model...")
    clip_rag = CLIPImageRAG(model_name=model_name, index_dir=None)

    # Encode all images
    logger.info("Encoding images with CLIP...")
    embeddings = []
    valid_metadata = []

    for img_meta in tqdm(image_list, desc="Encoding"):
        try:
            img_path = Path(img_meta['image_path'])
            embedding = clip_rag.encode_image(img_path)

            embeddings.append(embedding)
            valid_metadata.append(img_meta)

        except Exception as e:
            logger.warning(f"Failed to encode {img_meta['image_path']}: {e}")
            continue

    if not embeddings:
        logger.error("No images were successfully encoded!")
        return

    logger.info(f"Successfully encoded {len(embeddings)} images")

    # Convert to numpy array
    embeddings_np = np.array(embeddings, dtype='float32')
    dimension = embeddings_np.shape[1]

    logger.info(f"Embedding dimension: {dimension}")

    # Build FAISS index
    logger.info("Building FAISS index...")

    # Use L2 (Euclidean) distance for normalized vectors
    # For normalized vectors, L2 distance relates to cosine similarity
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    logger.info(f"Index built with {index.ntotal} vectors")

    # Save index
    index_path = output_dir / "clip.index"
    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))

    # Save metadata
    metadata_path = output_dir / "clip_metadata.jsonl"
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for meta in valid_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    # Save config
    config = {
        'country': country,
        'model': model_name,
        'dimension': dimension,
        'num_images': len(embeddings),
        'categories': list(set(m['category'] for m in valid_metadata)),
    }

    config_path = output_dir / "clip_config.json"
    logger.info(f"Saving config to {config_path}")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("="*60)
    logger.info("✓ CLIP Index Built Successfully")
    logger.info("="*60)
    logger.info(f"Images indexed: {len(embeddings)}")
    logger.info(f"Categories: {', '.join(config['categories'])}")
    logger.info(f"Index size: {index_path.stat().st_size / 1024:.1f} KB")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Build CLIP image index from Country Pack")
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., korea)'
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        required=True,
        help='Directory containing images (with category subdirs)'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default=None,
        help='Path to approved_dataset.json with descriptions'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: data/clip_index/<country>)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='CLIP model name'
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "data" / "clip_index" / args.country

    # Set default dataset path
    if args.dataset is None:
        default_dataset = Path.home() / f"ccub2-agent-data/country_packs/{args.country}/approved_dataset.json"
        if default_dataset.exists():
            args.dataset = default_dataset
            logger.info(f"Using default dataset: {args.dataset}")

    # Validate inputs
    if not args.images_dir.exists():
        logger.error(f"Images directory not found: {args.images_dir}")
        sys.exit(1)

    # Build index
    build_clip_index(
        country=args.country,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
