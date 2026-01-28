#!/usr/bin/env python3
"""
CLIP Retrieval Example: Find similar reference images using CLIP RAG.

This example demonstrates how to use the CLIP Image RAG standalone
to retrieve visually similar reference images from your dataset.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import shutil

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.clip_image_rag import CLIPImageRAG


def retrieve_similar_images(
    query_image_path: str,
    country: str = "korea",
    category: str = None,
    top_k: int = 5,
    output_dir: str = "output/retrieved_references",
):
    """
    Retrieve visually similar images from CLIP index.

    Args:
        query_image_path: Path to query image
        country: Country dataset to search
        category: Optional category filter
        top_k: Number of similar images to retrieve
        output_dir: Where to copy retrieved images

    Returns:
        List of retrieved image paths
    """
    print("=" * 80)
    print("CLIP Image Retrieval")
    print("=" * 80)
    print(f"Query Image: {query_image_path}")
    print(f"Country: {country}")
    print(f"Category: {category or 'all'}")
    print(f"Top-K: {top_k}")
    print()

    # Load query image
    query_image = Image.open(query_image_path).convert("RGB")
    print(f"✓ Query image loaded: {query_image.size[0]}x{query_image.size[1]}")
    print()

    # Initialize CLIP RAG
    print("Initializing CLIP Image RAG...")
    index_dir = PROJECT_ROOT / f"data/clip_index/{country}"
    images_dir = PROJECT_ROOT / f"data/country_packs/{country}/images"

    if not index_dir.exists():
        print(f"Error: CLIP index not found at {index_dir}")
        print(f"Please run: python scripts/indexing/build_clip_image_index.py --country {country}")
        sys.exit(1)

    clip_rag = CLIPImageRAG(
        index_dir=str(index_dir),
        images_dir=str(images_dir),
    )
    print(f"✓ CLIP RAG initialized ({clip_rag.index.ntotal} images indexed)")
    print()

    # Search for similar images
    print("Searching for similar images...")
    results = clip_rag.search_by_image(
        query_image=query_image,
        category=category,
        top_k=top_k,
    )
    print(f"✓ Found {len(results)} similar images")
    print()

    # Display results
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(results, 1):
        print(f"[{idx}] {result['path']}")
        print(f"    Similarity: {result['score']:.4f}")
        print(f"    Category: {result.get('category', 'unknown')}")
        print()

        # Copy to output directory
        src = Path(result['path'])
        dst = output_path / f"ref_{idx:02d}_{src.name}"
        shutil.copy2(src, dst)
        print(f"    → Copied to: {dst}")
        print()

    print("=" * 80)
    print(f"✓ Retrieved images saved to: {output_dir}")
    print("=" * 80)

    return results


def retrieve_by_text(
    query_text: str,
    country: str = "korea",
    category: str = None,
    top_k: int = 5,
    output_dir: str = "output/retrieved_references",
):
    """
    Retrieve images matching text description using CLIP.

    Args:
        query_text: Text description to search for
        country: Country dataset to search
        category: Optional category filter
        top_k: Number of images to retrieve
        output_dir: Where to copy retrieved images

    Returns:
        List of retrieved image paths
    """
    print("=" * 80)
    print("CLIP Text-to-Image Retrieval")
    print("=" * 80)
    print(f"Query Text: {query_text}")
    print(f"Country: {country}")
    print(f"Category: {category or 'all'}")
    print(f"Top-K: {top_k}")
    print()

    # Initialize CLIP RAG
    print("Initializing CLIP Image RAG...")
    index_dir = PROJECT_ROOT / f"data/clip_index/{country}"
    images_dir = PROJECT_ROOT / f"data/country_packs/{country}/images"

    if not index_dir.exists():
        print(f"Error: CLIP index not found at {index_dir}")
        print(f"Please run: python scripts/indexing/build_clip_image_index.py --country {country}")
        sys.exit(1)

    clip_rag = CLIPImageRAG(
        index_dir=str(index_dir),
        images_dir=str(images_dir),
    )
    print(f"✓ CLIP RAG initialized ({clip_rag.index.ntotal} images indexed)")
    print()

    # Search by text
    print("Searching for images matching text...")
    results = clip_rag.search_by_text(
        query_text=query_text,
        category=category,
        top_k=top_k,
    )
    print(f"✓ Found {len(results)} matching images")
    print()

    # Display results
    print("=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(results, 1):
        print(f"[{idx}] {result['path']}")
        print(f"    Similarity: {result['score']:.4f}")
        print(f"    Category: {result.get('category', 'unknown')}")
        print()

        # Copy to output directory
        src = Path(result['path'])
        dst = output_path / f"text_ref_{idx:02d}_{src.name}"
        shutil.copy2(src, dst)
        print(f"    → Copied to: {dst}")
        print()

    print("=" * 80)
    print(f"✓ Retrieved images saved to: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve similar images using CLIP")
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to query image (for image search)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text description (for text-to-image search)",
    )
    parser.add_argument("--country", type=str, default="korea", help="Country dataset")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (traditional_clothing, food, etc.)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of images to retrieve")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/retrieved_references",
        help="Output directory for retrieved images",
    )

    args = parser.parse_args()

    # Determine search mode
    if args.image_path and args.text:
        print("Error: Provide either --image-path OR --text, not both")
        sys.exit(1)
    elif args.image_path:
        # Image-to-image search
        retrieve_similar_images(
            query_image_path=args.image_path,
            country=args.country,
            category=args.category,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )
    elif args.text:
        # Text-to-image search
        retrieve_by_text(
            query_text=args.text,
            country=args.country,
            category=args.category,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )
    else:
        # Demo mode with example queries
        print("No query provided. Running demo with example queries...\n")

        # Example 1: Search by image
        print("\n--- Example 1: Image-to-Image Search ---\n")
        test_image = PROJECT_ROOT / "data/country_packs/korea/images/traditional_clothing"
        if test_image.exists():
            first_image = list(test_image.glob("*.jpg"))[0]
            retrieve_similar_images(
                query_image_path=str(first_image),
                country="korea",
                category="traditional_clothing",
                top_k=3,
                output_dir="output/demo_image_search",
            )
        else:
            print("No test images available for demo")

        # Example 2: Search by text
        print("\n\n--- Example 2: Text-to-Image Search ---\n")
        retrieve_by_text(
            query_text="traditional Korean hanbok with beautiful colors",
            country="korea",
            category="traditional_clothing",
            top_k=3,
            output_dir="output/demo_text_search",
        )
