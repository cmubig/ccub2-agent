#!/usr/bin/env python3
"""
Build FAISS index from Country Pack data for cultural RAG retrieval.

This adapts build_cultural_index.py to work with Country Pack JSON data
instead of Wikipedia PDFs.

Usage:
    python scripts/build_country_pack_index.py --country korea --data-dir ./data
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable
import logging

from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CulturalChunk:
    """Chunk of cultural knowledge from Country Pack."""
    country: str
    category: str
    section: str  # For compatibility with cultural_metric
    text: str
    source: str  # "worldccub" or other
    item_id: str


def load_country_pack_items(country_dir: Path) -> List[dict]:
    """Load items from approved_dataset.json (prefer enhanced version)."""
    # Prefer enhanced dataset with VLM-improved captions
    enhanced_path = country_dir / "approved_dataset_enhanced.json"
    dataset_path = country_dir / "approved_dataset.json"

    if enhanced_path.exists():
        logger.info(f"Using enhanced dataset: {enhanced_path}")
        with open(enhanced_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif dataset_path.exists():
        logger.info(f"Using original dataset: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"No dataset found in {country_dir}")

    # Support both {"items": [...]} and direct list
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected dataset format")


def extract_chunks_from_items(items: List[dict], country: str) -> Iterable[CulturalChunk]:
    """
    Extract text chunks from Country Pack items.

    Creates multiple chunks per item for better retrieval:
    - Description chunk
    - Cultural notes chunk (if available)
    - Key features chunk (if available)
    """
    for item in items:
        item_id = item.get('id', 'unknown')
        category = item.get('category', 'uncategorized')
        source = item.get('source', 'unknown')

        # 1. Description chunk (prefer enhanced)
        description_enhanced = item.get('description_enhanced', '').strip()
        description_original = item.get('description', '').strip()
        description = description_enhanced if description_enhanced else description_original

        if description:
            yield CulturalChunk(
                country=country,
                category=category,
                section="description",
                text=description,
                source=source,
                item_id=item_id,
            )

        # 2. Cultural notes chunk
        cultural_notes = item.get('cultural_notes', '').strip()
        if cultural_notes:
            # Combine description + cultural notes for richer context
            combined = f"{description}. Cultural context: {cultural_notes}"
            yield CulturalChunk(
                country=country,
                category=category,
                section="cultural_notes",
                text=combined,
                source=source,
                item_id=item_id,
            )

        # 3. Key features chunk
        key_features = item.get('key_features', [])
        if key_features:
            features_text = f"{description}. Key features: {', '.join(key_features)}"
            yield CulturalChunk(
                country=country,
                category=category,
                section="key_features",
                text=features_text,
                source=source,
                item_id=item_id,
            )


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
    """
    Split text into smaller chunks with overlap.

    Country Pack descriptions are typically short, so we use smaller chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break
        start = max(end - overlap, 0)

    return chunks


def build_index(
    country: str,
    data_dir: Path,
    out_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 200,
) -> None:
    """
    Build FAISS index from Country Pack data.

    Args:
        country: Country name
        data_dir: Data directory containing country_packs/
        out_dir: Output directory for index
        model_name: SentenceTransformer model name
        chunk_size: Maximum chunk size in words
    """
    logger.info(f"Building index for {country}")

    # Load country pack data
    country_dir = data_dir / "country_packs" / country
    if not country_dir.exists():
        raise ValueError(f"Country pack not found: {country_dir}")

    items = load_country_pack_items(country_dir)
    logger.info(f"Loaded {len(items)} items from {country} pack")

    # Extract chunks
    all_chunks: List[CulturalChunk] = []
    for cultural_chunk in extract_chunks_from_items(items, country):
        # Split long texts
        for text_chunk in chunk_text(cultural_chunk.text, chunk_size):
            all_chunks.append(CulturalChunk(
                country=cultural_chunk.country,
                category=cultural_chunk.category,
                section=cultural_chunk.section,
                text=text_chunk,
                source=cultural_chunk.source,
                item_id=cultural_chunk.item_id,
            ))

    if not all_chunks:
        raise ValueError(f"No chunks extracted from {country} pack")

    logger.info(f"Created {len(all_chunks)} text chunks")

    # Build embeddings
    logger.info(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    logger.info("Generating embeddings...")
    embeddings = embedder.encode(
        [chunk.text for chunk in all_chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)

    logger.info(f"Built FAISS index with {index.ntotal} vectors (dimension={dimension})")

    # Save index
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    logger.info(f"Saved index to {out_dir / 'faiss.index'}")

    # Save metadata
    metadata = [
        {
            "country": chunk.country,
            "category": chunk.category,
            "section": chunk.section,  # For compatibility with EnhancedCulturalKnowledgeBase
            "source": chunk.source,
            "item_id": chunk.item_id,
            "text": chunk.text,
        }
        for chunk in all_chunks
    ]

    metadata_path = out_dir / "metadata.jsonl"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    logger.info(f"Saved metadata to {metadata_path}")

    # Save config
    config = {
        "model_name": model_name,
        "embedding_dim": dimension,
        "chunk_count": len(all_chunks),
        "country": country,
        "source": "country_pack",
    }

    config_path = out_dir / "index_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved config to {config_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info(f"✓ Index built successfully for {country}!")
    logger.info("="*60)
    logger.info(f"Chunks: {len(all_chunks)}")
    logger.info(f"Dimension: {dimension}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {out_dir}")

    # Category breakdown
    category_counts = {}
    for chunk in all_chunks:
        category_counts[chunk.category] = category_counts.get(chunk.category, 0) + 1

    logger.info("\nCategory breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count} chunks")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from Country Pack data"
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., korea, japan, china)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Data directory (default: auto-detect)'
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='Output directory (default: data/cultural_index/<country>)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='SentenceTransformer model name'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='Maximum chunk size in words'
    )

    args = parser.parse_args()

    # Auto-detect data directory
    if args.data_dir is None:
        script_dir = Path(__file__).parent
        args.data_dir = script_dir.parent / "data"

    # Auto-detect output directory
    if args.out_dir is None:
        args.out_dir = args.data_dir / "cultural_index" / args.country

    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.out_dir}")

    build_index(
        country=args.country,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
