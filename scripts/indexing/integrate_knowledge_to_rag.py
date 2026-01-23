#!/usr/bin/env python3
"""
Integrate extracted cultural knowledge into RAG index.

This script:
1. Loads extracted knowledge from extract_cultural_knowledge.py
2. Converts knowledge to text documents
3. Integrates into existing FAISS index (or creates new one)
4. Updates cultural_index/ with Wikipedia + Enhanced Captions + Image Knowledge

Usage:
    python scripts/integrate_knowledge_to_rag.py \
        --knowledge-file /scratch/.../cultural_knowledge/korea_knowledge.json \
        --index-dir /scratch/.../cultural_index/korea \
        --rebuild  # Optional: rebuild from scratch
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeDocument:
    """A text document for RAG indexing."""
    text: str
    metadata: Dict
    source: str
    item_id: str


def load_extracted_knowledge(knowledge_file: Path) -> List[Dict]:
    """Load extracted knowledge JSON."""
    logger.info(f"Loading knowledge from {knowledge_file}")
    with open(knowledge_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    knowledge_items = data.get('knowledge', [])
    logger.info(f"Loaded {len(knowledge_items)} knowledge items")
    return knowledge_items


def convert_knowledge_to_documents(knowledge_items: List[Dict], country: str) -> List[KnowledgeDocument]:
    """
    Convert extracted cultural knowledge to RAG documents.

    Creates multiple documents per knowledge item for better retrieval:
    1. Visual features document
    2. Cultural elements document
    3. Key characteristics document
    4. Common mistakes document (what to avoid)
    """
    documents = []

    for item in tqdm(knowledge_items, desc="Converting knowledge"):
        item_id = item['item_id']
        category = item['category']

        # Document 1: Visual Features & Materials
        if item.get('visual_features') or item.get('materials_textures'):
            visual_text = f"Visual description of {country} {category}: "
            if item.get('visual_features'):
                visual_text += item['visual_features'] + " "
            if item.get('materials_textures'):
                visual_text += f"Materials and textures: {item['materials_textures']} "
            if item.get('colors_patterns'):
                visual_text += f"Colors and patterns: {item['colors_patterns']}"

            documents.append(KnowledgeDocument(
                text=visual_text.strip(),
                metadata={
                    "country": country,
                    "category": category,
                    "section": "visual_features",
                    "quality_score": item.get('quality_score', 0)
                },
                source="image_knowledge",
                item_id=item_id
            ))

        # Document 2: Cultural Elements (what makes it authentic)
        if item.get('cultural_elements'):
            cultural_text = f"Authentic {country} {category} cultural elements: {item['cultural_elements']}"

            documents.append(KnowledgeDocument(
                text=cultural_text,
                metadata={
                    "country": country,
                    "category": category,
                    "section": "cultural_elements",
                    "quality_score": item.get('quality_score', 0)
                },
                source="image_knowledge",
                item_id=item_id
            ))

        # Document 3: Correct Aspects (specific details that are right)
        if item.get('correct_aspects') and len(item['correct_aspects']) > 0:
            correct_text = f"Correct aspects of authentic {country} {category}: "
            correct_text += ", ".join(item['correct_aspects']) + ". "
            if item.get('key_characteristics'):
                correct_text += f"Key characteristics: {item['key_characteristics']}"

            documents.append(KnowledgeDocument(
                text=correct_text.strip(),
                metadata={
                    "country": country,
                    "category": category,
                    "section": "correct_aspects",
                    "quality_score": item.get('quality_score', 0)
                },
                source="image_knowledge",
                item_id=item_id
            ))

        # Document 4: Common Mistakes (what to avoid)
        if item.get('common_mistakes'):
            mistakes_text = f"Common mistakes to avoid in {country} {category}: {item['common_mistakes']}"

            documents.append(KnowledgeDocument(
                text=mistakes_text,
                metadata={
                    "country": country,
                    "category": category,
                    "section": "common_mistakes",
                    "quality_score": item.get('quality_score', 0)
                },
                source="image_knowledge",
                item_id=item_id
            ))

    logger.info(f"Created {len(documents)} documents from {len(knowledge_items)} knowledge items")
    return documents


def load_existing_index(index_dir: Path):
    """Load existing FAISS index and metadata."""
    index_file = index_dir / "faiss.index"
    metadata_file = index_dir / "metadata.jsonl"

    if not index_file.exists():
        logger.info("No existing index found")
        return None, []

    logger.info(f"Loading existing index from {index_dir}")
    index = faiss.read_index(str(index_file))

    # Load metadata
    metadata = []
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                metadata.append(json.loads(line))

    logger.info(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
    return index, metadata


def build_or_update_index(
    new_documents: List[KnowledgeDocument],
    existing_index,
    existing_metadata: List[Dict],
    embedder: SentenceTransformer,
    rebuild: bool = False
) -> tuple:
    """Build new index or update existing one."""

    # Generate embeddings for new documents
    logger.info(f"Generating embeddings for {len(new_documents)} new documents...")
    new_texts = [doc.text for doc in new_documents]
    new_embeddings = embedder.encode(
        new_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Normalize embeddings
    faiss.normalize_L2(new_embeddings)

    # Prepare new metadata
    new_metadata = []
    for doc, embedding in zip(new_documents, new_embeddings):
        new_metadata.append({
            "country": doc.metadata['country'],
            "category": doc.metadata['category'],
            "section": doc.metadata['section'],
            "source": doc.source,
            "item_id": doc.item_id,
            "text": doc.text,
            "quality_score": doc.metadata.get('quality_score', 0)
        })

    if rebuild or existing_index is None:
        # Build new index from scratch
        logger.info("Building new FAISS index...")
        dimension = new_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity after normalization)
        index.add(new_embeddings)
        all_metadata = new_metadata
    else:
        # Add to existing index
        logger.info("Adding to existing index...")
        index = existing_index
        index.add(new_embeddings)
        all_metadata = existing_metadata + new_metadata

    logger.info(f"Final index size: {index.ntotal} vectors")
    return index, all_metadata


def save_index(index, metadata: List[Dict], output_dir: Path):
    """Save FAISS index and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    index_file = output_dir / "faiss.index"
    metadata_file = output_dir / "metadata.jsonl"
    config_file = output_dir / "index_config.json"

    # Save index
    logger.info(f"Saving index to {index_file}")
    faiss.write_index(index, str(index_file))

    # Save metadata
    logger.info(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Save config
    config = {
        "index_type": "faiss.IndexFlatIP",
        "dimension": index.d,
        "total_vectors": index.ntotal,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "sources": list(set(m['source'] for m in metadata)),
        "last_updated": str(Path(__file__).stat().st_mtime)
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Index saved successfully!")
    logger.info(f"   Total vectors: {index.ntotal}")
    logger.info(f"   Metadata entries: {len(metadata)}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate cultural knowledge into RAG index"
    )
    parser.add_argument(
        "--knowledge-file",
        type=Path,
        required=True,
        help="Path to extracted knowledge JSON (e.g., PROJECT_ROOT/data/cultural_knowledge/korea_knowledge.json)"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Path to FAISS index directory (e.g., PROJECT_ROOT/data/cultural_index/korea)"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index from scratch (otherwise appends)"
    )

    args = parser.parse_args()

    # Check if knowledge file exists
    if not args.knowledge_file.exists():
        logger.error(f"Knowledge file not found: {args.knowledge_file}")
        logger.error("Please run extract_cultural_knowledge.py first!")
        return

    # Load extracted knowledge
    knowledge_items = load_extracted_knowledge(args.knowledge_file)
    if not knowledge_items:
        logger.error("No knowledge items found!")
        return

    # Get country from path
    country = args.knowledge_file.stem.split('_')[0]  # e.g., "korea" from "korea_knowledge.json"
    logger.info(f"Country: {country}")

    # Convert to documents
    documents = convert_knowledge_to_documents(knowledge_items, country)

    # Load embedder
    logger.info(f"Loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model)

    # Load existing index (if not rebuilding)
    existing_index, existing_metadata = None, []
    if not args.rebuild and args.index_dir.exists():
        existing_index, existing_metadata = load_existing_index(args.index_dir)

    # Build or update index
    final_index, final_metadata = build_or_update_index(
        new_documents=documents,
        existing_index=existing_index,
        existing_metadata=existing_metadata,
        embedder=embedder,
        rebuild=args.rebuild
    )

    # Save
    save_index(final_index, final_metadata, args.index_dir)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("INTEGRATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Knowledge items processed: {len(knowledge_items)}")
    logger.info(f"Documents created: {len(documents)}")
    logger.info(f"Total vectors in index: {final_index.ntotal}")
    logger.info(f"Index location: {args.index_dir}")
    logger.info("")

    # Source breakdown
    source_counts = {}
    for meta in final_metadata:
        source = meta['source']
        source_counts[source] = source_counts.get(source, 0) + 1

    logger.info("Source breakdown:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  - {source}: {count} documents")


if __name__ == "__main__":
    main()
