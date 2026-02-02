"""
CLIP-based image RAG for retrieving similar cultural reference images.

This module uses CLIP to encode images and retrieve visually similar
reference images from the Country Pack for cultural comparison.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import logging

import time

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss

logger = logging.getLogger(__name__)


class CLIPImageRAG:
    """
    CLIP-based image retrieval for cultural reference images.

    Uses OpenAI CLIP to encode images as vectors and retrieve
    visually similar reference images from FAISS index.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        index_dir: Optional[Path] = None,
        device: str = "auto",
    ):
        """
        Initialize CLIP image RAG.

        Args:
            model_name: CLIP model name from HuggingFace
            index_dir: Directory containing FAISS index and metadata
            device: Device to use (auto/cuda/cpu)
        """
        self.model_name = model_name
        self.index_dir = Path(index_dir) if index_dir else None

        # Determine device - use GPU 1 if available (GPU 0 is for VLM)
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Initializing CLIP model {model_name} on {self.device}")

        # Load CLIP model and processor with retry (transient HF Hub errors)
        self.model = None
        self.processor = None
        for attempt in range(3):
            try:
                self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model.eval()
                break
            except Exception as e:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"CLIP load attempt {attempt+1}/3 failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                if attempt == 2:
                    raise

        # Load FAISS index if provided
        self.index = None
        self.metadata = []
        self.config = {}

        self._is_loaded = True  # Track whether model is on GPU

        if self.index_dir and self.index_dir.exists():
            self._load_index()
        else:
            logger.warning(f"No index found at {self.index_dir}")

    @property
    def is_loaded(self) -> bool:
        """Whether the CLIP model is currently on GPU."""
        return self._is_loaded

    def unload(self):
        """Move CLIP model to CPU and free GPU memory for other models."""
        if not self._is_loaded:
            logger.debug("CLIP model already unloaded")
            return
        try:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._is_loaded = False
            logger.info("CLIP model unloaded from GPU → CPU (freed VRAM)")
        except Exception as e:
            logger.warning(f"Failed to unload CLIP model: {e}")

    def reload(self):
        """Move CLIP model back to GPU after other models are done."""
        if self._is_loaded:
            logger.debug("CLIP model already loaded on GPU")
            return
        try:
            self.model.to(self.device)
            self._is_loaded = True
            logger.info(f"CLIP model reloaded to {self.device}")
        except Exception as e:
            logger.warning(f"Failed to reload CLIP model: {e}")

    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        index_path = self.index_dir / "clip.index"
        metadata_path = self.index_dir / "clip_metadata.jsonl"
        config_path = self.index_dir / "clip_config.json"

        if not index_path.exists():
            logger.warning(f"CLIP index not found at {index_path}")
            return

        logger.info(f"Loading CLIP index from {index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(self.metadata)} image entries")

        # Load config
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info(f"Index config: {self.config}")

    @torch.no_grad()
    def encode_image(self, image_path: Path) -> np.ndarray:
        """
        Encode an image to CLIP vector.

        Args:
            image_path: Path to image file

        Returns:
            512-dim CLIP embedding (normalized)
        """
        last_error = None
        for attempt in range(3):
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get CLIP image embedding (vision_model → visual_projection → 512-dim)
                vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                image_embeds = self.model.visual_projection(vision_outputs.pooler_output)

                # Normalize (CLIP embeddings should be normalized for cosine similarity)
                embedding = image_embeds.cpu().numpy()[0]
                embedding = embedding / np.linalg.norm(embedding)

                return embedding
            except Exception as e:
                last_error = e
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"CLIP encode_image attempt {attempt+1}/3 failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"CLIP encode_image failed after 3 attempts: {last_error}")

    def retrieve_similar_images(
        self,
        image_path: Path,
        k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar images from the index.

        Args:
            image_path: Query image path
            k: Number of results to return
            category: Optional category filter (food, traditional_clothing, etc.)

        Returns:
            List of dicts with 'image_path', 'similarity', 'metadata'
        """
        if self.index is None:
            logger.warning("No FAISS index loaded, cannot retrieve")
            return []

        # Encode query image
        query_vec = self.encode_image(image_path)
        query_vec = query_vec.reshape(1, -1).astype('float32')

        # Search FAISS index
        # Get more results if we need to filter by category
        search_k = k * 5 if category else k
        distances, indices = self.index.search(query_vec, search_k)

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            meta = self.metadata[idx]

            # Filter by category if specified
            if category and meta.get('category') != category:
                continue

            # Convert L2 distance to cosine similarity
            # For normalized vectors: cosine_sim = 1 - (L2^2 / 2)
            similarity = 1.0 - (dist ** 2) / 2.0

            results.append({
                'image_path': meta['image_path'],
                'similarity': float(similarity),
                'category': meta.get('category', 'unknown'),
                'metadata': meta,
            })

            if len(results) >= k:
                break

        logger.info(f"Retrieved {len(results)} similar images")
        return results

    def get_reference_context(
        self,
        image_path: Path,
        k: int = 3,
        category: Optional[str] = None,
    ) -> str:
        """
        Get textual context about retrieved reference images.

        Args:
            image_path: Query image path
            k: Number of references to retrieve
            category: Optional category filter

        Returns:
            Formatted string describing reference images with descriptions
        """
        results = self.retrieve_similar_images(image_path, k=k, category=category)

        if not results:
            return "No reference images found."

        context_parts = [f"Found {len(results)} similar reference images:\n"]

        for i, result in enumerate(results, 1):
            meta = result['metadata']
            sim_pct = result['similarity'] * 100

            # Basic info
            context_parts.append(
                f"{i}. {meta.get('category', 'unknown')} "
                f"(similarity: {sim_pct:.1f}%)"
            )

            # Add enhanced description (preferred) or original description
            desc = meta.get('description_enhanced') or meta.get('description', '')
            if desc:
                # Truncate if too long
                if len(desc) > 150:
                    desc = desc[:147] + "..."
                context_parts.append(f"   {desc}")

        return "\n".join(context_parts)


def create_clip_rag(
    model_name: str = "openai/clip-vit-base-patch32",
    index_dir: Optional[Path] = None,
    device: str = "auto",
) -> CLIPImageRAG:
    """
    Factory function to create CLIP image RAG.

    Args:
        model_name: CLIP model name
        index_dir: Directory with FAISS index
        device: Device to use

    Returns:
        CLIPImageRAG instance
    """
    return CLIPImageRAG(
        model_name=model_name,
        index_dir=index_dir,
        device=device,
    )
