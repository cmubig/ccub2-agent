"""
Configuration for CCUB2 Agent GUI Backend

This file contains paths to data directories, model configurations,
and other settings needed to run the real CCUB2 pipeline.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Project root (ccub2-agent/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Country-specific configurations
COUNTRIES = {
    "korea": {
        "text_index": DATA_DIR / "cultural_index" / "korea",
        "clip_index": DATA_DIR / "clip_index" / "korea",
        "images_dir": DATA_DIR / "country_packs" / "korea" / "images",
        "knowledge_file": DATA_DIR / "cultural_knowledge" / "korea_knowledge.json",
        "dataset_file": DATA_DIR / "country_packs" / "korea" / "approved_dataset_enhanced.json",
    },
    "japan": {
        "text_index": DATA_DIR / "cultural_index" / "japan",
        "clip_index": DATA_DIR / "clip_index" / "japan",
        "images_dir": DATA_DIR / "country_packs" / "japan" / "images",
        "knowledge_file": DATA_DIR / "cultural_knowledge" / "japan_knowledge.json",
        "dataset_file": DATA_DIR / "country_packs" / "japan" / "approved_dataset_enhanced.json",
    },
    # Add more countries as needed
}

# Model configurations
MODELS = {
    "vlm": "Qwen/Qwen3-VL-8B-Instruct",  # VLM for cultural detection
    "clip": "openai/clip-vit-base-patch32",  # CLIP for image similarity
}

# Hugging Face cache directory (for large models)
# Priority: HF_CACHE_DIR env var > /scratch/hf_cache > ~/.cache/huggingface
import os
HF_CACHE_DIR = os.getenv('HF_CACHE_DIR')
if HF_CACHE_DIR:
    HF_CACHE_DIR = Path(HF_CACHE_DIR)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    logger.info(f"Using HF cache from environment: {HF_CACHE_DIR}")
elif Path("/scratch/hf_cache").exists():
    HF_CACHE_DIR = Path("/scratch/hf_cache")
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    logger.info(f"Using /scratch HF cache: {HF_CACHE_DIR}")
else:
    HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"
    logger.info(f"Using default HF cache: {HF_CACHE_DIR}")

# Output directory for generated images
OUTPUT_DIR = PROJECT_ROOT / "gui" / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Session management
MAX_SESSION_AGE_HOURS = 24  # Delete sessions older than 24 hours


def get_country_config(country: str) -> dict:
    """
    Get configuration for a specific country.

    Args:
        country: Country code (e.g., "korea", "japan")

    Returns:
        Dictionary with paths to indexes and data

    Raises:
        ValueError: If country not supported
    """
    if country not in COUNTRIES:
        raise ValueError(
            f"Country '{country}' not supported. "
            f"Available: {list(COUNTRIES.keys())}"
        )

    config = COUNTRIES[country]

    # Validate that indexes exist
    if not config["text_index"].exists():
        logger.warning(
            f"Text index not found for {country}: {config['text_index']}\n"
            f"Run: python scripts/03_indexing/integrate_knowledge_to_rag.py --country {country}"
        )

    if not config["clip_index"].exists():
        logger.warning(
            f"CLIP index not found for {country}: {config['clip_index']}\n"
            f"Run: python scripts/03_indexing/build_clip_image_index.py --country {country}"
        )

    return config


def get_session_dir(pipeline_id: str) -> Path:
    """
    Get directory for a specific pipeline session.

    Args:
        pipeline_id: Unique pipeline ID

    Returns:
        Path to session directory
    """
    session_dir = OUTPUT_DIR / pipeline_id
    session_dir.mkdir(exist_ok=True, parents=True)
    return session_dir


def cleanup_old_sessions():
    """
    Delete session directories older than MAX_SESSION_AGE_HOURS.
    """
    import time

    current_time = time.time()
    max_age_seconds = MAX_SESSION_AGE_HOURS * 3600

    deleted_count = 0
    for session_dir in OUTPUT_DIR.iterdir():
        if not session_dir.is_dir():
            continue

        # Check directory age
        dir_age = current_time - session_dir.stat().st_mtime
        if dir_age > max_age_seconds:
            import shutil
            shutil.rmtree(session_dir)
            deleted_count += 1
            logger.info(f"Deleted old session: {session_dir.name}")

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old sessions")
