#!/usr/bin/env python3
"""
Multi-Country Scaffold for WorldCCUB Phase 1-2

Initializes directory structures, data packs, and tracking for all
target countries defined in configs/country_config.yaml.

Usage:
    # Initialize all countries
    python scripts/setup/multi_country_scaffold.py --init-all

    # Initialize specific country
    python scripts/setup/multi_country_scaffold.py --init-country japan

    # Show status of all countries
    python scripts/setup/multi_country_scaffold.py --status

    # Generate data collection guidelines for reps
    python scripts/setup/multi_country_scaffold.py --generate-guidelines --country japan
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scaffold")

CONFIG_PATH = PROJECT_ROOT / "configs" / "country_config.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_country(country_id: str, config: Dict) -> bool:
    """Initialize directory structure and metadata for a country."""
    logger.info(f"Initializing country: {country_id}")

    # Find country config
    country_cfg = None
    for c in config["countries"]:
        if c["id"] == country_id:
            country_cfg = c
            break

    if country_cfg is None:
        logger.error(f"Country '{country_id}' not found in config")
        return False

    # Create directories
    country_dir = DATA_DIR / "country_packs" / country_id
    dirs = [
        country_dir / "images",
        DATA_DIR / "cultural_knowledge",
        DATA_DIR / "clip_index" / country_id,
        DATA_DIR / "cultural_index" / country_id,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Create metadata file
    categories = config.get("categories", [])
    data_sources = config.get("data_sources", {})

    metadata = {
        "country_id": country_id,
        "display_name": country_cfg.get("display", country_id.title()),
        "phase": country_cfg.get("phase", 2),
        "status": "initialized",
        "target_images": config["benchmark"]["target_images_per_country"],
        "categories": [c["id"] for c in categories],
        "category_targets": {c["id"]: c["min_per_country"] for c in categories},
        "data_sources": {
            source: {
                "target": src_cfg.get("target_per_country", 0),
                "current": 0,
            }
            for source, src_cfg in data_sources.items()
        },
        "reps": country_cfg.get("reps", []),
        "current_images": 0,
        "category_counts": {c["id"]: 0 for c in categories},
    }

    metadata_path = country_dir / "country_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Create empty approved dataset if not exists
    dataset_path = country_dir / "approved_dataset.json"
    if not dataset_path.exists():
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    logger.info(f"  Initialized: {country_dir}")
    logger.info(f"  Target: {metadata['target_images']} images, {len(categories)} categories")
    return True


def show_status(config: Dict):
    """Show status of all countries."""
    logger.info("=" * 80)
    logger.info(f"{'Country':20s} | {'Phase':>5s} | {'Status':>12s} | {'Images':>7s} | {'Target':>7s} | {'Pct':>5s}")
    logger.info("-" * 80)

    for country_cfg in config["countries"]:
        cid = country_cfg["id"]
        country_dir = DATA_DIR / "country_packs" / cid

        # Count images
        images_dir = country_dir / "images"
        n_images = len(list(images_dir.glob("*.*"))) if images_dir.exists() else 0
        target = config["benchmark"]["target_images_per_country"]
        pct = round(n_images / target * 100, 1) if target > 0 else 0

        status = "active" if country_dir.exists() else "planned"
        if n_images > 0:
            status = "has_data"

        logger.info(
            f"{country_cfg.get('display', cid):20s} | "
            f"{country_cfg.get('phase', '?'):>5s} | "
            f"{status:>12s} | "
            f"{n_images:>7d} | "
            f"{target:>7d} | "
            f"{pct:>4.1f}%"
        )

    logger.info("=" * 80)

    # Summary
    total_target = config["benchmark"]["target_images_per_country"] * len(config["countries"])
    total_current = sum(
        len(list((DATA_DIR / "country_packs" / c["id"] / "images").glob("*.*")))
        for c in config["countries"]
        if (DATA_DIR / "country_packs" / c["id"] / "images").exists()
    )
    logger.info(f"Total: {total_current}/{total_target} images ({total_current/total_target*100:.1f}%)")


def generate_guidelines(country_id: str, config: Dict) -> Path:
    """Generate data collection guidelines for country reps."""
    country_cfg = None
    for c in config["countries"]:
        if c["id"] == country_id:
            country_cfg = c
            break

    if country_cfg is None:
        logger.error(f"Country '{country_id}' not found")
        return None

    categories = config.get("categories", [])
    target = config["benchmark"]["target_images_per_country"]
    per_cat = config["benchmark"]["images_per_category"]

    guidelines = {
        "country": country_id,
        "display_name": country_cfg.get("display", country_id.title()),
        "target_total_images": target,
        "categories": [
            {
                "id": c["id"],
                "name": c["display"],
                "target_images": per_cat,
                "min_images": c["min_per_country"],
                "examples": _get_category_examples(c["id"], country_id),
            }
            for c in categories
        ],
        "rep_responsibilities": {
            "direct_submissions": 50,
            "agent_verification": 50,
            "total_per_rep": 100,
            "submission_guidelines": [
                "Images must be your own original photos or clearly licensed",
                "Each image should represent an authentic cultural element",
                "Aim for 6-7 images per category to ensure balance",
                "Include brief Korean/English description for each image",
                "Avoid images with identifiable faces (privacy)",
                "Minimum resolution: 512x512 pixels",
            ],
            "verification_guidelines": [
                "Review agent-curated images for cultural accuracy",
                "Flag images with cultural misrepresentation",
                "Approve images that accurately represent your culture",
                "Suggest better alternatives when possible",
            ],
        },
        "data_sources": {
            "user_submitted": {
                "pct": 60,
                "description": "Your own photos + contributor photos via WorldCCUB app",
            },
            "agent_curated": {
                "pct": 30,
                "description": "Wikimedia/Pixabay images curated by our agent — you verify",
            },
            "partner": {
                "pct": 10,
                "description": "Museum/university partnerships",
            },
        },
    }

    output_dir = DATA_DIR / "country_packs" / country_id
    output_dir.mkdir(parents=True, exist_ok=True)
    guidelines_path = output_dir / "collection_guidelines.json"
    with open(guidelines_path, "w", encoding="utf-8") as f:
        json.dump(guidelines, f, indent=2, ensure_ascii=False)

    logger.info(f"Guidelines saved: {guidelines_path}")
    return guidelines_path


def _get_category_examples(category_id: str, country_id: str) -> List[str]:
    """Get example descriptions for a category."""
    examples = {
        "architecture": [
            f"Traditional {country_id} temple or shrine",
            f"Historical {country_id} palace or fortress",
            f"Traditional residential architecture",
        ],
        "art": [
            f"Traditional {country_id} painting or calligraphy",
            f"Handicraft or folk art",
            f"Traditional sculpture or pottery",
        ],
        "food": [
            f"Traditional {country_id} dish",
            f"Street food or market scene",
            f"Traditional cooking method or utensils",
        ],
        "fashion": [
            f"Traditional {country_id} clothing/costume",
            f"Traditional accessories or jewelry",
            f"Modern fashion with cultural elements",
        ],
        "event": [
            f"Traditional {country_id} festival",
            f"Religious or cultural ceremony",
            f"National celebration or holiday",
        ],
        "people": [
            f"Traditional {country_id} daily life scene",
            f"Cultural practice or tradition",
            f"Traditional occupation or craft",
        ],
        "landscape": [
            f"Iconic {country_id} natural landscape",
            f"Culturally significant natural site",
            f"Traditional garden or park",
        ],
        "wildlife": [
            f"Native {country_id} animal species",
            f"Culturally significant animal",
            f"Animal in traditional context",
        ],
    }
    return examples.get(category_id, [f"Cultural image of {category_id}"])


def main():
    parser = argparse.ArgumentParser(description="Multi-Country Scaffold")
    parser.add_argument("--init-all", action="store_true", help="Initialize all countries")
    parser.add_argument("--init-country", type=str, help="Initialize specific country")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--generate-guidelines", action="store_true", help="Generate rep guidelines")
    parser.add_argument("--country", type=str, help="Country for guidelines")
    args = parser.parse_args()

    config = load_config()

    if args.status:
        show_status(config)
    elif args.init_all:
        for c in config["countries"]:
            init_country(c["id"], config)
    elif args.init_country:
        init_country(args.init_country, config)
    elif args.generate_guidelines:
        if not args.country:
            parser.error("--generate-guidelines requires --country")
        generate_guidelines(args.country, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
