"""
Country Data Pack

Manages cultural data for each country.
Loads from approved_dataset.json (if available) or falls back to CSV via FirebaseClient.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Any

from .firebase_client import FirebaseClient, normalize_category

logger = logging.getLogger(__name__)

# Base directories
DATA_DIR = Path(__file__).parent.parent.parent / "data"
COUNTRY_PACKS_DIR = DATA_DIR / "country_packs"


class CountryDataPack:
    """Manages cultural data for a single country."""

    def __init__(self, country: str, data_dir: Optional[Path] = None):
        self.country = country.lower().replace(" ", "_")
        self._data_dir = data_dir or DATA_DIR
        self._country_dir = (data_dir or COUNTRY_PACKS_DIR) / self.country
        self._dataset: Optional[List[Dict[str, Any]]] = None

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset: approved_dataset.json first, then CSV fallback."""
        if self._dataset is not None:
            return self._dataset

        # Try approved_dataset.json first
        json_path = self._country_dir / "approved_dataset.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                self._dataset = json.load(f)
            logger.info(f"Loaded {len(self._dataset)} items from {json_path}")
            return self._dataset

        # Fallback to CSV via FirebaseClient
        client = FirebaseClient(data_dir=self._data_dir)
        contribs = client.get_contributions(country=self.country)
        self._dataset = contribs
        logger.info(f"Loaded {len(contribs)} contributions for {self.country} from CSV")
        return self._dataset

    def get_dataset(self) -> List[Dict[str, Any]]:
        """Get all contributions for this country."""
        return self._load_dataset()

    def get_images_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get contributions filtered by normalized category."""
        dataset = self._load_dataset()
        target = category.lower().replace(" ", "_")
        return [
            d for d in dataset
            if d.get("category_normalized", normalize_category(d.get("category", ""))) == target
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics: category counts, status breakdown."""
        dataset = self._load_dataset()

        # Category counts (normalized)
        cat_counter: Counter = Counter()
        for d in dataset:
            cat = d.get("category_normalized", normalize_category(d.get("category", "")))
            cat_counter[cat] += 1

        # Status breakdown
        status_counter: Counter = Counter()
        for d in dataset:
            status_counter[d.get("status", "unknown")] += 1

        return {
            "country": self.country,
            "total_images": len(dataset),
            "categories": dict(cat_counter.most_common()),
            "num_categories": len(cat_counter),
            "status_breakdown": dict(status_counter.most_common()),
        }

    def save_as_approved_dataset(self) -> Path:
        """Save current dataset as approved_dataset.json for downstream compatibility."""
        self._country_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._country_dir / "approved_dataset.json"

        dataset = self._load_dataset()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved approved dataset: {out_path} ({len(dataset)} items)")
        return out_path

    @property
    def categories(self) -> List[str]:
        """List of unique normalized categories."""
        dataset = self._load_dataset()
        cats = set()
        for d in dataset:
            cats.add(d.get("category_normalized", normalize_category(d.get("category", ""))))
        return sorted(cats)

    @property
    def total_images(self) -> int:
        """Total number of images/contributions."""
        return len(self._load_dataset())
