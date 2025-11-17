"""
Country data pack - manages cultural data for each country.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class CountryDataPack:
    """
    Manages cultural data for a specific country.

    Data pack structure:
        country_packs/
        ├── korea/
        │   ├── approved_dataset.json
        │   ├── cultural_docs.pdf
        │   ├── glossary.csv
        │   └── metadata.json
        ├── japan/
        └── ...
    """

    def __init__(
        self, country: str, data_pack_path: Optional[Path] = None
    ):
        """
        Initialize country data pack.

        Args:
            country: Country name
            data_pack_path: Path to data packs directory
        """
        self.country = country

        if data_pack_path is None:
            # Default to project data directory
            data_pack_path = Path(__file__).parent.parent.parent / "data" / "country_packs"

        self.pack_path = data_pack_path / country
        logger.info(f"Initializing CountryDataPack for {country}")

        # Load data
        self.approved_dataset = self._load_approved_dataset()
        self.metadata = self._load_metadata()
        self.glossary = self._load_glossary()

        logger.info(
            f"Loaded {len(self.approved_dataset)} approved items for {country}"
        )

    def _load_approved_dataset(self) -> List[Dict]:
        """Load approved image-caption pairs."""
        dataset_path = self.pack_path / "approved_dataset.json"

        if dataset_path.exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Support both {"items": [...]} and direct list format
                if isinstance(data, dict) and "items" in data:
                    return data["items"]
                elif isinstance(data, list):
                    return data
                else:
                    logger.warning(f"Unexpected data format in {dataset_path}")
                    return []
        else:
            logger.warning(
                f"No approved dataset found at {dataset_path}, using empty dataset"
            )
            return []

    def _load_metadata(self) -> Dict:
        """Load pack metadata."""
        metadata_path = self.pack_path / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning(f"No metadata found at {metadata_path}")
            return {
                "country": self.country,
                "version": "0.1.0",
                "last_updated": None,
                "categories": {},
            }

    def _load_glossary(self) -> Dict:
        """Load cultural glossary."""
        glossary_path = self.pack_path / "glossary.csv"

        if glossary_path.exists():
            # Parse CSV
            import csv

            glossary = {}
            with open(glossary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = row.get("term", "")
                    definition = row.get("definition", "")
                    if term:
                        glossary[term] = definition
            return glossary
        else:
            logger.warning(f"No glossary found at {glossary_path}")
            return {}

    def retrieve(
        self, issues: List[Dict], country: str, top_k: int = 3
    ) -> Dict:
        """
        Retrieve relevant examples based on issues.

        Args:
            issues: Detected cultural issues
            country: Target country
            top_k: Number of examples to retrieve

        Returns:
            Dict with:
                - data: List of retrieved examples
                - images: List of image paths
        """
        logger.info(f"Retrieving examples for {len(issues)} issues")

        # Extract categories from issues
        categories = [issue.get("category") for issue in issues]

        # Filter dataset by categories
        relevant_data = [
            item
            for item in self.approved_dataset
            if item.get("category") in categories
        ]

        # If not enough, include general items
        if len(relevant_data) < top_k:
            general_data = [
                item
                for item in self.approved_dataset
                if item not in relevant_data
            ]
            relevant_data.extend(general_data[: top_k - len(relevant_data)])

        # Take top_k
        selected_data = relevant_data[:top_k]

        logger.info(f"Retrieved {len(selected_data)} examples")

        return {
            "data": selected_data,
            "images": [item.get("image_path") for item in selected_data],
        }

    def check_coverage(
        self, issues: List[Dict], country: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if we have data coverage for detected issues.

        Args:
            issues: Detected issues
            country: Target country

        Returns:
            Tuple of (has_coverage, missing_categories)
        """
        categories = [issue.get("category") for issue in issues]
        available_categories = set(
            item.get("category") for item in self.approved_dataset
        )

        missing_categories = [
            cat for cat in categories if cat not in available_categories
        ]

        has_coverage = len(missing_categories) == 0

        return has_coverage, missing_categories

    def get_coverage_ratio(self, category: str, country: str) -> float:
        """
        Get coverage ratio for a specific category.

        Args:
            category: Category name
            country: Target country

        Returns:
            Coverage ratio (0.0 - 1.0)
        """
        # Count items in this category
        category_items = [
            item
            for item in self.approved_dataset
            if item.get("category") == category
        ]

        # Define minimum threshold per category
        min_thresholds = {
            "text": 50,
            "traditional_clothing": 100,
            "architecture": 50,
            "food": 80,
            "symbols": 30,
            "festivals": 40,
        }

        threshold = min_thresholds.get(category, 50)
        ratio = min(1.0, len(category_items) / threshold)

        return ratio

    def update(self, fetch_from_firebase: bool = True):
        """
        Update country pack with latest data.

        Args:
            fetch_from_firebase: Fetch approved data from Firebase
        """
        logger.info(f"Updating country pack for {self.country}")

        if fetch_from_firebase:
            # Firebase fetching not yet implemented
            # Use scripts/01_setup/init_dataset.py instead
            logger.info("Fetching approved data from Firebase...")
            # approved_data = fetch_approved_data(self.country)
            # self._save_approved_dataset(approved_data)

        # Reload data
        self.approved_dataset = self._load_approved_dataset()
        self.metadata = self._load_metadata()

        logger.info("Country pack updated")

    def _save_approved_dataset(self, data: List[Dict]):
        """Save approved dataset to file."""
        dataset_path = self.pack_path / "approved_dataset.json"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data)} approved items")

    def get_glossary_term(self, term: str) -> Optional[str]:
        """
        Get definition for a cultural term.

        Args:
            term: Term to look up

        Returns:
            Definition or None
        """
        return self.glossary.get(term)

    def get_statistics(self) -> Dict:
        """Get statistics about the data pack."""
        # Count by category
        category_counts = {}
        for item in self.approved_dataset:
            category = item.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "total_items": len(self.approved_dataset),
            "categories": category_counts,
            "glossary_terms": len(self.glossary),
            "version": self.metadata.get("version"),
            "last_updated": self.metadata.get("last_updated"),
        }
