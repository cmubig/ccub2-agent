"""Merge curated, user-submitted, and partner datasets.

Combines all data sources into a unified country pack with consistent
provenance tracking and deduplication.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetMerger:
    """Merge multiple data sources into unified country packs."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.curated_dir = self.base_dir / "curated"
        self.country_packs_dir = self.base_dir / "country_packs"

    def merge_country(self, country: str, dry_run: bool = False) -> dict:
        """Merge all data sources for a given country.

        Args:
            country: Country code (e.g., 'korea')
            dry_run: If True, only report what would be merged

        Returns:
            Summary dict with counts per source
        """
        summary = {
            "country": country,
            "user_submitted": 0,
            "curated": 0,
            "partner": 0,
            "total": 0,
            "duplicates_removed": 0,
        }

        # Load existing user-submitted data
        user_data = self._load_user_data(country)
        summary["user_submitted"] = len(user_data)

        # Load curated data from approved directory
        curated_data = self._load_curated_data(country)
        summary["curated"] = len(curated_data)

        # Deduplicate
        all_records, dupes = self._deduplicate(user_data + curated_data)
        summary["duplicates_removed"] = dupes
        summary["total"] = len(all_records)

        if not dry_run:
            self._save_merged(country, all_records)
            logger.info(f"Merged {summary['total']} records for {country}")
        else:
            logger.info(f"[DRY RUN] Would merge {summary['total']} records for {country}")

        return summary

    def _load_user_data(self, country: str) -> list[dict]:
        """Load user-submitted data from country pack."""
        pack_file = self.country_packs_dir / country / "approved_dataset.json"
        if not pack_file.exists():
            return []
        with open(pack_file) as f:
            data = json.load(f)
        # Ensure source field
        for record in data:
            record.setdefault("source", "user_submitted")
        return data

    def _load_curated_data(self, country: str) -> list[dict]:
        """Load curated data from approved directory."""
        approved_dir = self.curated_dir / "approved" / country
        if not approved_dir.exists():
            return []

        records: list[dict] = []
        provenance_file = self.curated_dir / "metadata" / "provenance.jsonl"
        if provenance_file.exists():
            with open(provenance_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("country", "").lower() == country.lower():
                        records.append(record)
        return records

    def _deduplicate(self, records: list[dict]) -> tuple[list[dict], int]:
        """Remove duplicate records based on image_id or source_id."""
        seen: set[str] = set()
        unique: list[dict] = []
        dupes = 0

        for record in records:
            key = record.get("image_id") or record.get("source_id") or record.get("__id__", "")
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            unique.append(record)

        return unique, dupes

    def _save_merged(self, country: str, records: list[dict]) -> None:
        """Save merged dataset to country pack directory."""
        output_dir = self.country_packs_dir / country
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "merged_dataset.json"

        with open(output_file, "w") as f:
            json.dump(records, f, indent=2, default=str)

        logger.info(f"Saved {len(records)} records to {output_file}")
