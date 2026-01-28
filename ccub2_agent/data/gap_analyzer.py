"""
Data Gap Analyzer

Identifies missing data categories and coverage gaps.
Used by ScoutAgent to detect what cultural data is underrepresented.
"""

import logging
from typing import Dict, List, Any

from .firebase_client import normalize_category

logger = logging.getLogger(__name__)

# All expected normalized categories
EXPECTED_CATEGORIES = [
    "city_street",
    "nature_landscape",
    "food_drink",
    "architecture",
    "people_action",
    "traditional_clothing",
    "religion_festival",
    "arts",
    "entertainment",
    "sports",
    "daily_life",
    "animals",
]

# Minimum images per category to consider it adequately covered
MINIMUM_PER_CATEGORY = 15


class DataGapAnalyzer:
    """Analyzes coverage gaps in country data."""

    def __init__(self, country_pack):
        self.country_pack = country_pack

    def analyze(
        self,
        failure_modes: List[Dict[str, Any]],
        country: str,
    ) -> List[Dict[str, Any]]:
        """
        Analyze gaps based on category coverage and failure modes.

        Args:
            failure_modes: List of failure dicts from JudgeAgent (may be empty).
            country: Country code/name.

        Returns:
            List of gap dicts with keys: element, category, severity, current_count,
            needed, source.
        """
        gaps: List[Dict[str, Any]] = []

        # 1. Category coverage gaps
        stats = self.country_pack.get_stats()
        cat_counts = stats.get("categories", {})

        for cat in EXPECTED_CATEGORIES:
            count = cat_counts.get(cat, 0)
            if count < MINIMUM_PER_CATEGORY:
                deficit = MINIMUM_PER_CATEGORY - count
                if count == 0:
                    severity = "high"
                elif count < MINIMUM_PER_CATEGORY // 2:
                    severity = "high"
                elif count < MINIMUM_PER_CATEGORY:
                    severity = "medium"
                else:
                    severity = "low"

                gaps.append({
                    "element": f"{cat} images for {country}",
                    "category": cat,
                    "severity": severity,
                    "current_count": count,
                    "needed": deficit,
                    "source": "coverage_analysis",
                })

        # 2. Failure-mode-derived gaps
        for fm in failure_modes:
            fm_cat = fm.get("category", "")
            fm_element = fm.get("element", fm.get("description", "unknown"))
            normalized = normalize_category(fm_cat) if fm_cat else "uncategorized"

            gaps.append({
                "element": fm_element,
                "category": normalized,
                "severity": fm.get("severity", "medium"),
                "current_count": cat_counts.get(normalized, 0),
                "needed": fm.get("needed", 5),
                "source": "failure_mode",
            })

        # Sort: high severity first, then by deficit
        severity_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: (severity_order.get(g["severity"], 9), -g.get("needed", 0)))

        logger.info(f"Gap analysis for {country}: {len(gaps)} gaps found")
        return gaps
