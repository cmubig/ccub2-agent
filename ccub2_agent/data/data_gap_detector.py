"""
Data Gap Detector

Detects gaps in cultural data coverage.
Wraps CountryDataPack + DataGapAnalyzer for convenient access.
"""

import logging
from typing import Dict, List, Any, Optional

from .country_pack import CountryDataPack
from .gap_analyzer import DataGapAnalyzer

logger = logging.getLogger(__name__)


class DataGapDetector:
    """Stateless gap detector that creates internal components on demand."""

    def __init__(self):
        logger.info("DataGapDetector initialized")

    def detect_gaps(
        self,
        country: str,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect coverage gaps for a country.

        Args:
            country: Country name/code.
            category: Optional category filter.

        Returns:
            List of gap dicts.
        """
        pack = CountryDataPack(country)
        analyzer = DataGapAnalyzer(pack)
        gaps = analyzer.analyze(failure_modes=[], country=country)

        if category:
            target = category.lower().replace(" ", "_")
            gaps = [g for g in gaps if g.get("category") == target]

        return gaps

    def get_coverage_summary(self, country: str) -> Dict[str, Any]:
        """
        Get a coverage summary for a country.

        Returns:
            Dict with stats and gap count.
        """
        pack = CountryDataPack(country)
        stats = pack.get_stats()
        analyzer = DataGapAnalyzer(pack)
        gaps = analyzer.analyze(failure_modes=[], country=country)

        high_gaps = [g for g in gaps if g["severity"] == "high"]
        medium_gaps = [g for g in gaps if g["severity"] == "medium"]

        return {
            **stats,
            "total_gaps": len(gaps),
            "high_severity_gaps": len(high_gaps),
            "medium_severity_gaps": len(medium_gaps),
            "gap_categories": [g["category"] for g in gaps],
        }


def create_data_gap_detector() -> DataGapDetector:
    """Create a data gap detector instance."""
    return DataGapDetector()
