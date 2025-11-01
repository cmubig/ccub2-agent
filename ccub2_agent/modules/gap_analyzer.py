"""
Data gap analyzer - identifies missing data categories.
"""

from typing import Dict, List
import logging

from .country_pack import CountryDataPack

logger = logging.getLogger(__name__)


class DataGapAnalyzer:
    """
    Analyze detected issues to identify data gaps.

    This determines what specific data is missing and needs to be collected.
    """

    def __init__(self, country_pack: CountryDataPack):
        """
        Initialize gap analyzer.

        Args:
            country_pack: Country data pack to check against
        """
        self.country_pack = country_pack

    def analyze(self, issues: List[Dict], country: str) -> List[Dict]:
        """
        Analyze issues to identify data gaps.

        Args:
            issues: List of detected issues
            country: Target country

        Returns:
            List of data gap specifications, each containing:
                - category: Gap category
                - keywords: Keywords for this gap
                - description: Description for job posting
                - severity: Average severity
                - priority: Collection priority (1-10)
        """
        logger.info(f"Analyzing data gaps for {len(issues)} issues")

        # Group issues by category
        category_issues = {}
        for issue in issues:
            category = issue.get("category", "general")
            if category not in category_issues:
                category_issues[category] = []
            category_issues[category].append(issue)

        # Analyze each category
        gaps = []
        for category, cat_issues in category_issues.items():
            # Check if we have sufficient data for this category
            coverage_ratio = self.country_pack.get_coverage_ratio(
                category, country
            )

            if coverage_ratio < 0.5:  # Less than 50% coverage
                gap = self._create_gap_spec(category, cat_issues, country)
                gaps.append(gap)
                logger.info(
                    f"Gap identified: {category} (coverage: {coverage_ratio:.1%})"
                )

        # Sort by priority
        gaps.sort(key=lambda x: x["priority"], reverse=True)

        logger.info(f"Identified {len(gaps)} data gaps")
        return gaps

    def _create_gap_spec(
        self, category: str, issues: List[Dict], country: str
    ) -> Dict:
        """
        Create a gap specification for job creation.

        Args:
            category: Gap category
            issues: Issues in this category
            country: Target country

        Returns:
            Gap specification dict
        """
        # Extract keywords from issue descriptions
        keywords = self._extract_keywords(issues, category, country)

        # Calculate average severity
        avg_severity = sum(i.get("severity", 5) for i in issues) / len(issues)

        # Generate description
        description = self._generate_description(category, issues, country)

        # Calculate priority (based on severity and frequency)
        priority = min(10, int(avg_severity * len(issues) / 2))

        return {
            "category": category,
            "keywords": keywords,
            "description": description,
            "severity": avg_severity,
            "priority": priority,
            "issue_count": len(issues),
        }

    def _extract_keywords(
        self, issues: List[Dict], category: str, country: str
    ) -> List[str]:
        """Extract relevant keywords for data collection."""
        keywords = []

        # Add category-specific keywords
        category_keywords = {
            "text": {
                "korea": ["hangul", "korean text", "한글"],
                "japan": ["kanji", "hiragana", "katakana"],
                "china": ["chinese characters", "hanzi"],
            },
            "traditional_clothing": {
                "korea": ["hanbok", "jeogori", "chima"],
                "japan": ["kimono", "yukata"],
                "china": ["hanfu", "qipao"],
                "india": ["sari", "kurta"],
            },
            "architecture": {
                "korea": ["hanok", "korean palace", "traditional roof"],
                "japan": ["shrine", "temple", "pagoda"],
                "china": ["forbidden city", "traditional architecture"],
            },
            "food": {
                "korea": ["kimchi", "bibimbap", "traditional food"],
                "japan": ["sushi", "ramen", "traditional cuisine"],
                "india": ["curry", "naan", "traditional dishes"],
            },
        }

        # Get keywords for this category and country
        if category in category_keywords:
            if country in category_keywords[category]:
                keywords.extend(category_keywords[category][country])

        # Add generic keywords
        keywords.append(f"{country} culture")
        keywords.append(f"traditional {country}")

        return keywords

    def _generate_description(
        self, category: str, issues: List[Dict], country: str
    ) -> str:
        """Generate job description based on issues."""
        category_names = {
            "text": "text and writing",
            "traditional_clothing": "traditional clothing",
            "architecture": "architecture",
            "food": "food and cuisine",
            "symbols": "cultural symbols",
            "festivals": "festivals and celebrations",
        }

        category_name = category_names.get(category, category.replace("_", " "))

        description = (
            f"We need more authentic examples of {country.title()} {category_name} "
            f"to improve cultural accuracy in image generation.\n\n"
        )

        # Add specific issues
        description += "Specifically, we need help with:\n"
        for i, issue in enumerate(issues[:3], 1):  # Top 3 issues
            description += f"{i}. {issue['description']}\n"

        description += (
            f"\nYour contributions will help AI systems better understand and "
            f"represent {country.title()} culture accurately."
        )

        return description

    def get_priority_gaps(
        self, issues: List[Dict], country: str, top_k: int = 3
    ) -> List[Dict]:
        """
        Get top priority data gaps.

        Args:
            issues: Detected issues
            country: Target country
            top_k: Number of top gaps to return

        Returns:
            Top priority gaps
        """
        all_gaps = self.analyze(issues, country)
        return all_gaps[:top_k]
