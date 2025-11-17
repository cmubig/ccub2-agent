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
            Gap specification dict with subcategory inference
        """
        # Extract keywords from issue descriptions
        keywords = self._extract_keywords(issues, category, country)

        # Infer subcategory from issues and keywords
        subcategory = self._infer_subcategory(issues, category, country, keywords)

        # Calculate average severity
        avg_severity = sum(i.get("severity", 5) for i in issues) / len(issues)

        # Generate description
        description = self._generate_description(category, issues, country, subcategory)

        # Calculate priority (based on severity and frequency)
        priority = min(10, int(avg_severity * len(issues) / 2))

        return {
            "category": category,
            "subcategory": subcategory,
            "keywords": keywords,
            "description": description,
            "severity": avg_severity,
            "priority": priority,
            "issue_count": len(issues),
        }

    def _extract_keywords(
        self, issues: List[Dict], category: str, country: str
    ) -> List[str]:
        """
        Extract relevant keywords for data collection.

        FULLY DYNAMIC - Works for ANY country without hardcoding!
        Extracts keywords directly from issue descriptions.
        """
        keywords = set()

        # Extract keywords from issue descriptions
        for issue in issues:
            desc = issue.get("description", "").lower()

            # Split and clean words
            words = desc.split()
            for word in words:
                # Clean punctuation
                word = word.strip(".,!?()[]{}\"':;")

                # Keep meaningful words (length > 2, not common English words)
                if len(word) > 2 and word not in {
                    "the", "and", "for", "with", "not", "this", "that",
                    "should", "must", "need", "have", "from", "more", "less",
                    "very", "much", "such", "about", "into", "through",
                    "incorrect", "wrong", "missing", "lacking"
                }:
                    keywords.add(word)

        # Add generic keywords
        keywords.add(f"{country}")
        keywords.add(f"{category.replace('_', ' ')}")

        # Convert to list and return top keywords
        return list(keywords)[:10]  # Limit to top 10 most relevant

    def _infer_subcategory(
        self,
        issues: List[Dict],
        category: str,
        country: str,
        keywords: List[str]
    ) -> str:
        """
        Infer specific subcategory from issues and keywords.

        FULLY DYNAMIC - Works for ANY country without hardcoding!
        Uses keyword frequency and co-occurrence patterns.

        Examples:
        - keywords: ["jeogori", "collar", "neckline"] → "jeogori_collar"
        - keywords: ["kimono", "obi", "belt"] → "kimono_obi"
        - keywords: ["sari", "draping", "style"] → "sari_draping"
        """
        # Combine all keywords
        all_keywords = set(keywords)

        # Add keywords from issue descriptions
        for issue in issues:
            desc = issue.get("description", "").lower()
            # Extract potential subcategory indicators
            words = desc.split()
            for word in words:
                word = word.strip(".,!?()")
                if len(word) > 3:
                    all_keywords.add(word)

        all_keywords_lower = [k.lower() for k in all_keywords]

        # Generic subcategory inference based on keywords
        # Find most specific keywords (not generic terms)
        generic_terms = {
            "traditional", "authentic", "cultural", "correct", "proper",
            "should", "must", "need", "have", "show", "display",
            "incorrect", "wrong", "missing", "lacking",
            country.lower(), category.replace("_", " ").lower()
        }

        # Filter out generic terms
        specific_keywords = [
            k for k in keywords
            if k.lower() not in generic_terms and len(k) > 2
        ]

        # If we have specific keywords, combine them into subcategory
        if len(specific_keywords) >= 2:
            # Take top 2 most specific keywords
            subcategory = "_".join(specific_keywords[:2]).lower()
            # Clean up
            subcategory = subcategory.replace(" ", "_").replace("-", "_")
            return subcategory
        elif len(specific_keywords) == 1:
            # Single specific keyword
            return specific_keywords[0].lower().replace(" ", "_")

        # Default to general
        return "general"

    def _generate_description(
        self, category: str, issues: List[Dict], country: str, subcategory: str = "general"
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

        # Customize description based on subcategory
        if subcategory and subcategory != "general":
            subcategory_display = subcategory.replace("_", " ")
            description = (
                f"We need more authentic examples of **{country.title()} {subcategory_display}** "
                f"(in {category_name}) to improve cultural accuracy in AI-generated images.\n\n"
            )
        else:
            description = (
                f"We need more authentic examples of {country.title()} {category_name} "
                f"to improve cultural accuracy in image generation.\n\n"
            )

        # Add specific issues (filter out generic scoring messages)
        description += "**Specifically, we need help with:**\n"

        # Filter for meaningful issues (exclude generic scoring messages)
        meaningful_issues = []
        for issue in issues:
            desc = issue.get('description', '')

            # Skip generic scoring messages
            if any(skip_phrase in desc.lower() for skip_phrase in [
                'needs improvement (',
                'score:',
                '/10',
                'cultural accuracy needs',
                'prompt alignment needs'
            ]):
                continue

            # Prefer detailed issues
            if issue.get('is_detailed') or len(desc) > 50:
                meaningful_issues.append(issue)

        # If no meaningful issues found, take first non-scoring issue
        if not meaningful_issues:
            meaningful_issues = [i for i in issues if len(i.get('description', '')) > 30][:1]

        # Add meaningful issues to description
        for i, issue in enumerate(meaningful_issues[:3], 1):  # Top 3 meaningful issues
            desc = issue['description']

            # Truncate very long descriptions and clean up
            if len(desc) > 200:
                # Take first sentence or first 200 chars
                first_sentence = desc.split('.')[0] if '.' in desc[:200] else desc[:200]
                desc = first_sentence.strip() + '...'

            # Remove "SPECIFIC problems:" prefix if present
            desc = desc.replace('SPECIFIC problems:', '').strip()

            # Clean up numbered lists (remove leading numbers)
            if desc.startswith(('1.', '2.', '3.', '4.', '5.')):
                desc = desc[2:].strip()

            description += f"{i}. {desc}\n"

        # If no issues were added, add a generic message
        if not meaningful_issues:
            description += "General improvements to cultural authenticity and accuracy.\n"

        description += (
            f"\n**Why this matters:**\n"
            f"Your contributions will help AI systems better understand and "
            f"represent {country.title()} culture accurately, reducing stereotypes "
            f"and improving cultural representation in AI-generated content."
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
