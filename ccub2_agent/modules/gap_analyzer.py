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

                # CHANGED: Skip job creation if gap has skip_job_creation flag
                if gap.get("skip_job_creation", False):
                    logger.info(
                        f"Skipping job creation for {category} (coverage: {coverage_ratio:.1%}) "
                        f"- no meaningful issues found"
                    )
                    continue

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

        # CHANGED: If description is None, meaningful issues were not found
        # Return gap spec with skip_job_creation flag
        if description is None:
            return {
                "category": category,
                "subcategory": subcategory,
                "keywords": [],
                "description": f"Insufficient specific information for {country} {category}",
                "severity": 0,
                "priority": 0,
                "issue_count": len(issues),
                "skip_job_creation": True,  # Signal to skip job creation
            }

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
            "skip_job_creation": False,  # Explicit: this is a valid job
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

                # Keep meaningful words (length > 2, not common English words or evaluation terms)
                # Extended stopwords to filter out generic scoring/evaluation phrases
                if len(word) > 2 and word not in {
                    # Common words
                    "the", "and", "for", "with", "not", "this", "that", "these", "those",
                    "are", "was", "were", "been", "being", "but", "also", "can",
                    # Modals & auxiliaries
                    "should", "must", "need", "needs", "have", "has", "had", "will", "would", "could",
                    # Qualifiers & frequency terms
                    "from", "more", "less", "very", "much", "such", "about", "into", "through",
                    "often", "sometimes", "always", "never", "seen", "common", "typical", "usual",
                    # Problem descriptors (keep these generic)
                    "incorrect", "wrong", "missing", "lacking",
                    # Evaluation terms (CRITICAL: filter these out!)
                    "improvement", "improve", "improved", "improving",
                    "accuracy", "accurate", "inaccurate",
                    "quality", "score", "scoring", "evaluation",
                    "representation", "representative", "represent",
                    "authentic", "authenticity",
                    # Generic cultural terms (too broad to be useful keywords)
                    "traditional", "cultural", "culture",
                    # Style descriptors (too vague)
                    "stylistically", "stylistic", "style", "styled",
                    "simple", "complex", "basic", "advanced",
                    "inspired", "based", "derived", "similar",
                    # Visual/aesthetic terms (NOT useful for data collection)
                    "aesthetics", "aesthetic", "palette", "color", "colors", "colour", "colours",
                    "fabric", "fabrics", "tones", "tone", "ornate", "ornament", "ornamental",
                    "neckline", "necklines", "silhouette", "silhouettes",
                    "pattern", "patterns", "texture", "textures",
                    "design", "designs", "element", "elements",
                    "detail", "details", "detailed", "detailing",
                    # Generic clothing terms (need specific garment names instead)
                    "clothing", "clothes", "garment", "garments", "attire", "outfit", "outfits",
                    "wear", "wearing", "worn", "dress", "dresses",
                    # Generic body parts/positions
                    "collar", "collars", "sleeve", "sleeves", "hem", "waist", "shoulder",
                    # Color names (too generic alone)
                    "white", "black", "red", "blue", "green", "yellow", "pink", "purple",
                    "gold", "golden", "silver", "navy", "brown", "beige", "gray", "grey",
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
        Extracts meaningful terms dynamically from VLM output.

        Strategy:
        1. Look for quoted terms in VLM output (e.g., "hanbok", "jeogori")
        2. Look for capitalized proper nouns (culture-specific terms)
        3. Filter out generic visual/aesthetic terms
        4. Return most specific remaining term
        """
        import re

        # STEP 1: Extract quoted terms from VLM output (highest priority)
        # VLM often puts culture-specific terms in quotes: "hanbok", "jeogori"
        quoted_terms = []
        for issue in issues:
            desc = issue.get("description", "")
            # Find terms in quotes (single or double)
            matches = re.findall(r'["\']([a-zA-Z][a-zA-Z\s-]{2,20})["\']', desc)
            quoted_terms.extend([m.lower().strip() for m in matches])

        # STEP 2: Extract capitalized proper nouns (culture-specific terms)
        # Words that are capitalized mid-sentence are likely proper nouns
        proper_nouns = []
        for issue in issues:
            desc = issue.get("description", "")
            # Split into sentences, then find capitalized words not at start
            sentences = re.split(r'[.!?]', desc)
            for sentence in sentences:
                words = sentence.split()
                for i, word in enumerate(words):
                    # Skip first word of sentence
                    if i == 0:
                        continue
                    # Check if word starts with capital (proper noun indicator)
                    clean_word = word.strip(".,!?()[]{}\"':;")
                    if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                        # Skip common English capitalized words
                        if clean_word.lower() not in {"the", "this", "that", "korean", "japanese", "chinese", "indian", "thai", "vietnamese", country.lower()}:
                            proper_nouns.append(clean_word.lower())

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

        # Generic terms to filter out (NOT meaningful subcategories)
        generic_terms = {
            # Cultural/quality terms
            "traditional", "authentic", "cultural", "correct", "proper",
            "accuracy", "quality", "representation",
            # Modal verbs & common verbs
            "should", "must", "need", "needs", "have", "has", "show", "display",
            # Problem descriptors
            "incorrect", "wrong", "missing", "lacking",
            # Frequency/visibility terms
            "seen", "often", "sometimes", "always", "common", "typical",
            # Style descriptors
            "stylistically", "stylistic", "style", "simple", "complex",
            "inspired", "based", "derived", "-inspired",  # e.g., "qipao-inspired"
            # Context placeholders
            country.lower(),
            category.replace("_", " ").lower(),
            category.split("_")[0] if "_" in category else "",  # e.g., "traditional" from "traditional_clothing"
            # Visual/aesthetic terms (NOT meaningful subcategories!)
            "aesthetics", "aesthetic", "palette", "color", "colors", "colour", "colours",
            "fabric", "fabrics", "tones", "tone", "ornate", "ornament", "ornamental",
            "neckline", "necklines", "silhouette", "silhouettes",
            "pattern", "patterns", "texture", "textures",
            "design", "designs", "element", "elements",
            "detail", "details", "detailed", "detailing",
            # Generic clothing terms
            "clothing", "clothes", "garment", "garments", "attire", "outfit", "outfits",
            "wear", "wearing", "worn",
            # Generic body parts
            "collar", "collars", "sleeve", "sleeves", "hem", "waist", "shoulder",
            # Color names (too generic alone)
            "white", "red", "blue", "black", "green", "yellow", "pink", "purple",
            "gold", "golden", "silver", "navy", "brown", "beige", "gray", "grey",
            # Too generic for clothing categories
            "skirt", "dress", "cloth",
        }

        # Filter out generic terms
        specific_keywords = [
            k for k in keywords
            if k.lower() not in generic_terms
            and len(k) > 2
            and not any(gen in k.lower() for gen in ["-inspired", "style"])  # Filter compound terms
        ]

        # PRIORITY 1: Use quoted terms from VLM (most reliable)
        # Filter quoted terms against generic_terms too
        # ALSO filter out terms from OTHER cultures (e.g., "kimono" when country is Korea)
        other_culture_terms = {
            "kimono", "yukata", "hakama", "obi",  # Japanese
            "hanfu", "qipao", "cheongsam",  # Chinese
            "sari", "saree", "kurta",  # Indian
            "ao dai", "aodai",  # Vietnamese
        }
        # Determine which terms belong to other cultures based on country
        # Also filter out generic "X_style" terms that describe what's WRONG, not what we need
        country_lower = country.lower()
        if country_lower == "korea":
            wrong_culture_terms = {
                # Japanese items
                "kimono", "yukata", "hakama", "obi", "japanese", "japanese_style", "japan",
                # Chinese items
                "hanfu", "qipao", "cheongsam", "chinese", "chinese_style", "china",
                # Other cultures
                "sari", "ao dai", "indian", "vietnamese"
            }
        elif country_lower == "japan":
            wrong_culture_terms = {
                "hanbok", "jeogori", "chima", "korean", "korean_style", "korea",
                "hanfu", "qipao", "chinese", "chinese_style",
                "sari", "ao dai"
            }
        elif country_lower == "china":
            wrong_culture_terms = {
                "hanbok", "jeogori", "korean", "korean_style",
                "kimono", "yukata", "japanese", "japanese_style",
                "sari", "ao dai"
            }
        else:
            wrong_culture_terms = set()

        valid_quoted = [
            t for t in quoted_terms
            if t not in generic_terms
            and t not in wrong_culture_terms  # Don't use wrong culture terms as subcategory!
            and len(t) > 2
        ]
        if valid_quoted:
            # Return first valid quoted term
            subcategory = valid_quoted[0].replace(" ", "_").replace("-", "_")
            logger.debug(f"Subcategory from quoted term: {subcategory}")
            return subcategory

        # PRIORITY 2: Use proper nouns (culture-specific terms)
        valid_proper = [
            t for t in proper_nouns
            if t not in generic_terms
            and t not in wrong_culture_terms  # Don't use wrong culture terms!
            and len(t) > 2
        ]
        if valid_proper:
            subcategory = valid_proper[0].replace(" ", "_").replace("-", "_")
            logger.debug(f"Subcategory from proper noun: {subcategory}")
            return subcategory

        # PRIORITY 3: Use filtered specific keywords
        if len(specific_keywords) >= 2:
            # Take top 2 most specific keywords
            subcategory = "_".join(specific_keywords[:2]).lower()
            # Clean up
            subcategory = subcategory.replace(" ", "_").replace("-", "_")
            return subcategory
        elif len(specific_keywords) == 1:
            # Single specific keyword
            return specific_keywords[0].lower().replace(" ", "_")

        # Default to general (will be filtered out by job quality validation)
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

        # CHANGED: If no meaningful issues found, return None instead of using fallback
        # This prevents creating generic, low-quality jobs
        if not meaningful_issues:
            logger.warning(
                f"No meaningful issues found for {country} {category} - "
                f"skipping job creation (only generic scoring messages present)"
            )
            return None  # Signal to skip job creation

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

        # Note: No fallback message needed - if no meaningful_issues, we return None above

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
