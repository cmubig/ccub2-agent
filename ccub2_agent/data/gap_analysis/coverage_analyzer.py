"""
Coverage Analyzer

Analyzes data coverage gaps for countries and categories.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """Coverage analysis report."""
    country: str
    category: str
    total_images: int
    coverage_score: float  # 0-1
    missing_elements: List[str]
    priority_gaps: List[str]
    recommendations: List[str]


class CoverageAnalyzer:
    """
    Analyzes data coverage for countries and categories.
    
    Identifies gaps and prioritizes data collection needs.
    """
    
    def __init__(self, country: str, category: Optional[str] = None):
        """
        Initialize coverage analyzer.
        
        Args:
            country: Target country
            category: Optional category filter
        """
        self.country = country
        self.category = category
        
        logger.info(f"CoverageAnalyzer initialized: country={country}, category={category}")
    
    def analyze_coverage(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
    ) -> CoverageReport:
        """
        Analyze coverage for a country/category.
        
        Args:
            country: Target country (uses self.country if None)
            category: Target category (uses self.category if None)
            
        Returns:
            CoverageReport
        """
        country = country or self.country
        category = category or self.category
        
        # Load data
        from ..country_pack import CountryDataPack
        country_pack = CountryDataPack(country)
        
        # Count images
        total_images = self._count_images(country_pack, category)
        
        # Analyze gaps
        missing_elements = self._identify_missing_elements(country_pack, category)
        priority_gaps = self._prioritize_gaps(missing_elements)
        
        # Compute coverage score
        coverage_score = self._compute_coverage_score(total_images, missing_elements)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_images,
            missing_elements,
            priority_gaps,
        )
        
        return CoverageReport(
            country=country,
            category=category or "all",
            total_images=total_images,
            coverage_score=coverage_score,
            missing_elements=missing_elements,
            priority_gaps=priority_gaps,
            recommendations=recommendations,
        )
    
    def _count_images(
        self,
        country_pack: Any,
        category: Optional[str],
    ) -> int:
        """Count total images in country pack."""
        try:
            dataset = country_pack.get_dataset()
            if category:
                return len([item for item in dataset if item.get("category") == category])
            return len(dataset)
        except Exception as e:
            logger.warning(f"Error counting images: {e}")
            return 0
    
    def _identify_missing_elements(
        self,
        country_pack: Any,
        category: Optional[str],
    ) -> List[str]:
        """Identify missing cultural elements."""
        # This would analyze the dataset and identify gaps
        # For now, return placeholder
        return []
    
    def _prioritize_gaps(self, gaps: List[str]) -> List[str]:
        """Prioritize gaps by importance."""
        # Simple prioritization: return first 3 gaps
        return gaps[:3]
    
    def _compute_coverage_score(
        self,
        total_images: int,
        missing_elements: List[str],
    ) -> float:
        """
        Compute coverage score (0-1).
        
        Args:
            total_images: Total number of images
            missing_elements: List of missing elements
            
        Returns:
            Coverage score (0-1)
        """
        # Simple scoring: based on image count and gaps
        base_score = min(1.0, total_images / 100.0)  # 100 images = full coverage
        gap_penalty = len(missing_elements) * 0.1
        
        return max(0.0, base_score - gap_penalty)
    
    def _generate_recommendations(
        self,
        total_images: int,
        missing_elements: List[str],
        priority_gaps: List[str],
    ) -> List[str]:
        """Generate recommendations for data collection."""
        recommendations = []
        
        if total_images < 50:
            recommendations.append(f"Increase image count to at least 50 (currently {total_images})")
        
        if priority_gaps:
            recommendations.append(f"Priority: Collect data for {', '.join(priority_gaps)}")
        
        if not recommendations:
            recommendations.append("Coverage is adequate")
        
        return recommendations


def analyze_coverage(
    country: str,
    category: Optional[str] = None,
) -> CoverageReport:
    """
    Convenience function to analyze coverage.
    
    Args:
        country: Target country
        category: Optional category
        
    Returns:
        CoverageReport
    """
    analyzer = CoverageAnalyzer(country=country, category=category)
    return analyzer.analyze_coverage()
