"""
Reference Image Selector

Selects the most representative reference image for editing.
Uses CLIP similarity + metadata to find best match.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from PIL import Image

logger = logging.getLogger(__name__)


class ReferenceImageSelector:
    """
    Selects best reference image for editing.

    Considers:
    - CLIP similarity
    - Category match
    - Description relevance
    - Image quality metadata
    """

    def __init__(self, clip_rag, quality_weight: float = 0.2):
        """
        Initialize selector.

        Args:
            clip_rag: CLIP Image RAG instance
            quality_weight: Weight for quality score (0-1)
        """
        self.clip_rag = clip_rag
        self.quality_weight = quality_weight

    def select_best_reference(
        self,
        query_image: Path,
        issues: List[Dict[str, Any]],
        category: Optional[str] = None,
        k: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Select single best reference image.

        Args:
            query_image: Problem image
            issues: List of detected issues
            category: Image category
            k: Number of candidates to consider

        Returns:
            Dict with:
                - image_path: Path to reference image
                - similarity: Similarity score
                - reason: Why this was selected
                - metadata: Full metadata
            Or None if no good reference found
        """
        # Retrieve candidate references
        candidates = self.clip_rag.retrieve_similar_images(
            image_path=query_image,
            k=k,
            category=category,
        )

        if not candidates:
            logger.warning("No reference images found")
            return None

        # Score each candidate
        scored = []
        for cand in candidates:
            score = self._score_candidate(cand, issues)
            scored.append({
                'image_path': cand['image_path'],
                'similarity': cand['similarity'],
                'total_score': score,
                'metadata': cand['metadata'],
            })

        # Sort by total score
        scored.sort(key=lambda x: x['total_score'], reverse=True)

        best = scored[0]

        # Generate selection reason
        reason = self._generate_reason(best, issues)
        best['reason'] = reason

        logger.info(f"Selected reference: {Path(best['image_path']).name} (score: {best['total_score']:.3f})")
        logger.info(f"Reason: {reason}")

        return best

    def _score_candidate(self, candidate: Dict[str, Any], issues: List[Dict[str, Any]]) -> float:
        """
        Score a candidate reference image.

        Combines:
        - CLIP similarity (main factor)
        - Quality score if available
        - Description match with issues
        """
        score = candidate['similarity']  # Base score from CLIP

        # Boost if has quality score
        metadata = candidate.get('metadata', {})
        quality = metadata.get('quality_score', 0)
        if quality > 0:
            # Normalize quality (typically 0-100)
            quality_norm = quality / 100.0
            score = score * (1 - self.quality_weight) + quality_norm * self.quality_weight

        # Boost if description matches issues
        desc = metadata.get('description_enhanced') or metadata.get('description', '')
        if desc and issues:
            issue_keywords = set()
            for issue in issues:
                issue_keywords.update(issue.get('description', '').lower().split())

            desc_lower = desc.lower()
            matches = sum(1 for kw in issue_keywords if kw in desc_lower)
            if matches > 0:
                # Small boost for keyword matches
                score += 0.05 * min(matches, 3)  # Max +0.15

        return score

    def _generate_reason(self, best: Dict[str, Any], issues: List[Dict[str, Any]]) -> str:
        """Generate human-readable reason for selection."""
        sim = best['similarity']
        metadata = best['metadata']

        reason = f"High similarity ({sim:.1%})"

        if metadata.get('description_enhanced'):
            reason += ", has detailed cultural description"

        quality = metadata.get('quality_score', 0)
        if quality > 80:
            reason += ", high quality"

        category = metadata.get('category', '')
        if category:
            reason += f", from {category} category"

        return reason

    def select_multiple_references(
        self,
        query_image: Path,
        issues: List[Dict[str, Any]],
        category: Optional[str] = None,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Select multiple reference images for diversity.

        Args:
            query_image: Problem image
            issues: Detected issues
            category: Category filter
            k: Number of references to select

        Returns:
            List of reference dicts
        """
        candidates = self.clip_rag.retrieve_similar_images(
            image_path=query_image,
            k=k * 2,  # Get more to filter
            category=category,
        )

        if not candidates:
            return []

        # Score candidates
        scored = []
        for cand in candidates:
            score = self._score_candidate(cand, issues)
            scored.append({
                'image_path': cand['image_path'],
                'similarity': cand['similarity'],
                'total_score': score,
                'metadata': cand['metadata'],
            })

        # Sort and take top k
        scored.sort(key=lambda x: x['total_score'], reverse=True)
        selected = scored[:k]

        # Add reasons
        for ref in selected:
            ref['reason'] = self._generate_reason(ref, issues)

        return selected


def create_reference_selector(clip_rag, quality_weight: float = 0.2) -> ReferenceImageSelector:
    """Factory function to create reference selector."""
    return ReferenceImageSelector(clip_rag=clip_rag, quality_weight=quality_weight)
