"""
Data Gap Detector

Detects when there is insufficient data to fix cultural issues.
Triggers data collection jobs when gaps are found.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataGapDetector:
    """
    Detects gaps in cultural reference data.

    Analyzes:
    - CLIP similarity scores
    - Number of reference images
    - Category coverage
    - VLM confidence
    """

    def __init__(
        self,
        clip_rag=None,
        text_kb=None,
        min_similar_images: int = 5,
        min_similarity_score: float = 0.6,
        min_text_docs: int = 3,
    ):
        """
        Initialize Data Gap Detector.

        Args:
            clip_rag: CLIP Image RAG instance
            text_kb: Text knowledge base instance
            min_similar_images: Minimum number of similar images needed
            min_similarity_score: Minimum average similarity score
            min_text_docs: Minimum number of text documents needed
        """
        self.clip_rag = clip_rag
        self.text_kb = text_kb
        self.min_similar_images = min_similar_images
        self.min_similarity_score = min_similarity_score
        self.min_text_docs = min_text_docs

    def detect_gap(
        self,
        image_path: Path,
        prompt: str,
        issues: List[Dict[str, Any]],
        category: Optional[str] = None,
        country: str = None,
    ) -> Dict[str, Any]:
        """
        Detect if there's a data gap for fixing the given issues.

        Args:
            image_path: Path to problematic image
            prompt: Original generation prompt
            issues: List of detected issues
            category: Image category
            country: Target country

        Returns:
            Gap analysis dict with:
                - has_gap: bool
                - gap_type: 'image' | 'text' | 'both' | None
                - missing_concepts: List[str]
                - similar_image_count: int
                - avg_similarity: float
                - text_doc_count: int
                - confidence: float (0-1)
                - recommendation: str
        """
        gap_info = {
            'has_gap': False,
            'gap_type': None,
            'missing_concepts': [],
            'similar_image_count': 0,
            'avg_similarity': 0.0,
            'text_doc_count': 0,
            'confidence': 1.0,
            'recommendation': '',
            'category': category,
            'country': country,
        }

        # Check image data gap
        if self.clip_rag:
            image_gap = self._check_image_gap(image_path, category)
            gap_info.update(image_gap)

        # Check text data gap
        if self.text_kb:
            text_gap = self._check_text_gap(prompt, issues, category)
            gap_info.update(text_gap)

        # Determine gap type
        image_insufficient = gap_info['similar_image_count'] < self.min_similar_images
        image_low_quality = gap_info['avg_similarity'] < self.min_similarity_score
        text_insufficient = gap_info['text_doc_count'] < self.min_text_docs

        if image_insufficient or image_low_quality:
            if text_insufficient:
                gap_info['gap_type'] = 'both'
            else:
                gap_info['gap_type'] = 'image'
        elif text_insufficient:
            gap_info['gap_type'] = 'text'

        gap_info['has_gap'] = gap_info['gap_type'] is not None

        # Calculate confidence
        gap_info['confidence'] = self._calculate_confidence(gap_info)

        # Generate recommendation
        gap_info['recommendation'] = self._generate_recommendation(gap_info, issues)

        logger.info(f"Gap detection: has_gap={gap_info['has_gap']}, "
                   f"type={gap_info['gap_type']}, "
                   f"confidence={gap_info['confidence']:.2f}")

        return gap_info

    def _check_image_gap(self, image_path: Path, category: Optional[str]) -> Dict[str, Any]:
        """Check if there are enough similar reference images."""
        try:
            # Retrieve similar images
            similar = self.clip_rag.retrieve_similar_images(
                image_path=image_path,
                k=10,
                category=category,
            )

            if not similar:
                return {
                    'similar_image_count': 0,
                    'avg_similarity': 0.0,
                }

            # Calculate average similarity
            avg_sim = sum(r['similarity'] for r in similar) / len(similar)

            return {
                'similar_image_count': len(similar),
                'avg_similarity': avg_sim,
            }

        except Exception as e:
            logger.error(f"Image gap check failed: {e}")
            return {
                'similar_image_count': 0,
                'avg_similarity': 0.0,
            }

    def _check_text_gap(
        self,
        prompt: str,
        issues: List[Dict[str, Any]],
        category: Optional[str]
    ) -> Dict[str, Any]:
        """Check if there are enough text documents."""
        try:
            # Build query from prompt and issues
            queries = [prompt]
            for issue in issues:
                queries.append(issue.get('description', ''))

            query = ' '.join(queries)

            # Retrieve documents
            # This requires sample object, simplified here
            # In practice, use proper sample
            docs = []  # self.text_kb.retrieve(query, top_k=10)

            return {
                'text_doc_count': len(docs),
            }

        except Exception as e:
            logger.error(f"Text gap check failed: {e}")
            return {
                'text_doc_count': 0,
            }

    def _calculate_confidence(self, gap_info: Dict[str, Any]) -> float:
        """
        Calculate confidence score for gap detection.

        Higher confidence = more certain there's a gap
        """
        confidence = 1.0

        # Reduce confidence based on data availability
        if gap_info['similar_image_count'] > 0:
            # More images = less confident about gap
            confidence *= (1.0 - min(gap_info['similar_image_count'] / 10, 0.5))

        if gap_info['avg_similarity'] > 0:
            # Higher similarity = less confident about gap
            confidence *= (1.0 - gap_info['avg_similarity'] * 0.5)

        return max(0.0, min(1.0, confidence))

    def _generate_recommendation(
        self,
        gap_info: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> str:
        """Generate recommendation for data collection."""
        if not gap_info['has_gap']:
            return "Sufficient data available for fixing issues"

        category = gap_info['category'] or 'general'
        country = gap_info['country']

        # Extract missing concepts from issues
        missing_concepts = []
        for issue in issues:
            desc = issue.get('description', '')
            cat = issue.get('category', '')
            if cat:
                missing_concepts.append(cat)

        gap_info['missing_concepts'] = list(set(missing_concepts))

        rec = f"Need more {country} {category} data. "

        if gap_info['gap_type'] == 'image':
            rec += f"Only {gap_info['similar_image_count']} similar images found "
            rec += f"(avg similarity: {gap_info['avg_similarity']:.2%}). "
            rec += f"Recommend collecting {self.min_similar_images - gap_info['similar_image_count']} more images"
        elif gap_info['gap_type'] == 'text':
            rec += f"Only {gap_info['text_doc_count']} text documents found. "
            rec += f"Recommend collecting {self.min_text_docs - gap_info['text_doc_count']} more documents"
        elif gap_info['gap_type'] == 'both':
            rec += "Insufficient both image and text data. "
            rec += f"Need {self.min_similar_images - gap_info['similar_image_count']} images and "
            rec += f"{self.min_text_docs - gap_info['text_doc_count']} documents"

        if missing_concepts:
            rec += f" focusing on: {', '.join(missing_concepts)}"

        return rec


def create_data_gap_detector(
    clip_rag=None,
    text_kb=None,
    min_similar_images: int = 5,
    min_similarity_score: float = 0.6,
) -> DataGapDetector:
    """Factory function to create Data Gap Detector."""
    return DataGapDetector(
        clip_rag=clip_rag,
        text_kb=text_kb,
        min_similar_images=min_similar_images,
        min_similarity_score=min_similarity_score,
    )
