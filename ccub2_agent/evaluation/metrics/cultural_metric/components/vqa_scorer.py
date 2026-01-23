"""
VQA-based Cultural Scorer

Uses Vision-Language Model (VLM) for visual question answering
to score cultural authenticity.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VQACulturalScorer:
    """
    VQA-based cultural scoring component.
    
    Uses VLM to answer cultural questions about images.
    """
    
    def __init__(self, vlm_model: str = "Qwen3-VL-8B", load_in_4bit: bool = True):
        """
        Initialize VQA scorer.
        
        Args:
            vlm_model: VLM model name
            load_in_4bit: Whether to load in 4-bit quantization
        """
        self.vlm_model = vlm_model
        self.load_in_4bit = load_in_4bit
        self._vlm_client = None
        
        logger.info(f"VQACulturalScorer initialized: model={vlm_model}")
    
    @property
    def vlm_client(self):
        """Lazy load VLM client."""
        if self._vlm_client is None:
            from ...enhanced_cultural_metric_pipeline import EnhancedVLMClient
            self._vlm_client = EnhancedVLMClient(
                model_name=self.vlm_model,
                load_in_4bit=self.load_in_4bit,
            )
        return self._vlm_client
    
    def score_cultural_authenticity(
        self,
        image_path: Path,
        country: str,
        category: str,
        questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Score cultural authenticity using VQA.
        
        Args:
            image_path: Path to image
            country: Target country
            category: Cultural category
            questions: Optional custom questions. If None, uses default.
            
        Returns:
            Dictionary with scores and answers
        """
        if questions is None:
            questions = self._get_default_questions(country, category)
        
        answers = []
        scores = []
        
        for question in questions:
            answer = self.vlm_client.answer_question(
                image_path=image_path,
                question=question,
            )
            answers.append({
                "question": question,
                "answer": answer,
            })
            
            # Score answer (0-1, higher = more authentic)
            score = self._score_answer(answer, country, category)
            scores.append(score)
        
        # Aggregate scores
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "overall_score": overall_score,
            "dimension_scores": dict(zip([q.split("?")[0] for q in questions], scores)),
            "answers": answers,
            "num_questions": len(questions),
        }
    
    def _get_default_questions(self, country: str, category: str) -> List[str]:
        """
        Get default cultural questions for a country/category.
        
        Args:
            country: Target country
            category: Cultural category
            
        Returns:
            List of questions
        """
        base_questions = [
            f"Is this image culturally authentic for {country}?",
            f"Does this image accurately represent {category} from {country}?",
            f"Are there any cultural inaccuracies in this image?",
            f"Does this image show appropriate cultural elements for {country}?",
        ]
        
        # Add category-specific questions
        if category == "traditional_clothing":
            base_questions.append(f"Does the clothing match traditional {country} style?")
        elif category == "food":
            base_questions.append(f"Is this food presentation typical of {country}?")
        elif category == "architecture":
            base_questions.append(f"Does the architecture reflect {country} cultural style?")
        
        return base_questions
    
    def _score_answer(self, answer: str, country: str, category: str) -> float:
        """
        Score an answer (0-1, higher = more authentic).
        
        Args:
            answer: VLM answer text
            country: Target country
            category: Cultural category
            
        Returns:
            Score (0-1)
        """
        answer_lower = answer.lower()
        
        # Positive indicators
        positive_keywords = ["yes", "authentic", "accurate", "appropriate", "typical", "correct"]
        negative_keywords = ["no", "inaccurate", "inappropriate", "wrong", "incorrect", "not"]
        
        positive_count = sum(1 for kw in positive_keywords if kw in answer_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in answer_lower)
        
        if positive_count > negative_count:
            return 0.8 + (positive_count - negative_count) * 0.05
        elif negative_count > positive_count:
            return 0.2 - (negative_count - positive_count) * 0.05
        else:
            return 0.5  # Neutral


def create_vqa_scorer(
    vlm_model: str = "Qwen3-VL-8B",
    load_in_4bit: bool = True,
) -> VQACulturalScorer:
    """Create a VQA cultural scorer."""
    return VQACulturalScorer(vlm_model=vlm_model, load_in_4bit=load_in_4bit)
