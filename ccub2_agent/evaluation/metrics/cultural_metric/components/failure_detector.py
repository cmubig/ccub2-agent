"""
Failure Mode Detector

Detects and classifies cultural failure modes.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    """Types of cultural failure modes."""
    OVER_MODERNIZATION = "over_modernization"
    STEREOTYPE_RELIANCE = "stereotype_reliance"
    DE_IDENTIFICATION = "de_identification"
    SUPERFICIAL_CUES = "superficial_cues"
    CULTURAL_MIXING = "cultural_mixing"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    GEOGRAPHIC_MISMATCH = "geographic_mismatch"
    UNKNOWN = "unknown"


class FailureModeDetector:
    """
    Detects and classifies cultural failure modes.
    
    Identifies specific types of cultural inaccuracies.
    """
    
    def __init__(self, vlm_model: str = "Qwen3-VL-8B"):
        """
        Initialize failure mode detector.
        
        Args:
            vlm_model: VLM model name
        """
        self.vlm_model = vlm_model
        self._vlm_client = None
        
        logger.info(f"FailureModeDetector initialized: model={vlm_model}")
    
    @property
    def vlm_client(self):
        """Lazy load VLM client."""
        if self._vlm_client is None:
            from ...enhanced_cultural_metric_pipeline import EnhancedVLMClient
            self._vlm_client = EnhancedVLMClient(
                model_name=self.vlm_model,
                load_in_4bit=True,
            )
        return self._vlm_client
    
    def detect_failure_modes(
        self,
        image_path: Path,
        country: str,
        category: str,
        detected_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect failure modes in an image.
        
        Args:
            image_path: Path to image
            country: Target country
            category: Cultural category
            detected_issues: Optional pre-detected issues
            
        Returns:
            List of failure modes with details
        """
        if detected_issues is None:
            # Detect issues first
            detected_issues = self._detect_issues(image_path, country, category)
        
        failure_modes = []
        
        for issue in detected_issues:
            failure_mode = self._classify_failure_mode(issue, country, category)
            
            failure_modes.append({
                "mode": failure_mode.value,
                "confidence": issue.get("confidence", 0.5),
                "description": issue.get("description", ""),
                "location": issue.get("location", ""),
            })
        
        return failure_modes
    
    def _detect_issues(
        self,
        image_path: Path,
        country: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect cultural issues in an image.
        
        Args:
            image_path: Path to image
            country: Target country
            category: Cultural category
            
        Returns:
            List of detected issues
        """
        # Use VLM to detect issues
        questions = [
            f"Are there any cultural inaccuracies in this {category} from {country}?",
            f"Does this image show over-modernized elements?",
            f"Are there stereotypes in this image?",
            f"Is the cultural context appropriate for {country}?",
        ]
        
        issues = []
        
        for question in questions:
            answer = self.vlm_client.answer_question(
                image_path=image_path,
                question=question,
            )
            
            if self._is_issue_detected(answer):
                issues.append({
                    "question": question,
                    "answer": answer,
                    "confidence": 0.7,  # Can be refined
                    "description": answer,
                })
        
        return issues
    
    def _classify_failure_mode(
        self,
        issue: Dict[str, Any],
        country: str,
        category: str,
    ) -> FailureMode:
        """
        Classify a failure mode from an issue.
        
        Args:
            issue: Detected issue
            country: Target country
            category: Cultural category
            
        Returns:
            FailureMode enum
        """
        description = issue.get("description", "").lower()
        question = issue.get("question", "").lower()
        
        # Keyword-based classification
        if "modern" in description or "modern" in question:
            return FailureMode.OVER_MODERNIZATION
        elif "stereotype" in description or "stereotype" in question:
            return FailureMode.STEREOTYPE_RELIANCE
        elif "inappropriate" in description or "wrong" in description:
            return FailureMode.DE_IDENTIFICATION
        elif "superficial" in description or "surface" in description:
            return FailureMode.SUPERFICIAL_CUES
        elif "mix" in description or "combination" in description:
            return FailureMode.CULTURAL_MIXING
        elif "time" in description or "period" in description:
            return FailureMode.TEMPORAL_MISMATCH
        elif "location" in description or "place" in description:
            return FailureMode.GEOGRAPHIC_MISMATCH
        else:
            return FailureMode.UNKNOWN
    
    def _is_issue_detected(self, answer: str) -> bool:
        """
        Check if an issue was detected from VLM answer.
        
        Args:
            answer: VLM answer text
            
        Returns:
            True if issue detected
        """
        answer_lower = answer.lower()
        
        # Negative indicators
        negative_keywords = [
            "yes", "there are", "inaccurate", "wrong", "inappropriate",
            "stereotype", "modern", "incorrect",
        ]
        
        return any(kw in answer_lower for kw in negative_keywords)


def create_failure_detector(
    vlm_model: str = "Qwen3-VL-8B",
) -> FailureModeDetector:
    """Create a failure mode detector."""
    return FailureModeDetector(vlm_model=vlm_model)
