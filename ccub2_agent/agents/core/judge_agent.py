"""
Judge Agent - Cultural quality evaluation and loop decisions.
"""

from typing import Dict, Any
import logging
from pathlib import Path

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...detection.vlm_detector import VLMCulturalDetector

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """
    Re-evaluates outputs and makes loop decisions.
    
    Responsibilities:
    - Score cultural authenticity (1-10)
    - Detect failure modes
    - Decide STOP vs ITERATE
    - Generate feedback for Edit Agent
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.vlm_detector = VLMCulturalDetector(load_in_4bit=True)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Evaluate image cultural quality.
        
        Args:
            input_data: {
                "image_path": str,
                "prompt": str,
                "country": str,
                "category": str (optional)
            }
            
        Returns:
            AgentResult with scores, failure modes, and decision
        """
        try:
            image_path = Path(input_data["image_path"])
            prompt = input_data["prompt"]
            country = input_data.get("country", self.config.country)
            category = input_data.get("category", self.config.category)
            
            # Score cultural quality
            cultural_score, prompt_score = self.vlm_detector.score_cultural_quality(
                image_path=image_path,
                prompt=prompt,
                country=country,
                editing_prompt=None,
                category=category
            )
            
            # Detect issues
            issues = self.vlm_detector.detect(
                image_path=image_path,
                prompt=prompt,
                country=country,
                editing_prompt=None,
                category=category
            )
            
            # Classify failure modes
            failure_modes = self._classify_failure_modes(issues)
            
            # Make decision
            decision = "STOP" if cultural_score >= 8.0 else "ITERATE"
            
            return AgentResult(
                success=True,
                data={
                    "cultural_score": cultural_score,
                    "prompt_score": prompt_score,
                    "issues": issues,
                    "failure_modes": failure_modes,
                    "decision": decision,
                    "confidence": 0.85  # Can be calculated from scores
                },
                message=f"Evaluation complete: {cultural_score:.1f}/10"
            )
            
        except Exception as e:
            logger.error(f"Judge execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Judge error: {str(e)}"
            )
    
    def _classify_failure_modes(self, issues: list) -> list:
        """Classify issues into failure mode categories."""
        failure_modes = []
        for issue in issues:
            if isinstance(issue, dict):
                mode = issue.get("type", "unknown")
            else:
                mode = "unknown"
            failure_modes.append(mode)
        return failure_modes
