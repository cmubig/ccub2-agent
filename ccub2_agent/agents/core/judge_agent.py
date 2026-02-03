"""
Judge Agent - Cultural quality evaluation and loop decisions.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...detection.vlm_detector import VLMCulturalDetector

logger = logging.getLogger(__name__)

# Fix 4: Score momentum parameters for stabilization
MOMENTUM_WEIGHT = 0.7  # Weight for previous score (0.7 * prev + 0.3 * new)


class JudgeAgent(BaseAgent):
    """
    Re-evaluates outputs and makes loop decisions.

    Responsibilities:
    - Score cultural authenticity (1-10)
    - Detect failure modes
    - Decide STOP vs ITERATE
    - Generate feedback for Edit Agent
    """

    def __init__(self, config: AgentConfig, shared_vlm_detector: "VLMCulturalDetector | None" = None):
        super().__init__(config)
        if shared_vlm_detector is not None:
            self.vlm_detector = shared_vlm_detector
        else:
            self.vlm_detector = VLMCulturalDetector(load_in_4bit=True)
        # Fix 4: Track previous score for momentum
        self._previous_cultural_score: Optional[float] = None
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Evaluate image cultural quality.

        Args:
            input_data: {
                "image_path": str,
                "prompt": str,
                "country": str,
                "category": str (optional),
                "reset_momentum": bool (optional) - set True for initial evaluation
            }

        Returns:
            AgentResult with scores, failure modes, and decision
        """
        try:
            image_path = Path(input_data["image_path"])
            prompt = input_data["prompt"]
            country = input_data.get("country", self.config.country)
            category = input_data.get("category", self.config.category)
            reset_momentum = input_data.get("reset_momentum", False)

            # Reset momentum for initial evaluation
            if reset_momentum:
                self._previous_cultural_score = None

            # Score cultural quality
            raw_cultural_score, prompt_score = self.vlm_detector.score_cultural_quality(
                image_path=image_path,
                prompt=prompt,
                country=country,
                editing_prompt=None,
                category=category
            )

            # Fix 4: Apply score momentum to stabilize scoring
            if self._previous_cultural_score is not None:
                cultural_score = MOMENTUM_WEIGHT * self._previous_cultural_score + (1 - MOMENTUM_WEIGHT) * raw_cultural_score
                logger.debug(f"Score momentum applied: {self._previous_cultural_score:.1f} * {MOMENTUM_WEIGHT} + {raw_cultural_score:.1f} * {1-MOMENTUM_WEIGHT} = {cultural_score:.1f}")
            else:
                cultural_score = raw_cultural_score

            # Update previous score for next iteration
            self._previous_cultural_score = cultural_score

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
                    "raw_cultural_score": raw_cultural_score,  # Also return raw for debugging
                    "prompt_score": prompt_score,
                    "issues": issues,
                    "failure_modes": failure_modes,
                    "decision": decision,
                    "confidence": 0.85  # Can be calculated from scores
                },
                message=f"Evaluation complete: {cultural_score:.1f}/10 (raw: {raw_cultural_score:.1f})"
            )

        except Exception as e:
            logger.error(f"Judge execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Judge error: {str(e)}"
            )

    def reset_momentum(self):
        """Reset the score momentum for a new image evaluation sequence."""
        self._previous_cultural_score = None
    
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
