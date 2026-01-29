"""Hierarchical Failure Mode Detector (C4).

Detects and classifies cultural failure modes using a structured taxonomy
with severity levels and VLM probe-based detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..taxonomy import CulturalDimension, FailurePenalty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy enum (kept for backward compatibility)
# ---------------------------------------------------------------------------

class FailureMode(str, Enum):
    """Types of cultural failure modes (legacy)."""
    OVER_MODERNIZATION = "over_modernization"
    STEREOTYPE_RELIANCE = "stereotype_reliance"
    DE_IDENTIFICATION = "de_identification"
    SUPERFICIAL_CUES = "superficial_cues"
    CULTURAL_MIXING = "cultural_mixing"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    GEOGRAPHIC_MISMATCH = "geographic_mismatch"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# C4. Hierarchical Failure Taxonomy
# ---------------------------------------------------------------------------

class FailureSeverity(float, Enum):
    """Severity levels for failure modes — used as penalty multipliers."""
    CRITICAL = 0.8
    MAJOR = 0.5
    MINOR = 0.2
    COSMETIC = 0.05


class FailureCategory(str, Enum):
    """Top-level failure categories."""
    IDENTITY = "identity"
    AUTHENTICITY = "authenticity"
    SENSITIVITY = "sensitivity"
    COHERENCE = "coherence"


@dataclass
class EnhancedFailureMode:
    """A single detected failure mode with full metadata."""
    category: FailureCategory
    mode: str
    severity: FailureSeverity
    confidence: float  # 0-1, from VLM probe
    dimension_affected: Optional[CulturalDimension] = None
    evidence: str = ""


# ---------------------------------------------------------------------------
# Failure taxonomy definition: 4 categories x 10 modes
# ---------------------------------------------------------------------------

@dataclass
class _FailureSpec:
    """Specification for a single failure mode in the taxonomy."""
    mode: str
    severity: FailureSeverity
    dimension: CulturalDimension
    vlm_probe: str  # VLM question template


FAILURE_TAXONOMY: Dict[FailureCategory, List[_FailureSpec]] = {
    FailureCategory.IDENTITY: [
        _FailureSpec(
            mode="cultural_misattribution",
            severity=FailureSeverity.CRITICAL,
            dimension=CulturalDimension.CONTEXTUAL_COHERENCE,
            vlm_probe="Are there elements from a different culture incorrectly mixed into this {country} {category}?",
        ),
        _FailureSpec(
            mode="geographic_displacement",
            severity=FailureSeverity.CRITICAL,
            dimension=CulturalDimension.CONTEXTUAL_COHERENCE,
            vlm_probe="Does the geographic background or setting look inconsistent with {country}?",
        ),
    ],
    FailureCategory.AUTHENTICITY: [
        _FailureSpec(
            mode="material_anachronism",
            severity=FailureSeverity.MAJOR,
            dimension=CulturalDimension.MATERIAL_AUTHENTICITY,
            vlm_probe="Do the materials or textures look inauthentic or artificial for traditional {country} {category}?",
        ),
        _FailureSpec(
            mode="symbolic_error",
            severity=FailureSeverity.MAJOR,
            dimension=CulturalDimension.SYMBOLIC_FIDELITY,
            vlm_probe="Are any symbols, patterns, or motifs inaccurate or misused for {country} culture?",
        ),
        _FailureSpec(
            mode="temporal_mismatch",
            severity=FailureSeverity.MAJOR,
            dimension=CulturalDimension.TEMPORAL_CONSISTENCY,
            vlm_probe="Are there elements from incompatible time periods mixed together for {country} {category}?",
        ),
        _FailureSpec(
            mode="performative_error",
            severity=FailureSeverity.MINOR,
            dimension=CulturalDimension.PERFORMATIVE_ACCURACY,
            vlm_probe="Are gestures, rituals, or ceremonial practices shown inaccurately for {country}?",
        ),
    ],
    FailureCategory.SENSITIVITY: [
        _FailureSpec(
            mode="stereotype_reliance",
            severity=FailureSeverity.CRITICAL,
            dimension=CulturalDimension.COMPOSITIONAL_HARMONY,
            vlm_probe="Does this image rely on cultural stereotypes rather than authentic {country} representation?",
        ),
        _FailureSpec(
            mode="cultural_appropriation",
            severity=FailureSeverity.MAJOR,
            dimension=CulturalDimension.SYMBOLIC_FIDELITY,
            vlm_probe="Are sacred or culturally significant elements used inappropriately or out of context for {country}?",
        ),
    ],
    FailureCategory.COHERENCE: [
        _FailureSpec(
            mode="cultural_mixing",
            severity=FailureSeverity.MINOR,
            dimension=CulturalDimension.CONTEXTUAL_COHERENCE,
            vlm_probe="Are incompatible cultural elements from different traditions mixed in this {country} {category}?",
        ),
        _FailureSpec(
            mode="over_modernization",
            severity=FailureSeverity.MINOR,
            dimension=CulturalDimension.TEMPORAL_CONSISTENCY,
            vlm_probe="Is the image excessively modernized, losing traditional {country} cultural character?",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Enhanced Failure Detector
# ---------------------------------------------------------------------------

class FailureModeDetector:
    """Detects and classifies cultural failure modes using VLM probes.

    Supports both legacy keyword-based detection and the new hierarchical
    VLM-probe taxonomy (C4).
    """

    def __init__(self, vlm_model: str = "Qwen3-VL-8B"):
        self.vlm_model = vlm_model
        self._vlm_client = None
        logger.info(f"FailureModeDetector initialized: model={vlm_model}")

    @property
    def vlm_client(self):
        """Lazy load VLM client."""
        if self._vlm_client is None:
            from ..enhanced_cultural_metric_pipeline import EnhancedVLMClient
            self._vlm_client = EnhancedVLMClient(
                model_name=self.vlm_model,
                load_in_4bit=True,
            )
        return self._vlm_client

    # -- Legacy interface (backward compatible) --

    def detect_failure_modes(
        self,
        image_path: Path,
        country: str,
        category: str,
        detected_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect failure modes (legacy interface).

        Args:
            image_path: Path to image
            country: Target country
            category: Cultural category
            detected_issues: Optional pre-detected issues

        Returns:
            List of failure modes with details
        """
        if detected_issues is None:
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

    # -- Enhanced VLM-probe based detection (C4) --

    def detect_enhanced(
        self,
        image_path: Path,
        country: str,
        category: str,
        context: str = "",
    ) -> List[EnhancedFailureMode]:
        """Detect failures using hierarchical VLM-probe taxonomy.

        Args:
            image_path: Path to image
            country: Target country
            category: Cultural category
            context: Cultural context from RAG retrieval

        Returns:
            List of EnhancedFailureMode detections
        """
        detections: List[EnhancedFailureMode] = []

        for cat, specs in FAILURE_TAXONOMY.items():
            for spec in specs:
                probe = spec.vlm_probe.format(country=country, category=category)
                answer = self._probe_vlm(image_path, probe, context)
                is_failure, confidence = self._interpret_probe_answer(answer)

                if is_failure:
                    detections.append(EnhancedFailureMode(
                        category=cat,
                        mode=spec.mode,
                        severity=spec.severity,
                        confidence=confidence,
                        dimension_affected=spec.dimension,
                        evidence=answer,
                    ))

        return detections

    def compute_penalties(
        self,
        detections: List[EnhancedFailureMode],
    ) -> List[FailurePenalty]:
        """Convert enhanced detections to CultScore failure penalties."""
        return [
            FailurePenalty(
                mode=d.mode,
                severity=float(d.severity.value),
                confidence=d.confidence,
                dimension_affected=d.dimension_affected,
            )
            for d in detections
        ]

    # -- Internal helpers --

    def _probe_vlm(self, image_path: Path, question: str, context: str) -> str:
        """Ask a single VLM probe question."""
        try:
            return self.vlm_client.answer(image_path, question, context)
        except Exception as e:
            logger.debug(f"VLM probe failed: {e}")
            return "ambiguous"

    def _interpret_probe_answer(self, answer: str) -> tuple[bool, float]:
        """Interpret VLM answer as (is_failure, confidence).

        For failure probes, "yes" means a failure was detected.
        """
        answer_lower = answer.lower().strip()
        if answer_lower == "yes":
            return True, 0.8
        elif answer_lower == "no":
            return False, 0.9
        else:
            # Ambiguous — mild positive signal
            return True, 0.4

    def _detect_issues(
        self,
        image_path: Path,
        country: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """Detect cultural issues in an image (legacy)."""
        questions = [
            f"Are there any cultural inaccuracies in this {category} from {country}?",
            f"Does this image show over-modernized elements?",
            f"Are there stereotypes in this image?",
            f"Is the cultural context appropriate for {country}?",
        ]

        issues = []
        for question in questions:
            answer = self.vlm_client.answer(
                image_path=image_path,
                question=question,
                context="",
            )
            if self._is_issue_detected(answer):
                issues.append({
                    "question": question,
                    "answer": answer,
                    "confidence": 0.7,
                    "description": answer,
                })
        return issues

    def _classify_failure_mode(
        self,
        issue: Dict[str, Any],
        country: str,
        category: str,
    ) -> FailureMode:
        """Classify a failure mode from an issue (legacy keyword-based)."""
        description = issue.get("description", "").lower()
        question = issue.get("question", "").lower()

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
        """Check if an issue was detected from VLM answer (legacy)."""
        answer_lower = answer.lower()
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
