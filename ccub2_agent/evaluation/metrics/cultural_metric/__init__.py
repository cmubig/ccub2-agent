"""Cultural Metric — CultScore Evaluation System.

NeurIPS-level cultural fidelity metric with:
- C1: 6-Dimensional Cultural Fidelity Taxonomy
- C2: CultScore — Confidence-Calibrated Composite Metric
- C3: Source-Authority Weighted Retrieval
- C4: Hierarchical Failure Mode Taxonomy + Severity
- C5: Human Validation (Krippendorff's alpha, ICC, ECE)
"""

from .taxonomy import (
    CulturalDimension,
    DimensionScore,
    CulturalProfile,
    FailurePenalty,
    CATEGORY_DIMENSION_WEIGHTS,
    DIMENSION_QUESTION_TEMPLATES,
    get_dimension_weights,
    format_dimension_questions,
)
from .cultscore import CultScoreComputer, CultScoreConfig

__all__ = [
    # C1 Taxonomy
    "CulturalDimension",
    "DimensionScore",
    "CulturalProfile",
    "FailurePenalty",
    "CATEGORY_DIMENSION_WEIGHTS",
    "DIMENSION_QUESTION_TEMPLATES",
    "get_dimension_weights",
    "format_dimension_questions",
    # C2 CultScore
    "CultScoreComputer",
    "CultScoreConfig",
]
