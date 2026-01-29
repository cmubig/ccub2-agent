"""CultScore — Confidence-Calibrated Composite Cultural Metric (C2).

Implements the scoring formula:
    CultScore      = sum(w_d * c_d * A_d * S_d) / sum(w_d * c_d * A_d)
    CultScore_conf = sum(w_d * c_d * A_d) / sum(w_d)
    CultScore_final = CultScore * prod(1 - severity_f * confidence_f)

where:
    w_d = category-specific dimension weight
    c_d = confidence (1 - std/sigma_max, clipped)
    A_d = source authority weight
    S_d = raw dimension score (0-1)
    severity_f, confidence_f = failure penalty terms
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .taxonomy import (
    CulturalDimension,
    CulturalProfile,
    DimensionScore,
    FailurePenalty,
    get_dimension_weights,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CultScoreConfig:
    """Configuration for CultScore computation."""
    num_passes: int = 3  # stochastic VLM passes per dimension
    temperature: float = 0.7  # VLM sampling temperature
    sigma_max: float = 0.3  # max std for confidence clipping
    # Source authority weights by source type
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "unesco_ich": 1.0,
        "wikipedia": 0.7,
        "wikivoyage": 0.5,
    })
    default_source_weight: float = 0.5
    min_confidence: float = 0.1  # floor for confidence
    min_authority: float = 0.3  # floor for authority weight


# ---------------------------------------------------------------------------
# CultScore computation engine
# ---------------------------------------------------------------------------

class CultScoreComputer:
    """Computes CultScore from multi-pass VLM dimension scores."""

    def __init__(self, config: Optional[CultScoreConfig] = None):
        self.config = config or CultScoreConfig()

    # -- Per-dimension scoring --

    def compute_dimension_score(
        self,
        raw_scores: List[float],
        source_authority: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Compute aggregated dimension score from multi-pass raw scores.

        Args:
            raw_scores: List of 0-1 scores from N stochastic VLM passes.
            source_authority: Authority weight from retrieval source.

        Returns:
            (mean_score, confidence, clamped_authority)
        """
        if not raw_scores:
            return 0.0, self.config.min_confidence, self.config.min_authority

        n = len(raw_scores)
        mean = sum(raw_scores) / n

        if n >= 2:
            variance = sum((x - mean) ** 2 for x in raw_scores) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = self.config.sigma_max  # single pass → lowest confidence

        # Confidence: higher when std is low relative to sigma_max
        confidence = max(
            self.config.min_confidence,
            1.0 - std / self.config.sigma_max,
        )
        confidence = min(confidence, 1.0)

        authority = max(self.config.min_authority, source_authority)

        return mean, confidence, authority

    def build_dimension_score(
        self,
        dimension: CulturalDimension,
        raw_scores: List[float],
        source_authority: float = 1.0,
    ) -> DimensionScore:
        """Build a DimensionScore from raw multi-pass scores."""
        mean, confidence, authority = self.compute_dimension_score(
            raw_scores, source_authority
        )
        return DimensionScore(
            dimension=dimension,
            raw_score=mean,
            confidence=confidence,
            evidence_count=len(raw_scores),
            source_authority=authority,
        )

    # -- Composite CultScore --

    def compute_cultscore(
        self,
        dimension_scores: Dict[CulturalDimension, DimensionScore],
        category: str,
        failure_penalties: Optional[List[FailurePenalty]] = None,
    ) -> Tuple[float, float, float]:
        """Compute the final CultScore.

        Args:
            dimension_scores: Per-dimension DimensionScore objects.
            category: Cultural category for weight lookup.
            failure_penalties: Optional failure penalties.

        Returns:
            (cultscore, cultscore_confidence, cultscore_penalised)
        """
        weights = get_dimension_weights(category)
        failure_penalties = failure_penalties or []

        numerator = 0.0
        denominator = 0.0
        weight_sum = 0.0

        for dim, w_d in weights.items():
            weight_sum += w_d
            ds = dimension_scores.get(dim)
            if ds is None:
                continue

            w = w_d
            c = ds.confidence
            a = ds.source_authority
            s = ds.raw_score

            numerator += w * c * a * s
            denominator += w * c * a

        # Base CultScore
        if denominator > 0:
            cultscore = numerator / denominator
        else:
            cultscore = 0.0

        # Confidence of overall score
        if weight_sum > 0:
            cultscore_confidence = denominator / weight_sum
        else:
            cultscore_confidence = 0.0

        # Apply failure penalties
        penalty_product = 1.0
        for fp in failure_penalties:
            penalty_product *= (1.0 - fp.severity * fp.confidence)

        cultscore_penalised = cultscore * penalty_product

        # Clamp to [0, 1]
        cultscore = max(0.0, min(1.0, cultscore))
        cultscore_confidence = max(0.0, min(1.0, cultscore_confidence))
        cultscore_penalised = max(0.0, min(1.0, cultscore_penalised))

        return cultscore, cultscore_confidence, cultscore_penalised

    def build_cultural_profile(
        self,
        dimension_scores: Dict[CulturalDimension, DimensionScore],
        category: str,
        country: str,
        failure_penalties: Optional[List[FailurePenalty]] = None,
    ) -> CulturalProfile:
        """Build a complete CulturalProfile from dimension scores."""
        failure_penalties = failure_penalties or []

        cultscore, cultscore_conf, cultscore_pen = self.compute_cultscore(
            dimension_scores, category, failure_penalties
        )

        return CulturalProfile(
            dimension_scores=dimension_scores,
            cultscore=cultscore,
            cultscore_confidence=cultscore_conf,
            cultscore_penalised=cultscore_pen,
            failure_penalties=failure_penalties,
            category=category,
            country=country,
        )

    # -- Utility --

    def get_source_authority(self, source_type: str) -> float:
        """Look up source authority weight by source type string."""
        return self.config.source_weights.get(
            source_type, self.config.default_source_weight
        )
