"""6-Dimensional Cultural Fidelity Taxonomy (C1).

Defines the cultural dimensions, scoring structures, category-specific
weights, and VLM question templates used by CultScore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# C1. Six Cultural Dimensions
# ---------------------------------------------------------------------------

class CulturalDimension(str, Enum):
    """Six orthogonal axes of cultural fidelity."""
    MATERIAL_AUTHENTICITY = "material_authenticity"
    SYMBOLIC_FIDELITY = "symbolic_fidelity"
    CONTEXTUAL_COHERENCE = "contextual_coherence"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    PERFORMATIVE_ACCURACY = "performative_accuracy"
    COMPOSITIONAL_HARMONY = "compositional_harmony"


DIMENSION_DESCRIPTIONS: Dict[CulturalDimension, str] = {
    CulturalDimension.MATERIAL_AUTHENTICITY: "Materials, textures, and physical substances",
    CulturalDimension.SYMBOLIC_FIDELITY: "Symbols, patterns, colors, and iconography",
    CulturalDimension.CONTEXTUAL_COHERENCE: "Scene placement, social norms, and setting",
    CulturalDimension.TEMPORAL_CONSISTENCY: "Era-appropriate elements (traditional vs modern)",
    CulturalDimension.PERFORMATIVE_ACCURACY: "Gestures, rituals, ceremonies, and practices",
    CulturalDimension.COMPOSITIONAL_HARMONY: "Overall aesthetic unity and cultural coherence",
}


# ---------------------------------------------------------------------------
# Scoring dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single cultural dimension."""
    dimension: CulturalDimension
    raw_score: float  # 0-1 normalised
    confidence: float  # 0-1, derived from multi-pass std
    evidence_count: int = 0  # number of questions/probes
    source_authority: float = 1.0  # authority weight from retrieval


@dataclass
class FailurePenalty:
    """A single failure penalty applied to CultScore."""
    mode: str
    severity: float  # 0-1
    confidence: float  # 0-1
    dimension_affected: Optional[CulturalDimension] = None


@dataclass
class CulturalProfile:
    """Complete 6-dimensional cultural evaluation profile."""
    dimension_scores: Dict[CulturalDimension, DimensionScore]
    cultscore: float = 0.0
    cultscore_confidence: float = 0.0
    cultscore_penalised: float = 0.0
    failure_penalties: List[FailurePenalty] = field(default_factory=list)
    category: str = ""
    country: str = ""


# ---------------------------------------------------------------------------
# Category-dimension weight matrix (8 categories x 6 dims)
# Rows sum to 1.0 for each category.
# ---------------------------------------------------------------------------

CATEGORY_DIMENSION_WEIGHTS: Dict[str, Dict[CulturalDimension, float]] = {
    "architecture": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.25,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.15,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.20,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.15,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.05,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.20,
    },
    "art": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.15,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.25,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.15,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.10,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.10,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.25,
    },
    "event": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.10,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.15,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.25,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.10,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.25,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.15,
    },
    "fashion": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.25,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.20,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.15,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.15,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.05,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.20,
    },
    "food": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.30,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.10,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.20,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.10,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.10,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.20,
    },
    "landscape": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.15,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.10,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.30,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.10,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.05,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.30,
    },
    "people": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.10,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.15,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.20,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.10,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.25,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.20,
    },
    "wildlife": {
        CulturalDimension.MATERIAL_AUTHENTICITY: 0.15,
        CulturalDimension.SYMBOLIC_FIDELITY: 0.10,
        CulturalDimension.CONTEXTUAL_COHERENCE: 0.30,
        CulturalDimension.TEMPORAL_CONSISTENCY: 0.10,
        CulturalDimension.PERFORMATIVE_ACCURACY: 0.05,
        CulturalDimension.COMPOSITIONAL_HARMONY: 0.30,
    },
}

# Default uniform weights for unknown categories
_DEFAULT_WEIGHT = 1.0 / len(CulturalDimension)
DEFAULT_DIMENSION_WEIGHTS: Dict[CulturalDimension, float] = {
    dim: _DEFAULT_WEIGHT for dim in CulturalDimension
}


def get_dimension_weights(category: str) -> Dict[CulturalDimension, float]:
    """Return dimension weights for a category, falling back to uniform."""
    return CATEGORY_DIMENSION_WEIGHTS.get(
        category.lower() if category else "",
        DEFAULT_DIMENSION_WEIGHTS,
    )


# ---------------------------------------------------------------------------
# VLM question templates per dimension
# ---------------------------------------------------------------------------

DIMENSION_QUESTION_TEMPLATES: Dict[CulturalDimension, List[str]] = {
    CulturalDimension.MATERIAL_AUTHENTICITY: [
        "Are the materials and textures shown authentic to {country} {category} traditions?",
        "Do the physical substances (wood, fabric, stone, etc.) match what is traditionally used in {country}?",
        "Are there any materials that look artificial or foreign to {country} {category}?",
    ],
    CulturalDimension.SYMBOLIC_FIDELITY: [
        "Are the patterns, symbols, and motifs accurate for {country} {category}?",
        "Are the colors used culturally appropriate for {country}?",
        "Do any symbols or icons appear incorrect or misattributed for {country} culture?",
    ],
    CulturalDimension.CONTEXTUAL_COHERENCE: [
        "Is the scene setting appropriate for {country} cultural context?",
        "Does the social context (surroundings, background) match {country} norms?",
        "Are there any contextually inappropriate elements for {country}?",
    ],
    CulturalDimension.TEMPORAL_CONSISTENCY: [
        "Are all elements temporally consistent (all traditional or all modern) for {country}?",
        "Are there anachronistic elements that mix incompatible time periods for {country}?",
        "Does the overall era depicted match the intended {variant} representation of {country}?",
    ],
    CulturalDimension.PERFORMATIVE_ACCURACY: [
        "Are gestures, body language, and postures culturally accurate for {country}?",
        "If rituals or ceremonies are shown, are they performed correctly for {country} traditions?",
        "Are social interactions and behavioral norms appropriate for {country}?",
    ],
    CulturalDimension.COMPOSITIONAL_HARMONY: [
        "Does the overall composition reflect {country} aesthetic sensibilities?",
        "Is there visual harmony and cultural coherence in the image for {country}?",
        "Does the image feel authentic as a whole representation of {country} {category}?",
    ],
}


def format_dimension_questions(
    dimension: CulturalDimension,
    country: str,
    category: str = "cultural",
    variant: str = "general",
) -> List[str]:
    """Fill placeholders in dimension question templates."""
    templates = DIMENSION_QUESTION_TEMPLATES.get(dimension, [])
    return [
        t.format(country=country, category=category, variant=variant)
        for t in templates
    ]
