"""Cultural Metric Components

Modular components for cultural metric evaluation:
- VQA Scorer: VQA-based cultural scoring
- RAG Retriever: Cultural knowledge retrieval
- Failure Detector: Failure mode classification (legacy + hierarchical C4)
"""

from .vqa_scorer import VQACulturalScorer, create_vqa_scorer
from .rag_retriever import CulturalRAGRetriever, create_rag_retriever
from .failure_detector import (
    FailureModeDetector,
    create_failure_detector,
    # C4 hierarchical taxonomy
    FailureMode,
    FailureSeverity,
    FailureCategory,
    EnhancedFailureMode,
    FAILURE_TAXONOMY,
)

__all__ = [
    "VQACulturalScorer",
    "create_vqa_scorer",
    "CulturalRAGRetriever",
    "create_rag_retriever",
    "FailureModeDetector",
    "create_failure_detector",
    "FailureMode",
    "FailureSeverity",
    "FailureCategory",
    "EnhancedFailureMode",
    "FAILURE_TAXONOMY",
]
