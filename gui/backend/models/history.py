"""
Pipeline History Data Models
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class PromptFlowStep(BaseModel):
    """Records how a prompt was transformed at each step"""
    step: str  # e.g., "original", "adapted", "final"
    prompt: str
    metadata: Dict[str, Any] = {}


class RAGResult(BaseModel):
    """RAG search result"""
    query: str
    results: List[Dict[str, Any]]
    top_k: int
    search_time: float


class NodeHistory(BaseModel):
    """Detailed history for a single node execution"""
    node_id: str
    node_type: str
    status: str
    start_time: float
    end_time: Optional[float] = None

    # Prompt flow tracking
    prompt_flow: List[PromptFlowStep] = []

    # RAG results (for VLM, Reference Selector)
    text_rag: Optional[RAGResult] = None
    clip_rag: Optional[RAGResult] = None

    # Reference selection details
    reference_candidates: List[Dict[str, Any]] = []
    reference_scores: Dict[str, Any] = {}
    selected_reference: Optional[Dict[str, Any]] = None

    # VLM detection details
    vlm_analysis: Optional[Dict[str, Any]] = None

    # I2I editing details
    editing_params: Optional[Dict[str, Any]] = None
    editing_prompt: Optional[str] = None
    issues_fixed: Optional[int] = None
    iteration: Optional[int] = None

    # Output data
    output_data: Dict[str, Any] = {}
    error: Optional[str] = None


class PipelineHistory(BaseModel):
    """Complete history of a pipeline execution"""
    pipeline_id: str
    status: str  # "running", "completed", "error"

    # Configuration
    config: Dict[str, Any]

    # Timing
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # Results
    initial_image_path: Optional[str] = None
    final_image_path: Optional[str] = None
    iteration_count: int = 0
    final_cultural_score: Optional[float] = None
    final_prompt_score: Optional[float] = None

    # Detailed node history
    nodes: Dict[str, NodeHistory] = {}

    # Score progression
    scores_history: List[Dict[str, float]] = []

    # Jobs created
    jobs_created: List[Dict[str, Any]] = []

    def to_summary(self) -> Dict[str, Any]:
        """Create a summary for history list"""
        return {
            "pipeline_id": self.pipeline_id,
            "timestamp": self.start_time,
            "prompt": self.config.get("prompt", "Unknown"),
            "country": self.config.get("country", "Unknown"),
            "status": self.status,
            "duration": self.duration,
            "iterations": self.iteration_count,
            "final_score": self.final_cultural_score,
            "t2i_model": self.config.get("t2i_model", "Unknown"),
            "i2i_model": self.config.get("i2i_model", "Unknown"),
        }
