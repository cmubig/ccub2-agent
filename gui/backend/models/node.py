"""
Pydantic Models for Node Data
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class NodeStatus(str, Enum):
    """Node execution status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class NodeType(str, Enum):
    """Types of nodes in the pipeline"""
    INPUT = "input"
    T2I_GENERATOR = "t2i_generator"
    VLM_DETECTOR = "vlm_detector"
    TEXT_KB_QUERY = "text_kb_query"
    CLIP_RAG_SEARCH = "clip_rag_search"
    REFERENCE_SELECTOR = "reference_selector"
    PROMPT_ADAPTER = "prompt_adapter"
    I2I_EDITOR = "i2i_editor"
    ITERATION_CHECK = "iteration_check"
    OUTPUT = "output"


class NodePosition(BaseModel):
    """Node position on canvas"""
    x: float
    y: float


class VLMDetectorData(BaseModel):
    """Data specific to VLM Detector node"""
    cultural_score: Optional[float] = None
    prompt_score: Optional[float] = None
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    text_kb_results: List[Dict[str, Any]] = Field(default_factory=list)
    clip_rag_results: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_time: Optional[float] = None


class ReferenceSelectorData(BaseModel):
    """Data specific to Reference Selector node"""
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    selected_reference: Optional[Dict[str, Any]] = None
    clip_scores: List[float] = Field(default_factory=list)
    keyword_matches: List[int] = Field(default_factory=list)
    final_scores: List[float] = Field(default_factory=list)


class I2IEditorData(BaseModel):
    """Data specific to I2I Editor node"""
    model_name: str
    iteration: int
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    prompt_used: Optional[str] = None
    reference_image: Optional[str] = None
    editing_time: Optional[float] = None


class T2IGeneratorData(BaseModel):
    """Data specific to T2I Generator node"""
    model_name: str
    prompt: str
    generated_image: Optional[str] = None
    generation_time: Optional[float] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class NodeData(BaseModel):
    """Base node data model"""
    id: str
    type: NodeType
    label: str
    status: NodeStatus = NodeStatus.PENDING
    position: NodePosition
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0  # 0.0 to 1.0

    class Config:
        use_enum_values = True


class NodeUpdate(BaseModel):
    """Node update message"""
    node_id: str
    status: Optional[NodeStatus] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None

    class Config:
        use_enum_values = True
