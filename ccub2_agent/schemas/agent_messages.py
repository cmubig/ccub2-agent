"""
Agent Message Schemas

Type-safe message protocols for inter-agent communication.
All agent outputs follow these schemas for reproducibility and logging.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AgentMessageType(str, Enum):
    """Types of agent messages."""
    DETECTION = "detection"
    RETRIEVAL = "retrieval"
    EDITING = "editing"
    EVALUATION = "evaluation"
    JOB_CREATION = "job_creation"
    GAP_ANALYSIS = "gap_analysis"
    VERIFICATION = "verification"
    CAPTION_NORMALIZATION = "caption_normalization"


class DetectionOutput(BaseModel):
    """
    Output from detection agents (JudgeAgent, VLMCulturalDetector).
    
    Records what cultural issues were detected and with what confidence.
    """
    failure_modes: List[str] = Field(
        default_factory=list,
        description="List of detected failure mode types (e.g., 'over_modernization', 'stereotype_reliance')"
    )
    cultural_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Cultural authenticity score (0-10)"
    )
    prompt_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Prompt alignment score (0-10)"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in detection (0-1)"
    )
    reference_needed: bool = Field(
        default=False,
        description="Whether reference images are needed for correction"
    )
    detected_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed list of detected issues with locations and descriptions"
    )
    decision: Literal["STOP", "ITERATE"] = Field(
        description="Loop decision based on scores"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this detection was performed"
    )


class RetrievalOutput(BaseModel):
    """
    Output from retrieval agents (CLIPImageRAG, ReferenceImageSelector).
    
    Records what references were retrieved and why.
    """
    reference_images: List[str] = Field(
        default_factory=list,
        description="Paths to retrieved reference images"
    )
    reference_texts: List[str] = Field(
        default_factory=list,
        description="Retrieved text knowledge chunks"
    )
    coverage_gap: Optional[str] = Field(
        default=None,
        description="Identified coverage gap (None if data is sufficient)"
    )
    retrieval_scores: List[float] = Field(
        default_factory=list,
        description="Similarity scores for each retrieved reference"
    )
    retrieval_method: Literal["CLIP", "text_rag", "hybrid"] = Field(
        description="Method used for retrieval"
    )
    num_references: int = Field(
        ge=0,
        description="Number of references retrieved"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When retrieval was performed"
    )


class EditingOutput(BaseModel):
    """
    Output from editing agents (EditAgent, I2I adapters).
    
    Records what edits were made and with what model.
    """
    output_image: str = Field(
        description="Path to edited output image"
    )
    editing_prompt: str = Field(
        description="Model-specific editing prompt used"
    )
    model: str = Field(
        description="I2I model used (e.g., 'qwen', 'flux', 'sdxl')"
    )
    reference_used: Optional[str] = Field(
        default=None,
        description="Reference image path if used"
    )
    strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Editing strength (0-1)"
    )
    iteration: int = Field(
        ge=0,
        description="Iteration number in the loop"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When editing was performed"
    )


class EvaluationOutput(BaseModel):
    """
    Output from evaluation agents (MetricAgent, BenchmarkAgent).
    
    Records evaluation results with detailed scores and rationales.
    """
    cultural_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall cultural score"
    )
    dimension_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores per dimension (e.g., 'authenticity', 'modernization', 'stereotype')"
    )
    failure_modes: List[str] = Field(
        default_factory=list,
        description="Detected failure modes"
    )
    rationale: str = Field(
        description="Explanation for the scores"
    )
    metric_version: str = Field(
        default="v1.0",
        description="Version of metric used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When evaluation was performed"
    )


class JobCreationOutput(BaseModel):
    """
    Output from job creation agents (JobAgent, AgentJobCreator).
    
    Records what data collection jobs were created.
    """
    job_id: str = Field(
        description="Firebase job ID"
    )
    country: str = Field(
        description="Target country for data collection"
    )
    category: str = Field(
        description="Target category"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of missing cultural elements"
    )
    target_count: int = Field(
        ge=0,
        description="Target number of contributions"
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Job priority"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When job was created"
    )


class GapAnalysisOutput(BaseModel):
    """
    Output from gap analysis agents (ScoutAgent, DataGapAnalyzer).
    
    Records what gaps were identified and their priorities.
    """
    gaps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of identified gaps"
    )
    needs_more_data: bool = Field(
        description="Whether more data is needed"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of missing cultural elements"
    )
    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Current coverage score (0-1)"
    )
    priority_gaps: List[str] = Field(
        default_factory=list,
        description="High-priority gaps to address first"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When analysis was performed"
    )


class AgentMessage(BaseModel):
    """
    Generic agent message wrapper.
    
    All agent outputs should be wrapped in this for logging and tracking.
    """
    message_type: AgentMessageType = Field(
        description="Type of message"
    )
    agent_name: str = Field(
        description="Name of the agent that produced this message"
    )
    data: Dict[str, Any] = Field(
        description="Message payload (one of the Output schemas)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model versions, configs, etc.)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When message was created"
    )
    message_id: str = Field(
        default="",
        description="Unique message ID for tracking"
    )
