"""
Pydantic Models for Pipeline
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution"""
    prompt: str = Field(..., description="Text prompt for image generation")
    country: str = Field(..., description="Target country (e.g., 'korea')")
    category: str = Field(..., description="Image category (e.g., 'traditional_clothing')")
    t2i_model: str = Field(default="sdxl", description="T2I model to use")
    i2i_model: str = Field(default="qwen", description="I2I model to use")
    max_iterations: int = Field(default=3, description="Maximum editing iterations")
    target_score: float = Field(default=8.0, description="Target cultural score")
    load_in_4bit: bool = Field(default=True, description="Use 4-bit quantization for models")

    class Config:
        protected_namespaces = ()  # Disable protected namespace warning for model_* fields


class PipelineRequest(BaseModel):
    """Request to start pipeline execution"""
    config: PipelineConfig


class PipelineState(BaseModel):
    """Current state of the pipeline"""
    status: PipelineStatus = PipelineStatus.IDLE
    current_node_id: Optional[str] = None
    current_iteration: int = 0
    progress: float = 0.0
    config: Optional[PipelineConfig] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None

    # Results
    initial_image: Optional[str] = None
    final_image: Optional[str] = None
    iteration_images: List[str] = Field(default_factory=list)
    scores_history: List[Dict[str, float]] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class PipelineResponse(BaseModel):
    """Response after starting pipeline"""
    success: bool
    message: str
    pipeline_id: Optional[str] = None
    state: Optional[PipelineState] = None
