"""
Hyperparameter Configuration

Tracks all hyperparameters for exact reproducibility.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """
    Complete hyperparameter configuration.
    
    All hyperparameters used in experiments are tracked here.
    """
    # Loop parameters
    max_iterations: int = 5
    target_cultural_score: float = 8.5
    score_threshold: float = 8.0
    
    # VLM parameters
    vlm_model: str = "Qwen3-VL-8B"
    vlm_load_in_4bit: bool = True
    detection_confidence_threshold: float = 0.7
    
    # Retrieval parameters
    clip_top_k: int = 3
    text_rag_top_k: int = 5
    retrieval_method: str = "hybrid"  # "CLIP", "text_rag", "hybrid"
    
    # Editing parameters
    default_i2i_model: str = "qwen"
    editing_strength: float = 0.35
    reference_strength: float = 0.4
    
    # Prompt adaptation
    prompt_adaptation_enabled: bool = True
    
    # Gap analysis
    gap_analysis_enabled: bool = True
    min_coverage_score: float = 0.7
    
    # Job creation
    default_target_count: int = 20
    job_priority: str = "medium"  # "low", "medium", "high"
    
    # Reproducibility
    random_seed: int = 42
    numpy_seed: Optional[int] = None
    torch_seed: Optional[int] = None
    
    # Logging
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    
    # Timestamp
    created_at: str = ""
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        
        # Set seeds if not provided
        if self.numpy_seed is None:
            self.numpy_seed = self.random_seed
        if self.torch_seed is None:
            self.torch_seed = self.random_seed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def apply_seeds(self):
        """Apply random seeds for reproducibility."""
        import random
        import numpy as np
        import torch
        
        random.seed(self.random_seed)
        np.random.seed(self.numpy_seed)
        torch.manual_seed(self.torch_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.torch_seed)
        
        logger.info(f"Random seeds applied: {self.random_seed}, {self.numpy_seed}, {self.torch_seed}")


def get_default_hyperparameters() -> HyperparameterConfig:
    """Get default hyperparameter configuration."""
    return HyperparameterConfig()


def load_hyperparameters(config_path: Path) -> HyperparameterConfig:
    """
    Load hyperparameters from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        HyperparameterConfig instance
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # Remove timestamp if present (will be set in __post_init__)
    if "created_at" in config_dict:
        del config_dict["created_at"]
    
    config = HyperparameterConfig(**config_dict)
    
    logger.info(f"Hyperparameters loaded from: {config_path}")
    return config


def save_hyperparameters(config: HyperparameterConfig, config_path: Path):
    """
    Save hyperparameters to YAML file.
    
    Args:
        config: HyperparameterConfig instance
        config_path: Path to save YAML file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Hyperparameters saved to: {config_path}")


# Default hyperparameters file
DEFAULT_HYPERPARAMETERS_YAML = """
# WorldCCUB Agent Hyperparameters
# For exact reproducibility of experiments

# Loop parameters
max_iterations: 5
target_cultural_score: 8.5
score_threshold: 8.0

# VLM parameters
vlm_model: "Qwen3-VL-8B"
vlm_load_in_4bit: true
detection_confidence_threshold: 0.7

# Retrieval parameters
clip_top_k: 3
text_rag_top_k: 5
retrieval_method: "hybrid"  # "CLIP", "text_rag", "hybrid"

# Editing parameters
default_i2i_model: "qwen"
editing_strength: 0.35
reference_strength: 0.4

# Prompt adaptation
prompt_adaptation_enabled: true

# Gap analysis
gap_analysis_enabled: true
min_coverage_score: 0.7

# Job creation
default_target_count: 20
job_priority: "medium"  # "low", "medium", "high"

# Reproducibility
random_seed: 42
numpy_seed: 42
torch_seed: 42

# Logging
log_level: "INFO"
save_intermediate_results: true
"""
