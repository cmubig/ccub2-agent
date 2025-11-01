"""
CCUB2 Agent: Model-Agnostic Cultural Bias Mitigation System
"""

__version__ = "0.1.0"

from .agent import CulturalAgent
from .modules.adapter import CulturalCorrectionAdapter
from .modules.detector import CulturalDetector
from .modules.ccub_metric import CCUBMetric
from .modules.agent_job_creator import AgentJobCreator

__all__ = [
    "CulturalAgent",
    "CulturalCorrectionAdapter",
    "CulturalDetector",
    "CCUBMetric",
    "AgentJobCreator",
]
