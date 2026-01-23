"""
CCUB2 Agent: Model-Agnostic Cultural Bias Mitigation System
"""

__version__ = "0.1.0"

from .data.job_creator import AgentJobCreator

# Import agents
from .agents import (
    OrchestratorAgent,
    ScoutAgent,
    JobAgent,
    EditAgent,
    JudgeAgent,
    MetricAgent,
    BenchmarkAgent,
    ReviewQAAgent,
    CaptionAgent,
    IndexReleaseAgent,
    DataValidatorAgent,
    CountryLeadAgent,
    VerificationAgent,
)

__all__ = [
    "AgentJobCreator",
    # Agents
    "OrchestratorAgent",
    "ScoutAgent",
    "JobAgent",
    "EditAgent",
    "JudgeAgent",
    "MetricAgent",
    "BenchmarkAgent",
    "ReviewQAAgent",
    "CaptionAgent",
    "IndexReleaseAgent",
    "DataValidatorAgent",
    "CountryLeadAgent",
    "VerificationAgent",
]
