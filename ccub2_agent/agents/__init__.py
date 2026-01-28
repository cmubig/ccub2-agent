"""
WorldCCUB Multi-Agent Loop System Agents

This package contains the agent implementations for the WorldCCUB multi-agent loop.
Each agent is a Python class that handles a specific responsibility in the cultural
improvement pipeline.

Organized into sub-packages:
- core: Core multi-agent loop agents
- evaluation: Evaluation and benchmark agents
- data: Data pipeline agents
- governance: Governance agents
"""

# Core loop agents
from .core import (
    OrchestratorAgent,
    ScoutAgent,
    EditAgent,
    JudgeAgent,
    JobAgent,
    VerificationAgent,
)

# Evaluation agents
from .evaluation import (
    MetricAgent,
    BenchmarkAgent,
    ReviewQAAgent,
)

# Data pipeline agents
from .data import (
    CaptionAgent,
    IndexReleaseAgent,
    DataValidatorAgent,
)

# Governance agents
from .governance import (
    CountryRepAgent,
)

__all__ = [
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
    "CountryRepAgent",
    "VerificationAgent",
]
