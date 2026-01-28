"""
CCUB2 Agent: Model-Agnostic Cultural Bias Mitigation System
"""

__version__ = "0.1.0"

from .data.job_creator import AgentJobCreator

# Lazy imports for agents (require ML dependencies like torch, transformers)
def __getattr__(name):
    _agent_names = {
        "OrchestratorAgent", "ScoutAgent", "JobAgent", "EditAgent",
        "JudgeAgent", "MetricAgent", "BenchmarkAgent", "ReviewQAAgent",
        "CaptionAgent", "IndexReleaseAgent", "DataValidatorAgent",
        "CountryLeadAgent", "VerificationAgent",
    }
    if name in _agent_names:
        from . import agents
        return getattr(agents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AgentJobCreator",
    # Agents (lazy-loaded)
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
