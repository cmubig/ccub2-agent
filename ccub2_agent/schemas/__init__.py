"""
Agent Message Schemas

Type-safe message protocols for agent communication.
Essential for reproducibility and logging.
"""

from .agent_messages import (
    DetectionOutput,
    RetrievalOutput,
    EditingOutput,
    EvaluationOutput,
    JobCreationOutput,
    GapAnalysisOutput,
    AgentMessage,
    AgentMessageType,
)

__all__ = [
    "DetectionOutput",
    "RetrievalOutput",
    "EditingOutput",
    "EvaluationOutput",
    "JobCreationOutput",
    "GapAnalysisOutput",
    "AgentMessage",
    "AgentMessageType",
]
