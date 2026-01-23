"""
Unit tests for agent message schemas.
"""

import pytest
from datetime import datetime

from ccub2_agent.schemas import (
    DetectionOutput,
    RetrievalOutput,
    EditingOutput,
    AgentMessage,
    AgentMessageType,
)


def test_detection_output():
    """Test DetectionOutput schema."""
    output = DetectionOutput(
        failure_modes=["over_modernization"],
        cultural_score=6.5,
        prompt_score=7.0,
        confidence=0.8,
        reference_needed=True,
        decision="ITERATE",
    )
    
    assert output.cultural_score == 6.5
    assert output.reference_needed is True
    assert output.decision == "ITERATE"


def test_retrieval_output():
    """Test RetrievalOutput schema."""
    output = RetrievalOutput(
        reference_images=["ref1.png", "ref2.png"],
        reference_texts=["text1", "text2"],
        retrieval_method="CLIP",
        num_references=2,
    )
    
    assert len(output.reference_images) == 2
    assert output.retrieval_method == "CLIP"


def test_editing_output():
    """Test EditingOutput schema."""
    output = EditingOutput(
        output_image="edited.png",
        editing_prompt="Improve cultural accuracy",
        model="qwen",
        strength=0.35,
        iteration=1,
    )
    
    assert output.model == "qwen"
    assert output.iteration == 1


def test_agent_message():
    """Test AgentMessage wrapper."""
    detection = DetectionOutput(
        failure_modes=["stereotype"],
        cultural_score=5.0,
        prompt_score=6.0,
        confidence=0.7,
        decision="ITERATE",
    )
    
    message = AgentMessage(
        message_type=AgentMessageType.DETECTION,
        agent_name="JudgeAgent",
        data=detection.dict(),
    )
    
    assert message.message_type == AgentMessageType.DETECTION
    assert message.agent_name == "JudgeAgent"
