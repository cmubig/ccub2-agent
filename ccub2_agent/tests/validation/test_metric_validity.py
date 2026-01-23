"""
Validation tests for cultural metric validity.

Tests correlation with human judgments.
"""

import pytest
from pathlib import Path

from ccub2_agent.evaluation.metrics.cultural_metric.calibration import (
    HumanValidationProtocol,
    HumanJudgment,
    MetricPrediction,
    compute_correlation,
)


def test_correlation_computation():
    """Test correlation computation."""
    human_scores = [8.0, 7.5, 6.0, 9.0, 5.5]
    metric_scores = [7.8, 7.3, 6.2, 8.9, 5.7]
    
    correlation = compute_correlation(human_scores, metric_scores)
    
    assert "pearson_r" in correlation
    assert "spearman_r" in correlation
    assert correlation["pearson_r"] > 0.9  # Should be highly correlated


def test_validation_protocol():
    """Test human validation protocol."""
    protocol = HumanValidationProtocol(Path("test_output/"))
    
    # Add judgments
    protocol.add_human_judgment(
        image_id="img1",
        judge_id="judge1",
        cultural_score=8.0,
        prompt_score=7.5,
        failure_modes=["over_modernization"],
    )
    
    # Add predictions
    protocol.add_metric_prediction(
        image_id="img1",
        cultural_score=7.8,
        prompt_score=7.3,
        failure_modes=["over_modernization"],
    )
    
    correlation = protocol.compute_correlation()
    
    assert correlation["num_pairs"] == 1
    assert "cultural_pearson" in correlation
