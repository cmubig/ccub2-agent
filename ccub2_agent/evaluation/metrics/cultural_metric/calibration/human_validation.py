"""
Human Validation Protocol

Validates cultural metric against human judgments.
Essential for NeurIPS metric validity section.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import logging
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class HumanJudgment:
    """A single human judgment."""
    image_id: str
    judge_id: str
    cultural_score: float  # 0-10
    prompt_score: float  # 0-10
    failure_modes: List[str]
    comments: str = ""
    timestamp: str = ""


@dataclass
class MetricPrediction:
    """A metric prediction."""
    image_id: str
    cultural_score: float  # 0-10
    prompt_score: float  # 0-10
    failure_modes: List[str]
    confidence: float = 0.0


class HumanValidationProtocol:
    """
    Human validation protocol for metric calibration.
    
    Compares metric predictions with human judgments.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize validation protocol.
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.human_judgments: List[HumanJudgment] = []
        self.metric_predictions: List[MetricPrediction] = []
        
        logger.info(f"HumanValidationProtocol initialized: output_dir={output_dir}")
    
    def add_human_judgment(
        self,
        image_id: str,
        judge_id: str,
        cultural_score: float,
        prompt_score: float,
        failure_modes: List[str],
        comments: str = "",
    ):
        """
        Add a human judgment.
        
        Args:
            image_id: Image ID
            judge_id: Judge ID (insider from target country)
            cultural_score: Cultural score (0-10)
            prompt_score: Prompt alignment score (0-10)
            failure_modes: List of detected failure modes
            comments: Optional comments
        """
        judgment = HumanJudgment(
            image_id=image_id,
            judge_id=judge_id,
            cultural_score=cultural_score,
            prompt_score=prompt_score,
            failure_modes=failure_modes,
            comments=comments,
            timestamp=datetime.now().isoformat(),
        )
        
        self.human_judgments.append(judgment)
        
        logger.debug(f"Human judgment added: image_id={image_id}, judge_id={judge_id}")
    
    def add_metric_prediction(
        self,
        image_id: str,
        cultural_score: float,
        prompt_score: float,
        failure_modes: List[str],
        confidence: float = 0.0,
    ):
        """
        Add a metric prediction.
        
        Args:
            image_id: Image ID
            cultural_score: Predicted cultural score (0-10)
            prompt_score: Predicted prompt score (0-10)
            failure_modes: Predicted failure modes
            confidence: Prediction confidence
        """
        prediction = MetricPrediction(
            image_id=image_id,
            cultural_score=cultural_score,
            prompt_score=prompt_score,
            failure_modes=failure_modes,
            confidence=confidence,
        )
        
        self.metric_predictions.append(prediction)
        
        logger.debug(f"Metric prediction added: image_id={image_id}")
    
    def compute_correlation(self) -> Dict[str, Any]:
        """
        Compute correlation between human judgments and metric predictions.
        
        Returns:
            Dictionary with correlation metrics
        """
        # Match judgments and predictions by image_id
        matched_pairs = []
        
        for judgment in self.human_judgments:
            prediction = next(
                (p for p in self.metric_predictions if p.image_id == judgment.image_id),
                None,
            )
            
            if prediction is not None:
                matched_pairs.append((judgment, prediction))
        
        if len(matched_pairs) < 2:
            logger.warning("Not enough matched pairs for correlation")
            return {
                "num_pairs": len(matched_pairs),
                "cultural_pearson": None,
                "cultural_spearman": None,
                "prompt_pearson": None,
                "prompt_spearman": None,
            }
        
        # Extract scores
        human_cultural = [j.cultural_score for j, p in matched_pairs]
        metric_cultural = [p.cultural_score for j, p in matched_pairs]
        
        human_prompt = [j.prompt_score for j, p in matched_pairs]
        metric_prompt = [p.prompt_score for j, p in matched_pairs]
        
        # Compute correlations
        cultural_pearson, cultural_pearson_p = pearsonr(human_cultural, metric_cultural)
        cultural_spearman, cultural_spearman_p = spearmanr(human_cultural, metric_cultural)
        
        prompt_pearson, prompt_pearson_p = pearsonr(human_prompt, metric_prompt)
        prompt_spearman, prompt_spearman_p = spearmanr(human_prompt, metric_prompt)
        
        return {
            "num_pairs": len(matched_pairs),
            "cultural_pearson": float(cultural_pearson),
            "cultural_pearson_p": float(cultural_pearson_p),
            "cultural_spearman": float(cultural_spearman),
            "cultural_spearman_p": float(cultural_spearman_p),
            "prompt_pearson": float(prompt_pearson),
            "prompt_pearson_p": float(prompt_pearson_p),
            "prompt_spearman": float(prompt_spearman),
            "prompt_spearman_p": float(prompt_spearman_p),
        }
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save validation results to JSON.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_file = self.output_dir / filename
        
        correlation = self.compute_correlation()
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "num_judgments": len(self.human_judgments),
            "num_predictions": len(self.metric_predictions),
            "correlation": correlation,
            "judgments": [
                {
                    "image_id": j.image_id,
                    "judge_id": j.judge_id,
                    "cultural_score": j.cultural_score,
                    "prompt_score": j.prompt_score,
                    "failure_modes": j.failure_modes,
                }
                for j in self.human_judgments
            ],
            "predictions": [
                {
                    "image_id": p.image_id,
                    "cultural_score": p.cultural_score,
                    "prompt_score": p.prompt_score,
                    "failure_modes": p.failure_modes,
                    "confidence": p.confidence,
                }
                for p in self.metric_predictions
            ],
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation results saved: {results_file}")
        return results_file


def run_validation_study(
    human_judgments: List[HumanJudgment],
    metric_predictions: List[MetricPrediction],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run a validation study.
    
    Args:
        human_judgments: List of human judgments
        metric_predictions: List of metric predictions
        output_dir: Output directory
        
    Returns:
        Correlation results
    """
    protocol = HumanValidationProtocol(output_dir)
    
    for judgment in human_judgments:
        protocol.add_human_judgment(
            image_id=judgment.image_id,
            judge_id=judgment.judge_id,
            cultural_score=judgment.cultural_score,
            prompt_score=judgment.prompt_score,
            failure_modes=judgment.failure_modes,
            comments=judgment.comments,
        )
    
    for prediction in metric_predictions:
        protocol.add_metric_prediction(
            image_id=prediction.image_id,
            cultural_score=prediction.cultural_score,
            prompt_score=prediction.prompt_score,
            failure_modes=prediction.failure_modes,
            confidence=prediction.confidence,
        )
    
    correlation = protocol.compute_correlation()
    protocol.save_results()
    
    return correlation


def compute_correlation(
    human_scores: List[float],
    metric_scores: List[float],
) -> Dict[str, float]:
    """
    Compute correlation between human and metric scores.
    
    Args:
        human_scores: List of human scores
        metric_scores: List of metric scores
        
    Returns:
        Dictionary with correlation metrics
    """
    if len(human_scores) != len(metric_scores):
        raise ValueError("Human and metric scores must have same length")
    
    pearson_r, pearson_p = pearsonr(human_scores, metric_scores)
    spearman_r, spearman_p = spearmanr(human_scores, metric_scores)
    
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
