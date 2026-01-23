"""
Decision Logger

Tracks all decisions made by agents and the system.
Essential for answering "why was this choice made?" in NeurIPS reviews.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionReason(str, Enum):
    """Types of reasons for decisions."""
    SCORE_THRESHOLD = "score_threshold"
    CONFIDENCE_LOW = "confidence_low"
    REFERENCE_AVAILABLE = "reference_available"
    REFERENCE_MISSING = "reference_missing"
    ITERATION_LIMIT = "iteration_limit"
    COVERAGE_GAP = "coverage_gap"
    MODEL_CAPABILITY = "model_capability"
    USER_PREFERENCE = "user_preference"
    DEFAULT = "default"


@dataclass
class DecisionLogEntry:
    """
    A single decision log entry.
    
    Records what decision was made, by whom, when, and why.
    """
    timestamp: datetime
    agent_name: str
    decision_type: str  # e.g., "STOP", "ITERATE", "SELECT_REFERENCE"
    decision_value: Any  # The actual decision (e.g., selected reference path)
    reason: DecisionReason
    context: Dict[str, Any]  # Additional context (scores, options considered, etc.)
    input_data: Optional[Dict[str, Any]] = None  # What triggered this decision
    output_data: Optional[Dict[str, Any]] = None  # What was produced
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class DecisionLogger:
    """
    Centralized decision logging system.
    
    Tracks all decisions made by agents for full transparency and reproducibility.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize decision logger.
        
        Args:
            log_dir: Directory to save logs. If None, uses current directory.
        """
        self.log_dir = log_dir or Path("logs/decisions")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.entries: List[DecisionLogEntry] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"DecisionLogger initialized: session_id={self.session_id}")
    
    def log_decision(
        self,
        agent_name: str,
        decision_type: str,
        decision_value: Any,
        reason: DecisionReason,
        context: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
    ) -> DecisionLogEntry:
        """
        Log a decision made by an agent.
        
        Args:
            agent_name: Name of the agent making the decision
            decision_type: Type of decision (e.g., "STOP", "ITERATE")
            decision_value: The actual decision value
            reason: Why this decision was made
            context: Additional context (scores, options, etc.)
            input_data: Input that triggered this decision
            output_data: Output produced by this decision
            
        Returns:
            The created log entry
        """
        entry = DecisionLogEntry(
            timestamp=datetime.now(),
            agent_name=agent_name,
            decision_type=decision_type,
            decision_value=decision_value,
            reason=reason,
            context=context or {},
            input_data=input_data,
            output_data=output_data,
        )
        
        self.entries.append(entry)
        
        logger.debug(
            f"Decision logged: {agent_name} -> {decision_type} "
            f"({reason.value})"
        )
        
        return entry
    
    def log_loop_iteration(
        self,
        iteration: int,
        cultural_score: float,
        decision: str,
        agent_decisions: List[Dict[str, Any]],
    ):
        """
        Log a complete loop iteration.
        
        Args:
            iteration: Iteration number
            cultural_score: Final cultural score
            decision: Final decision (STOP/ITERATE)
            agent_decisions: List of decisions made by each agent
        """
        entry = self.log_decision(
            agent_name="OrchestratorAgent",
            decision_type=f"LOOP_ITERATION_{iteration}",
            decision_value=decision,
            reason=DecisionReason.SCORE_THRESHOLD if decision == "STOP" else DecisionReason.ITERATION_LIMIT,
            context={
                "iteration": iteration,
                "cultural_score": cultural_score,
                "agent_decisions": agent_decisions,
            },
        )
        
        logger.info(
            f"Loop iteration {iteration} logged: score={cultural_score:.2f}, "
            f"decision={decision}"
        )
    
    def log_reference_selection(
        self,
        agent_name: str,
        selected_references: List[str],
        scores: List[float],
        reason: DecisionReason,
    ):
        """
        Log reference selection decision.
        
        Args:
            agent_name: Agent that selected references
            selected_references: List of selected reference paths
            scores: Similarity scores for each reference
            reason: Why these references were selected
        """
        self.log_decision(
            agent_name=agent_name,
            decision_type="SELECT_REFERENCE",
            decision_value=selected_references,
            reason=reason,
            context={
                "num_references": len(selected_references),
                "scores": scores,
            },
        )
    
    def log_model_selection(
        self,
        selected_model: str,
        available_models: List[str],
        reason: DecisionReason,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log model selection decision.
        
        Args:
            selected_model: Selected I2I model
            available_models: List of available models
            reason: Why this model was selected
            context: Additional context
        """
        self.log_decision(
            agent_name="EditAgent",
            decision_type="SELECT_MODEL",
            decision_value=selected_model,
            reason=reason,
            context={
                "available_models": available_models,
                **(context or {}),
            },
        )
    
    def save_log(self, filename: Optional[str] = None) -> Path:
        """
        Save all logged decisions to a JSON file.
        
        Args:
            filename: Optional filename. If None, uses session_id.
            
        Returns:
            Path to saved log file
        """
        if filename is None:
            filename = f"decisions_{self.session_id}.json"
        
        log_file = self.log_dir / filename
        
        log_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "num_entries": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Decision log saved: {log_file}")
        return log_file
    
    def get_decisions_by_agent(self, agent_name: str) -> List[DecisionLogEntry]:
        """Get all decisions made by a specific agent."""
        return [e for e in self.entries if e.agent_name == agent_name]
    
    def get_decisions_by_type(self, decision_type: str) -> List[DecisionLogEntry]:
        """Get all decisions of a specific type."""
        return [e for e in self.entries if e.decision_type == decision_type]
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get full decision history as list of dicts."""
        return [entry.to_dict() for entry in self.entries]


# Global logger instance
_global_logger: Optional[DecisionLogger] = None


def get_decision_logger(log_dir: Optional[Path] = None) -> DecisionLogger:
    """Get or create global decision logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = DecisionLogger(log_dir=log_dir)
    return _global_logger


def log_agent_decision(
    agent_name: str,
    decision_type: str,
    decision_value: Any,
    reason: DecisionReason,
    context: Optional[Dict[str, Any]] = None,
) -> DecisionLogEntry:
    """Convenience function to log a decision using global logger."""
    logger = get_decision_logger()
    return logger.log_decision(
        agent_name=agent_name,
        decision_type=decision_type,
        decision_value=decision_value,
        reason=reason,
        context=context,
    )


def log_loop_iteration(
    iteration: int,
    cultural_score: float,
    decision: str,
    agent_decisions: List[Dict[str, Any]],
):
    """Convenience function to log a loop iteration."""
    logger = get_decision_logger()
    logger.log_loop_iteration(
        iteration=iteration,
        cultural_score=cultural_score,
        decision=decision,
        agent_decisions=agent_decisions,
    )


def log_benchmark_run(
    benchmark_name: str,
    variant: str,
    results: Dict[str, Any],
    config: Dict[str, Any],
):
    """Log a benchmark run for reproducibility."""
    logger = get_decision_logger()
    logger.log_decision(
        agent_name="BenchmarkAgent",
        decision_type="BENCHMARK_RUN",
        decision_value=results,
        reason=DecisionReason.DEFAULT,
        context={
            "benchmark_name": benchmark_name,
            "variant": variant,
            "config": config,
        },
    )
