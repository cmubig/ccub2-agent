"""
Ablation Runner

Systematic ablation study runner for NeurIPS paper.
Implements 4 core variants:
1. no_correction: Baseline (no cultural correction)
2. retrieval_only: Only retrieval, no editing
3. single_agent: Single agent (no loop)
4. multi_agent_loop: Full multi-agent loop
"""

from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
import json
from datetime import datetime

from ...orchestrator_agent import OrchestratorAgent
from ...base_agent import AgentConfig, AgentResult
from ....orchestration.logging import DecisionLogger, get_decision_logger

logger = logging.getLogger(__name__)


class AblationVariant(str, Enum):
    """Ablation study variants."""
    NO_CORRECTION = "no_correction"
    RETRIEVAL_ONLY = "retrieval_only"
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT_LOOP = "multi_agent_loop"


@dataclass
class AblationResult:
    """Results from a single ablation variant run."""
    variant: AblationVariant
    initial_score: float
    final_score: float
    score_gain: float
    regression_rate: float  # How often score decreased
    coverage_gain: float  # How many gaps were filled
    num_iterations: int
    execution_time: float  # seconds
    config: Dict[str, Any]
    timestamp: datetime


class AblationRunner:
    """
    Runs systematic ablation studies.
    
    Essential for NeurIPS paper ablation table.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize ablation runner.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.decision_logger = get_decision_logger(
            log_dir=self.output_dir / "logs"
        )
        
        logger.info(f"AblationRunner initialized: output_dir={output_dir}")
    
    def get_variant_config(self, variant: AblationVariant) -> Dict[str, Any]:
        """
        Get configuration for a specific variant.
        
        Args:
            variant: Ablation variant
            
        Returns:
            Configuration dict for this variant
        """
        configs = {
            AblationVariant.NO_CORRECTION: {
                "use_detection": False,
                "use_retrieval": False,
                "use_editing": False,
                "use_loop": False,
                "max_iterations": 0,
            },
            AblationVariant.RETRIEVAL_ONLY: {
                "use_detection": True,
                "use_retrieval": True,
                "use_editing": False,
                "use_loop": False,
                "max_iterations": 0,
            },
            AblationVariant.SINGLE_AGENT: {
                "use_detection": True,
                "use_retrieval": True,
                "use_editing": True,
                "use_loop": False,
                "max_iterations": 1,
            },
            AblationVariant.MULTI_AGENT_LOOP: {
                "use_detection": True,
                "use_retrieval": True,
                "use_editing": True,
                "use_loop": True,
                "max_iterations": 5,
            },
        }
        
        return configs.get(variant, {})
    
    def run_variant(
        self,
        variant: AblationVariant,
        input_data: Dict[str, Any],
        agent_config: AgentConfig,
    ) -> AblationResult:
        """
        Run a single ablation variant.
        
        Args:
            variant: Variant to run
            input_data: Input data (image_path, prompt, country, etc.)
            agent_config: Agent configuration
            
        Returns:
            Ablation result
        """
        import time
        
        logger.info(f"Running ablation variant: {variant.value}")
        
        variant_config = self.get_variant_config(variant)
        start_time = time.time()
        
        # Get initial score
        from ....detection.vlm_detector import VLMCulturalDetector
        vlm_detector = VLMCulturalDetector(load_in_4bit=True)
        
        initial_score, _ = vlm_detector.score_cultural_quality(
            image_path=Path(input_data["image_path"]),
            prompt=input_data["prompt"],
            country=input_data.get("country", agent_config.country),
            category=input_data.get("category", agent_config.category),
        )
        
        # Run variant
        if variant == AblationVariant.NO_CORRECTION:
            # No correction, just return initial
            final_score = initial_score
            num_iterations = 0
            coverage_gain = 0.0
            regression_rate = 0.0
            
        elif variant == AblationVariant.RETRIEVAL_ONLY:
            # Only retrieval, no editing
            from ....retrieval.clip_image_rag import create_clip_rag
            clip_rag = create_clip_rag(
                country=agent_config.country,
                category=agent_config.category,
            )
            
            references = clip_rag.retrieve(
                query_image=Path(input_data["image_path"]),
                top_k=3,
            )
            
            # Score doesn't change (no editing), but we track coverage
            final_score = initial_score
            num_iterations = 0
            coverage_gain = 1.0 if len(references) > 0 else 0.0
            regression_rate = 0.0
            
        elif variant == AblationVariant.SINGLE_AGENT:
            # Single iteration (no loop)
            orchestrator = OrchestratorAgent(agent_config)
            orchestrator.max_iterations = 1
            
            result = orchestrator.execute(input_data)
            
            if result.success and "final_image" in result.data:
                final_image_path = result.data["final_image"]
                final_score, _ = vlm_detector.score_cultural_quality(
                    image_path=Path(final_image_path),
                    prompt=input_data["prompt"],
                    country=input_data.get("country", agent_config.country),
                    category=input_data.get("category", agent_config.category),
                )
            else:
                final_score = initial_score
            
            num_iterations = 1
            coverage_gain = result.data.get("coverage_gain", 0.0)
            regression_rate = 1.0 if final_score < initial_score else 0.0
            
        else:  # MULTI_AGENT_LOOP
            # Full multi-agent loop
            orchestrator = OrchestratorAgent(agent_config)
            orchestrator.max_iterations = variant_config["max_iterations"]
            
            result = orchestrator.execute(input_data)
            
            if result.success and "final_image" in result.data:
                final_image_path = result.data["final_image"]
                final_score, _ = vlm_detector.score_cultural_quality(
                    image_path=Path(final_image_path),
                    prompt=input_data["prompt"],
                    country=input_data.get("country", agent_config.country),
                    category=input_data.get("category", agent_config.category),
                )
            else:
                final_score = initial_score
            
            num_iterations = result.data.get("iterations", 0)
            coverage_gain = result.data.get("coverage_gain", 0.0)
            
            # Calculate regression rate from score history
            score_history = result.data.get("score_history", [])
            if len(score_history) > 1:
                regressions = sum(
                    1 for i in range(1, len(score_history))
                    if score_history[i] < score_history[i-1]
                )
                regression_rate = regressions / (len(score_history) - 1)
            else:
                regression_rate = 0.0
        
        execution_time = time.time() - start_time
        score_gain = final_score - initial_score
        
        result = AblationResult(
            variant=variant,
            initial_score=initial_score,
            final_score=final_score,
            score_gain=score_gain,
            regression_rate=regression_rate,
            coverage_gain=coverage_gain,
            num_iterations=num_iterations,
            execution_time=execution_time,
            config=variant_config,
            timestamp=datetime.now(),
        )
        
        logger.info(
            f"Variant {variant.value} completed: "
            f"score_gain={score_gain:.2f}, iterations={num_iterations}"
        )
        
        return result
    
    def run_ablation_study(
        self,
        input_data: Dict[str, Any],
        agent_config: AgentConfig,
        variants: Optional[List[AblationVariant]] = None,
    ) -> Dict[AblationVariant, AblationResult]:
        """
        Run full ablation study across all variants.
        
        Args:
            input_data: Input data for all variants
            agent_config: Agent configuration
            variants: List of variants to run. If None, runs all.
            
        Returns:
            Dictionary mapping variants to results
        """
        if variants is None:
            variants = list(AblationVariant)
        
        results = {}
        
        for variant in variants:
            try:
                result = self.run_variant(variant, input_data, agent_config)
                results[variant] = result
            except Exception as e:
                logger.error(f"Error running variant {variant.value}: {e}", exc_info=True)
                # Continue with other variants
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict[AblationVariant, AblationResult]):
        """Save ablation results to JSON."""
        results_file = self.output_dir / f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "variants": {
                variant.value: {
                    "initial_score": result.initial_score,
                    "final_score": result.final_score,
                    "score_gain": result.score_gain,
                    "regression_rate": result.regression_rate,
                    "coverage_gain": result.coverage_gain,
                    "num_iterations": result.num_iterations,
                    "execution_time": result.execution_time,
                    "config": result.config,
                }
                for variant, result in results.items()
            },
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ablation results saved: {results_file}")
        
        # Also save decision logs
        self.decision_logger.save_log()


def run_ablation_study(
    input_data: Dict[str, Any],
    agent_config: AgentConfig,
    output_dir: Path,
    variants: Optional[List[AblationVariant]] = None,
) -> Dict[AblationVariant, AblationResult]:
    """
    Convenience function to run ablation study.
    
    Args:
        input_data: Input data
        agent_config: Agent configuration
        output_dir: Output directory
        variants: Variants to run (None = all)
        
    Returns:
        Dictionary of results
    """
    runner = AblationRunner(output_dir)
    return runner.run_ablation_study(input_data, agent_config, variants)


def get_variant_config(variant: AblationVariant) -> Dict[str, Any]:
    """Get configuration for a variant."""
    runner = AblationRunner(Path("."))
    return runner.get_variant_config(variant)
