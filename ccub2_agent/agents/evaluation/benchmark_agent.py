"""
Benchmark Agent - CultureBench-Global execution.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ..core.orchestrator_agent import OrchestratorAgent
from ..core.judge_agent import JudgeAgent
from .metric_agent import MetricAgent

logger = logging.getLogger(__name__)


class BenchmarkAgent(BaseAgent):
    """
    Runs CultureBench-Global benchmark tasks.
    
    Responsibilities:
    - Execute standardized evaluation protocols
    - Generate reproducible reports
    - Run ablation studies
    - Compare baseline variants
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.judge_agent = JudgeAgent(config)
        self.metric_agent = MetricAgent(config)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute benchmark evaluation.
        
        Args:
            input_data: {
                "task": str,  # "fidelity" | "degradation" | "transfer" | "portrait" | "all"
                "split": str,  # "train" | "val" | "test"
                "variants": List[str],  # ["no_correction", "retrieval_only", "single_agent", "multi_agent"]
                "output_dir": str (optional)
            }
            
        Returns:
            AgentResult with benchmark results and report
        """
        try:
            task = input_data.get("task", "all")
            split = input_data.get("split", "test")
            variants = input_data.get("variants", ["no_correction", "multi_agent"])
            output_dir = Path(input_data.get("output_dir", self.config.output_dir or Path("results/benchmark")))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = {}
            
            # Task A: Cultural Fidelity Scoring
            if task in ["fidelity", "all"]:
                results["task_a_fidelity"] = self._run_fidelity_task(split, variants, output_dir)
            
            # Task B: Iterative I2I Degradation
            if task in ["degradation", "all"]:
                results["task_b_degradation"] = self._run_degradation_task(split, variants, output_dir)
            
            # Task C: Cross-Country Transfer
            if task in ["transfer", "all"]:
                results["task_c_transfer"] = self._run_transfer_task(split, variants, output_dir)
            
            # Task D: Portrait Editing (optional)
            if task in ["portrait", "all"]:
                results["task_d_portrait"] = self._run_portrait_task(split, variants, output_dir)
            
            # Generate report
            report_path = self._generate_report(results, output_dir, task, split)
            
            return AgentResult(
                success=True,
                data={
                    "results": results,
                    "report_path": str(report_path),
                    "tasks_completed": list(results.keys()),
                    "variants_tested": variants
                },
                message=f"Benchmark completed: {len(results)} tasks"
            )
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Benchmark error: {str(e)}"
            )
    
    def _run_fidelity_task(
        self, split: str, variants: List[str], output_dir: Path
    ) -> Dict[str, Any]:
        """Task A: Cultural Fidelity Scoring."""
        logger.info("Running Task A: Cultural Fidelity Scoring")
        
        # Load test samples for this split
        samples = self._load_benchmark_samples(split, task="fidelity")
        
        variant_results = {}
        for variant in variants:
            scores = []
            for sample in samples[:10]:  # Limit for testing
                if variant == "no_correction":
                    # Direct evaluation without correction
                    result = self.metric_agent.execute({
                        "image_path": sample["image_path"],
                        "prompt": sample["prompt"],
                        "country": sample["country"],
                        "category": sample.get("category")
                    })
                elif variant == "multi_agent":
                    # Full multi-agent loop
                    orchestrator = OrchestratorAgent(self.config)
                    result = orchestrator.execute({
                        "image_path": sample["image_path"],
                        "prompt": sample["prompt"],
                        "country": sample["country"],
                        "category": sample.get("category"),
                        "max_iterations": 5
                    })
                    # Get final score from result
                    if result.success:
                        final_image = result.data.get("final_image")
                        metric_result = self.metric_agent.execute({
                            "image_path": final_image,
                            "prompt": sample["prompt"],
                            "country": sample["country"]
                        })
                        result = metric_result
                else:
                    continue
                
                if result.success:
                    scores.append(result.data.get("cultural_score", 0))
            
            if scores:
                variant_results[variant] = {
                    "mean_score": sum(scores) / len(scores),
                    "std_score": self._calculate_std(scores),
                    "samples": len(scores)
                }
        
        return variant_results
    
    def _run_degradation_task(
        self, split: str, variants: List[str], output_dir: Path
    ) -> Dict[str, Any]:
        """Task B: Iterative I2I Degradation Test."""
        logger.info("Running Task B: Iterative I2I Degradation")
        
        # Load seed images
        samples = self._load_benchmark_samples(split, task="degradation")
        
        results = {}
        for sample in samples[:5]:  # Limit for testing
            seed_image = sample["image_path"]
            trajectory = []
            
            # Run multiple edit iterations
            current_image = seed_image
            for iteration in range(5):
                # Evaluate current state
                metric_result = self.metric_agent.execute({
                    "image_path": current_image,
                    "prompt": sample["prompt"],
                    "country": sample["country"],
                    "iteration_number": iteration
                })
                
                if metric_result.success:
                    score = metric_result.data.get("cultural_score", 0)
                    trajectory.append(score)
                
                # Apply next edit (simplified - would use EditAgent)
                # For now, just track trajectory
                break  # Simplified
            
            results[sample["id"]] = {
                "trajectory": trajectory,
                "degradation_rate": self._calculate_degradation_rate(trajectory)
            }
        
        return results
    
    def _run_transfer_task(
        self, split: str, variants: List[str], output_dir: Path
    ) -> Dict[str, Any]:
        """Task C: Cross-Country Restylization."""
        logger.info("Running Task C: Cross-Country Transfer")
        
        # Load transfer pairs (source_country -> target_country)
        samples = self._load_benchmark_samples(split, task="transfer")
        
        results = {}
        for sample in samples[:5]:  # Limit for testing
            source_country = sample["source_country"]
            target_country = sample["target_country"]
            
            # Evaluate source
            source_result = self.metric_agent.execute({
                "image_path": sample["image_path"],
                "prompt": sample["prompt"],
                "country": source_country
            })
            
            # Evaluate transfer (would use EditAgent for actual transfer)
            transfer_result = self.metric_agent.execute({
                "image_path": sample.get("transferred_image", sample["image_path"]),
                "prompt": sample["prompt"],
                "country": target_country
            })
            
            if source_result.success and transfer_result.success:
                results[sample["id"]] = {
                    "source_score": source_result.data.get("cultural_score", 0),
                    "transfer_score": transfer_result.data.get("cultural_score", 0),
                    "faithfulness": self._calculate_faithfulness(source_result, transfer_result)
                }
        
        return results
    
    def _run_portrait_task(
        self, split: str, variants: List[str], output_dir: Path
    ) -> Dict[str, Any]:
        """Task D: Portrait Editing Robustness (optional)."""
        logger.info("Running Task D: Portrait Editing")
        
        # Load portrait samples
        samples = self._load_benchmark_samples(split, task="portrait")
        
        results = {}
        for sample in samples[:5]:  # Limit for testing
            # Evaluate for demographic misrepresentation
            result = self.metric_agent.execute({
                "image_path": sample["image_path"],
                "prompt": sample["prompt"],
                "country": sample["country"],
                "category": "portrait"
            })
            
            if result.success:
                failure_modes = result.data.get("failure_modes", [])
                demographic_failures = [
                    fm for fm in failure_modes
                    if "demographic" in str(fm).lower() or "identity" in str(fm).lower()
                ]
                
                results[sample["id"]] = {
                    "cultural_score": result.data.get("cultural_score", 0),
                    "demographic_failures": len(demographic_failures),
                    "failure_details": demographic_failures
                }
        
        return results
    
    def _load_benchmark_samples(self, split: str, task: str) -> List[Dict[str, Any]]:
        """Load benchmark samples for a task and split."""
        # In real implementation, load from benchmark dataset
        # For now, return empty list (would load from data/benchmark/{task}/{split}.jsonl)
        benchmark_dir = Path(__file__).parent.parent.parent.parent / "data" / "benchmark"
        sample_file = benchmark_dir / task / f"{split}.jsonl"
        
        if sample_file.exists():
            samples = []
            with open(sample_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            return samples
        
        # Return empty for now (benchmark data not yet created)
        logger.warning(f"Benchmark samples not found at {sample_file}, returning empty")
        return []
    
    def _calculate_std(self, scores: List[float]) -> float:
        """Calculate standard deviation."""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5
    
    def _calculate_degradation_rate(self, trajectory: List[float]) -> float:
        """Calculate degradation rate from score trajectory."""
        if len(trajectory) < 2:
            return 0.0
        return (trajectory[0] - trajectory[-1]) / len(trajectory)
    
    def _calculate_faithfulness(
        self, source_result: AgentResult, transfer_result: AgentResult
    ) -> float:
        """Calculate faithfulness in cross-country transfer."""
        # Simplified: compare how well non-cultural attributes preserved
        # Would need more sophisticated comparison
        return 0.8  # Placeholder
    
    def _generate_report(
        self, results: Dict[str, Any], output_dir: Path, task: str, split: str
    ) -> Path:
        """Generate benchmark report."""
        report = {
            "benchmark": "CultureBench-Global",
            "version": "1.0.0",
            "task": task,
            "split": split,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        report_path = output_dir / f"benchmark_report_{task}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark report saved to {report_path}")
        return report_path
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {}
        for task_name, task_results in results.items():
            if isinstance(task_results, dict) and "mean_score" in list(task_results.values())[0] if task_results else False:
                # Aggregate variant results
                variant_scores = {
                    variant: data.get("mean_score", 0)
                    for variant, data in task_results.items()
                }
                summary[task_name] = {
                    "best_variant": max(variant_scores, key=variant_scores.get) if variant_scores else None,
                    "scores": variant_scores
                }
        return summary
