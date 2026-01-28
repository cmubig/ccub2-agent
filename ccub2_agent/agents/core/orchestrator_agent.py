"""
Orchestrator Agent - Master controller for WorldCCUB Multi-Agent Loop.

Coordinates all specialized agents to execute the cultural improvement pipeline.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from .scout_agent import ScoutAgent
from .edit_agent import EditAgent
from .judge_agent import JudgeAgent
from .job_agent import JobAgent
from .verification_agent import VerificationAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Master controller for the multi-agent loop.
    
    Responsibilities:
    - Initialize and coordinate all agents
    - Manage loop state and iteration tracking
    - Route tasks to appropriate agents
    - Handle loop termination conditions
    
    Enhanced with VerificationAgent (MA-RAG pattern) for reference verification.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize sub-agents
        self.scout_agent = ScoutAgent(config)
        self.edit_agent = EditAgent(config)
        self.judge_agent = JudgeAgent(config)
        self.job_agent = JobAgent(config)
        self.verification_agent = VerificationAgent(config)  # NEW: Reference verification
        
        # Loop state
        self.current_iteration = 0
        self.max_iterations = 5
        self.score_threshold = 8.0
        self.score_history: List[float] = []
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the full multi-agent loop.
        
        Args:
            input_data: {
                "image_path": str,
                "prompt": str,
                "country": str,
                "category": str,
                "max_iterations": int (optional),
                "score_threshold": float (optional)
            }
            
        Returns:
            AgentResult with final image, scores, and iteration history
        """
        try:
            # Initialize loop state
            self.current_iteration = 0
            self.max_iterations = input_data.get("max_iterations", 5)
            self.score_threshold = input_data.get("score_threshold", 8.0)
            self.score_history = []
            
            current_image = Path(input_data["image_path"])
            prompt = input_data["prompt"]
            
            # Phase 1: Initial evaluation
            judge_input = {
                "image_path": str(current_image),
                "prompt": prompt,
                "country": self.config.country,
                "category": input_data.get("category")
            }
            
            judge_result = self.judge_agent.execute(judge_input)
            if not judge_result.success:
                return AgentResult(
                    success=False,
                    data={},
                    message=f"Initial evaluation failed: {judge_result.message}"
                )
            
            cultural_score = judge_result.data.get("cultural_score", 0)
            self.score_history.append(cultural_score)
            
            # Phase 2-6: Iterative improvement loop
            for iteration in range(1, self.max_iterations + 1):
                self.current_iteration = iteration
                
                # Check termination conditions
                if cultural_score >= self.score_threshold:
                    logger.info(f"Target score reached at iteration {iteration}")
                    break
                
                # Phase 3: Gap detection + Reference retrieval
                scout_input = {
                    "image_path": str(current_image),  # Query image for CLIP RAG
                    "failure_modes": judge_result.data.get("failure_modes", []),
                    "country": self.config.country,
                    "category": input_data.get("category")
                }
                scout_result = self.scout_agent.execute(scout_input)
                
                # Phase 4: Job creation (if needed)
                if scout_result.data.get("needs_more_data", False):
                    job_input = {
                        "country": self.config.country,
                        "category": scout_result.data.get("category"),
                        "missing_elements": scout_result.data.get("missing_elements", [])
                    }
                    self.job_agent.execute(job_input)
                
                # Phase 4.5: Reference Verification (NEW)
                # Get references from Scout (would need to integrate ReferenceSelector)
                # For now, use issues to get references
                references = scout_result.data.get("references", [])
                
                if references:
                    verification_input = {
                        "references": references,
                        "failure_modes": judge_result.data.get("failure_modes", []),
                        "prompt": prompt,
                        "country": self.config.country,
                        "category": input_data.get("category"),
                        "original_image_path": str(current_image)
                    }
                    verification_result = self.verification_agent.execute(verification_input)
                    
                    if verification_result.success:
                        # Use only verified references
                        references = verification_result.data.get("verified_references", [])
                        logger.info(f"Verified {len(references)} references (filtered {verification_result.data.get('filtered_count', 0)})")
                
                # Phase 5: Edit
                edit_input = {
                    "image_path": str(current_image),
                    "prompt": prompt,
                    "issues": judge_result.data.get("issues", []),
                    "references": [r.get("image_path") for r in references] if references else [],
                    "country": self.config.country
                }
                edit_result = self.edit_agent.execute(edit_input)
                
                if not edit_result.success:
                    logger.warning(f"Edit failed at iteration {iteration}")
                    break
                
                current_image = Path(edit_result.data["output_image"])
                
                # Phase 6: Re-evaluation
                judge_input["image_path"] = str(current_image)
                judge_result = self.judge_agent.execute(judge_input)
                
                if not judge_result.success:
                    break
                
                cultural_score = judge_result.data.get("cultural_score", 0)
                self.score_history.append(cultural_score)
            
            # Final result
            return AgentResult(
                success=True,
                data={
                    "final_image": str(current_image),
                    "final_score": cultural_score,
                    "iterations": self.current_iteration,
                    "score_history": self.score_history,
                    "improvement": self.score_history[-1] - self.score_history[0] if len(self.score_history) > 1 else 0
                },
                message=f"Loop completed after {self.current_iteration} iterations"
            )
            
        except Exception as e:
            logger.error(f"Orchestrator execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Execution error: {str(e)}"
            )
