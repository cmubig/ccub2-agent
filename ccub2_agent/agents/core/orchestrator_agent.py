"""
Orchestrator Agent - Master controller for WorldCCUB Multi-Agent Loop.

Coordinates all specialized agents to execute the cultural improvement pipeline.
GPU strategy: VLM (Qwen3-VL-8B) on GPU 0, CLIP + Edit on GPU 1.
Models are created once and shared across agents to avoid duplicate loading.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import time
import gc

import torch

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from .scout_agent import ScoutAgent
from .edit_agent import EditAgent
from .judge_agent import JudgeAgent
from .job_agent import JobAgent
from .verification_agent import VerificationAgent

logger = logging.getLogger(__name__)


def _get_free_vram_gb(gpu_id: int = 1) -> float:
    """Get free VRAM in GB for the specified GPU (Fix 11)."""
    if not torch.cuda.is_available():
        return float('inf')  # No GPU constraints
    try:
        free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
        return free_bytes / (1024 ** 3)
    except Exception:
        return float('inf')  # Assume plenty of memory if we can't check

# Rollback + Edit Skip constants
EDIT_SKIP_THRESHOLD = 7.0  # Skip editing if score >= this threshold
MAX_CONSECUTIVE_DROPS = 4  # Fix 2: Increased from 2 to 4 for more recovery attempts

# Fix 11: VRAM threshold for conditional CLIP unloading (in GB)
VRAM_THRESHOLD_GB = 20.0  # Only unload CLIP if free VRAM is below this

# Fix 12: Edit retry parameters
MAX_EDIT_RETRIES = 3  # Number of retry attempts for edit failures

# Project data dirs
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_CLIP_INDEX_BASE = _PROJECT_ROOT / "data" / "clip_index"


class OrchestratorAgent(BaseAgent):
    """
    Master controller for the multi-agent loop.

    Responsibilities:
    - Initialize and coordinate all agents
    - Manage loop state and iteration tracking
    - Route tasks to appropriate agents
    - Handle loop termination conditions

    GPU allocation:
    - GPU 0: VLM (Qwen3-VL-8B) for Judge + Detection
    - GPU 1: CLIP (reference retrieval) + Edit model
    Models are loaded once and shared across agents.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # ---- Load shared models (each on its own GPU) ----
        t_start = time.time()

        # 1. CLIP RAG on GPU 1 (small, loads fast)
        logger.info("Loading CLIP RAG → GPU 1...")
        shared_clip = None
        clip_index_dir = _CLIP_INDEX_BASE / config.country
        if clip_index_dir.exists():
            from ...retrieval.clip_image_rag import CLIPImageRAG
            shared_clip = CLIPImageRAG(index_dir=clip_index_dir, device="auto")
            logger.info(f"CLIP RAG loaded on GPU 1")
        else:
            logger.warning(f"No CLIP index for {config.country}")

        # 2. VLM detector on GPU 0 (heavy, takes longer)
        logger.info("Loading VLM (Qwen3-VL-8B) → GPU 0...")
        from ...detection.vlm_detector import VLMCulturalDetector
        cultural_index_dir = _PROJECT_ROOT / "data" / "cultural_index" / config.country
        shared_vlm = VLMCulturalDetector(
            load_in_4bit=True,
            index_dir=cultural_index_dir if cultural_index_dir.exists() else None,
            clip_index_dir=clip_index_dir if clip_index_dir.exists() else None,
        )

        logger.info(f"All models loaded in {time.time()-t_start:.1f}s")

        # ---- Initialize sub-agents with shared models ----
        self.clip_rag = shared_clip  # Keep reference for unload/reload around edit phase
        self.vlm_detector = shared_vlm  # Keep reference for sharing with verification agent
        self.judge_agent = JudgeAgent(config, shared_vlm_detector=shared_vlm)
        self.scout_agent = ScoutAgent(config, shared_clip_rag=shared_clip)
        self.edit_agent = EditAgent(config)
        self.job_agent = JobAgent(config)
        self.verification_agent = VerificationAgent(config, shared_vlm_detector=shared_vlm)

        # Loop state
        self.current_iteration = 0
        self.max_iterations = 5
        self.score_threshold = 8.0
        self.score_history: List[float] = []
        self.consecutive_drops: int = 0  # Track consecutive score drops for rollback

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
            self.consecutive_drops = 0

            current_image = Path(input_data["image_path"])
            prompt = input_data["prompt"]

            # Fix 10: Clear scout reference cache for new image
            self.scout_agent.clear_cache()

            # Phase 1: Initial evaluation
            # Fix 4: Reset judge momentum for new image sequence
            self.judge_agent.reset_momentum()

            judge_input = {
                "image_path": str(current_image),
                "prompt": prompt,
                "country": self.config.country,
                "category": input_data.get("category"),
                "reset_momentum": True  # First evaluation resets momentum
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

                # Edit Skip: Don't edit images that are already good enough
                if cultural_score >= EDIT_SKIP_THRESHOLD:
                    logger.info(f"Score {cultural_score:.1f} >= {EDIT_SKIP_THRESHOLD}, skipping edit (already good)")
                    break

                # Phase 3: Gap detection + Reference retrieval
                scout_input = {
                    "image_path": str(current_image),
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

                # Phase 4.5: Reference Verification
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
                        references = verification_result.data.get("verified_references", [])
                        logger.info(f"Verified {len(references)} references (filtered {verification_result.data.get('filtered_count', 0)})")

                # Phase 5: Edit — Fix 11: Only unload CLIP if VRAM is low
                clip_was_unloaded = False
                free_vram = _get_free_vram_gb(gpu_id=1)
                if self.clip_rag is not None and free_vram < VRAM_THRESHOLD_GB:
                    logger.info(f"Free VRAM {free_vram:.1f}GB < {VRAM_THRESHOLD_GB}GB, unloading CLIP")
                    self.clip_rag.unload()
                    clip_was_unloaded = True
                else:
                    logger.debug(f"Free VRAM {free_vram:.1f}GB >= {VRAM_THRESHOLD_GB}GB, keeping CLIP loaded")

                # Fix 7: Pass structured knowledge to Edit for targeted corrections
                item_knowledge = None
                if self.vlm_detector.structured_knowledge:
                    item_id = Path(input_data["image_path"]).stem
                    item_knowledge = self.vlm_detector.structured_knowledge.get(item_id)

                edit_input = {
                    "image_path": str(current_image),
                    "prompt": prompt,
                    "issues": judge_result.data.get("issues", []),
                    "references": [r.get("image_path") for r in references] if references else [],
                    "country": self.config.country,
                    "category": input_data.get("category"),
                    "item_knowledge": item_knowledge,
                    # Fix 5 & 7: Pass cultural score and iteration for adaptive editing
                    "cultural_score": cultural_score,
                    "iteration_number": iteration,
                }

                # Fix 12: Edit with retry logic for OOM handling
                edit_result = None
                for attempt in range(MAX_EDIT_RETRIES):
                    try:
                        edit_result = self.edit_agent.execute(edit_input)
                        if edit_result.success:
                            break
                        # Non-OOM failure, don't retry
                        if "out of memory" not in edit_result.message.lower():
                            break
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(f"Edit OOM at attempt {attempt + 1}/{MAX_EDIT_RETRIES}, clearing cache...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            # Force unload CLIP on OOM
                            if self.clip_rag is not None and not clip_was_unloaded:
                                self.clip_rag.unload()
                                clip_was_unloaded = True
                            continue
                        raise

                # Reload CLIP if it was unloaded (needed for next iteration's retrieval)
                if clip_was_unloaded and self.clip_rag is not None:
                    self.clip_rag.reload()

                if edit_result is None or not edit_result.success:
                    logger.warning(f"Edit failed at iteration {iteration} after {MAX_EDIT_RETRIES} attempts")
                    break

                edited_image = Path(edit_result.data["output_image"])

                # Phase 6: Re-evaluation
                judge_input["image_path"] = str(edited_image)
                judge_result = self.judge_agent.execute(judge_input)

                if not judge_result.success:
                    break

                new_score = judge_result.data.get("cultural_score", 0)
                prev_score = self.score_history[-1]

                # Rollback: If score dropped, keep the previous image
                if new_score < prev_score:
                    self.consecutive_drops += 1
                    logger.warning(
                        f"Score dropped {prev_score:.1f} → {new_score:.1f}, "
                        f"rolling back (consecutive drops: {self.consecutive_drops})"
                    )

                    # Stop if too many consecutive drops
                    if self.consecutive_drops >= MAX_CONSECUTIVE_DROPS:
                        logger.info(
                            f"{MAX_CONSECUTIVE_DROPS} consecutive drops, stopping loop early"
                        )
                        break

                    # Don't update current_image (rollback), but record the drop
                    self.score_history.append(prev_score)  # Keep tracking with prev score
                    continue

                # Score improved or stayed same: accept the edit
                self.consecutive_drops = 0
                current_image = edited_image
                cultural_score = new_score
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
