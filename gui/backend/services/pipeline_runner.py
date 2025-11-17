"""
Pipeline Runner Service

This service orchestrates the execution of the CCUB2 pipeline
and broadcasts updates via WebSocket.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import uuid
import sys
import base64
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Add ccub2_agent to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.pipeline import PipelineConfig, PipelineState, PipelineStatus
from models.node import NodeData, NodeStatus, NodeType, NodePosition
from models.history import PipelineHistory, NodeHistory, PromptFlowStep, RAGResult
from api.websocket import manager
from config import get_session_dir
from services.history_manager import get_history_manager

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Manages pipeline execution and state

    This class runs the CCUB2 pipeline and broadcasts real-time updates
    to connected WebSocket clients.
    """

    def __init__(self):
        self.state: PipelineState = PipelineState()
        self.nodes: Dict[str, NodeData] = {}
        self.running = False
        self.pipeline_id: Optional[str] = None

        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Track current image path
        self.current_image_path: Optional[Path] = None

        # GPU monitoring task
        self.gpu_monitor_task: Optional[asyncio.Task] = None

        # Main pipeline task (for cancellation)
        self.main_task: Optional[asyncio.Task] = None

        # History tracking
        self.history: Optional[PipelineHistory] = None

        # Iteration history - tracks each iteration's data
        self.iteration_history: List[Dict[str, Any]] = []

        # Pending job proposals (waiting for user approval)
        self.pending_job_proposals: List[Dict[str, Any]] = []

        # Initialize node structure
        self._initialize_nodes()

    def _initialize_nodes(self):
        """Initialize the pipeline node structure"""
        # Define node positions (for React Flow layout)
        self.nodes = {
            "input": NodeData(
                id="input",
                type=NodeType.INPUT,
                label="Input",
                position=NodePosition(x=250, y=0),
                status=NodeStatus.PENDING
            ),
            "t2i_generator": NodeData(
                id="t2i_generator",
                type=NodeType.T2I_GENERATOR,
                label="T2I Generator",
                position=NodePosition(x=250, y=120),
                status=NodeStatus.PENDING
            ),
            "vlm_detector": NodeData(
                id="vlm_detector",
                type=NodeType.VLM_DETECTOR,
                label="VLM Detector",
                position=NodePosition(x=250, y=240),
                status=NodeStatus.PENDING
            ),
            "text_kb_query": NodeData(
                id="text_kb_query",
                type=NodeType.TEXT_KB_QUERY,
                label="Text KB Query",
                position=NodePosition(x=100, y=360),
                status=NodeStatus.PENDING
            ),
            "clip_rag_search": NodeData(
                id="clip_rag_search",
                type=NodeType.CLIP_RAG_SEARCH,
                label="CLIP RAG Search",
                position=NodePosition(x=400, y=360),
                status=NodeStatus.PENDING
            ),
            "reference_selector": NodeData(
                id="reference_selector",
                type=NodeType.REFERENCE_SELECTOR,
                label="Reference Selector",
                position=NodePosition(x=250, y=480),
                status=NodeStatus.PENDING
            ),
            "prompt_adapter": NodeData(
                id="prompt_adapter",
                type=NodeType.PROMPT_ADAPTER,
                label="Prompt Adapter",
                position=NodePosition(x=100, y=600),
                status=NodeStatus.PENDING
            ),
            "i2i_editor": NodeData(
                id="i2i_editor",
                type=NodeType.I2I_EDITOR,
                label="I2I Editor",
                position=NodePosition(x=250, y=720),
                status=NodeStatus.PENDING
            ),
            "iteration_check": NodeData(
                id="iteration_check",
                type=NodeType.ITERATION_CHECK,
                label="Iteration Check",
                position=NodePosition(x=250, y=840),
                status=NodeStatus.PENDING
            ),
            "output": NodeData(
                id="output",
                type=NodeType.OUTPUT,
                label="Output",
                position=NodePosition(x=250, y=960),
                status=NodeStatus.PENDING
            ),
        }

    def _record_node_history(self, node_id: str, **kwargs):
        """Record detailed history for a node."""
        if not self.history:
            return

        # Create or update node history
        if node_id not in self.history.nodes:
            node = self.nodes.get(node_id)
            if not node:
                return

            self.history.nodes[node_id] = NodeHistory(
                node_id=node_id,
                node_type=node.type if isinstance(node.type, str) else node.type.value,
                status=node.status if isinstance(node.status, str) else node.status.value,
                start_time=node.start_time or time.time(),
            )

        node_history = self.history.nodes[node_id]

        # Update fields
        for key, value in kwargs.items():
            if hasattr(node_history, key):
                setattr(node_history, key, value)

    async def _gpu_monitoring_loop(self):
        """
        Background task that periodically broadcasts GPU stats
        """
        from api.system import get_gpu_stats

        while self.running:
            try:
                # Get current GPU stats
                stats = get_gpu_stats()

                # Broadcast to all connected clients
                await manager.broadcast_gpu_stats(stats)

                # Wait 2 seconds before next update
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def start(self, config: PipelineConfig) -> str:
        """
        Start pipeline execution

        Args:
            config: Pipeline configuration

        Returns:
            pipeline_id: Unique identifier for this pipeline run
        """
        if self.running:
            raise RuntimeError("Pipeline is already running")

        # Generate pipeline ID
        self.pipeline_id = str(uuid.uuid4())

        # Reset state
        self.state = PipelineState(
            status=PipelineStatus.RUNNING,
            config=config,
            start_time=time.time(),
            current_iteration=0
        )

        # Initialize history
        self.history = PipelineHistory(
            pipeline_id=self.pipeline_id,
            status="running",
            config=config.dict(),
            start_time=time.time(),
        )

        # Reset all nodes
        for node in self.nodes.values():
            node.status = NodeStatus.PENDING
            node.data = {}
            node.error = None
            node.progress = 0.0

        self.running = True

        # Start GPU monitoring background task
        self.gpu_monitor_task = asyncio.create_task(self._gpu_monitoring_loop())

        logger.info(f"Pipeline {self.pipeline_id} started")
        await manager.broadcast_pipeline_status("running", progress=0.0)

        return self.pipeline_id

    async def run(self):
        """
        Execute the pipeline

        This method runs asynchronously and broadcasts updates via WebSocket.
        """
        try:
            if not self.state.config:
                raise RuntimeError("Pipeline config not set")

            config = self.state.config

            # Step 1: Input node
            await self._execute_input_node(config)

            # Step 2: T2I Generation
            await self._execute_t2i_generator(config)

            # Iteration loop
            for iteration in range(config.max_iterations):
                # Check if stopped by user
                if not self.running:
                    logger.info("Pipeline stopped by user")
                    break

                self.state.current_iteration = iteration + 1

                # Step 3: VLM Detection
                score = await self._execute_vlm_detector(config, iteration)

                # Get VLM analysis data for stopping decision
                vlm_data = self.nodes["vlm_detector"].data
                issue_count = vlm_data.get("issue_count", 0)
                severe_count = vlm_data.get("severe_count", 0)
                moderate_count = vlm_data.get("moderate_count", 0)

                # Determine if we should stop based on issues and score
                should_stop = False
                stop_reason = ""

                # Condition 1: No issues at all detected
                if issue_count == 0:
                    should_stop = True
                    stop_reason = f"No issues detected (score: {score:.1f}/10)"

                # Condition 2: Target score reached AND no severe issues
                elif score >= config.target_score and severe_count == 0:
                    should_stop = True
                    stop_reason = f"Target score {config.target_score:.1f} reached ({score:.1f}/10) with no severe issues"
                    if moderate_count > 0:
                        stop_reason += f" ({moderate_count} moderate issues remain but acceptable)"

                # Condition 3: Very high score (>= 9) AND no severe issues
                elif score >= 9.0 and severe_count == 0:
                    should_stop = True
                    stop_reason = f"Excellent score achieved ({score:.1f}/10) with no severe issues"

                # Condition 4: Check for stagnation (no improvement in last 2 iterations)
                if not should_stop and iteration >= 2 and len(self.iteration_history) >= 2:
                    prev_iter = self.iteration_history[-1]
                    prev_issue_count = prev_iter.get("vlm_issue_count", 0)
                    prev_severe_count = prev_iter.get("vlm_severe_count", 0)

                    # If issue counts have not decreased
                    if issue_count >= prev_issue_count and severe_count >= prev_severe_count:
                        should_stop = True
                        stop_reason = f"No improvement detected (issues: {issue_count}, severe: {severe_count})"
                        logger.warning(f"Stagnation detected: {stop_reason}")

                if should_stop:
                    logger.info(f"✓ Stopping iteration: {stop_reason}")
                    break

                # Check if stopped before continuing
                if not self.running:
                    logger.info("Pipeline stopped by user")
                    break

                # Step 4: Reference Selection
                await self._execute_reference_selection(config)

                # Step 5: Prompt Adaptation
                await self._execute_prompt_adapter(config, iteration)

                # Step 6: I2I Editing
                await self._execute_i2i_editor(config, iteration)

            # Final output
            await self._execute_output_node()

            # Pipeline completed
            self.state.status = PipelineStatus.COMPLETED
            self.state.end_time = time.time()
            await manager.broadcast_pipeline_status("completed", progress=1.0)

            # Update history
            if self.history:
                self.history.status = "completed"
                self.history.end_time = time.time()
                self.history.duration = self.history.end_time - self.history.start_time
                self.history.iteration_count = self.state.current_iteration

                # Get final scores from VLM detector
                vlm_data = self.nodes["vlm_detector"].data
                if vlm_data:
                    self.history.final_cultural_score = vlm_data.get("cultural_score")
                    self.history.final_prompt_score = vlm_data.get("prompt_score")

                # Save final image path
                if self.current_image_path:
                    self.history.final_image_path = str(self.current_image_path)

            logger.info(f"Pipeline {self.pipeline_id} completed successfully")

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled by user")
            self.state.status = PipelineStatus.IDLE
            await manager.broadcast_pipeline_status("idle", progress=0.0)
            if self.history:
                self.history.status = "cancelled"
            raise  # Re-raise to properly handle cancellation

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.state.status = PipelineStatus.ERROR
            self.state.error = str(e)
            await manager.broadcast_error(str(e))

        finally:
            self.running = False

            # Stop GPU monitoring task
            if self.gpu_monitor_task and not self.gpu_monitor_task.done():
                self.gpu_monitor_task.cancel()
                try:
                    await self.gpu_monitor_task
                except asyncio.CancelledError:
                    pass

            # Save history
            if self.history:
                try:
                    # Update final state if not already set
                    if self.history.status == "running":
                        self.history.status = "error" if self.state.status == PipelineStatus.ERROR else "completed"
                        self.history.end_time = time.time()
                        self.history.duration = self.history.end_time - self.history.start_time

                    # Save to disk
                    history_mgr = get_history_manager()
                    history_mgr.save_history(self.history)
                    logger.info(f"Saved history for pipeline {self.pipeline_id}")

                except Exception as e:
                    logger.error(f"Failed to save history: {e}")

            # Cleanup GPU memory
            self._cleanup_gpu()

    async def _execute_input_node(self, config: PipelineConfig):
        """Execute input node"""
        node = self.nodes["input"]
        node.status = NodeStatus.PROCESSING
        node.start_time = time.time()

        await manager.broadcast_node_update(
            "input",
            "processing",
            {
                "prompt": config.prompt,
                "country": config.country,
                "category": config.category
            }
        )

        # Simulate processing
        await asyncio.sleep(0.5)

        node.status = NodeStatus.COMPLETED
        node.end_time = time.time()
        node.data = {
            "prompt": config.prompt,
            "country": config.country,
            "category": config.category
        }

        await manager.broadcast_node_update("input", "completed", node.data)

        # Save to history
        self._record_node_history(
            "input",
            output_data=node.data,
            status="completed",
            end_time=node.end_time
        )

    async def _execute_t2i_generator(self, config: PipelineConfig):
        """Execute T2I generator node with REAL adapter."""
        node = self.nodes["t2i_generator"]
        node.status = NodeStatus.PROCESSING
        node.start_time = time.time()

        await manager.broadcast_node_update(
            "t2i_generator",
            "processing",
            {"model": config.t2i_model, "prompt": config.prompt}
        )

        try:
            # Get component manager
            from services.component_manager import get_component_manager
            comp_mgr = get_component_manager(config.country)

            # Get T2I adapter
            logger.info(f"Creating T2I adapter: {config.t2i_model}")
            t2i_adapter = comp_mgr.get_t2i_adapter(config.t2i_model)

            # Generate image in thread pool (avoid blocking async loop)
            loop = asyncio.get_event_loop()

            # Progress callback for real-time updates
            def progress_callback(current_step, total_steps):
                # Schedule broadcast in async loop
                import time
                eta = (total_steps - current_step) * 2.5  # ~2.5s per step estimate
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast_progress(
                        "t2i_generator",
                        current_step,
                        total_steps,
                        eta_seconds=eta,
                        message=f"Generating image ({current_step}/{total_steps})"
                    ),
                    loop
                )

            def generate():
                # Check if pipeline was stopped
                if not self.running:
                    logger.info("T2I generation cancelled")
                    return None

                return t2i_adapter.generate(
                    prompt=config.prompt,
                    width=1024,
                    height=1024,
                    progress_callback=progress_callback,
                )

            image = await loop.run_in_executor(self.executor, generate)

            # Check if generation was cancelled
            if image is None:
                logger.info("T2I generation was cancelled, stopping pipeline")
                return

            # Save image
            session_dir = get_session_dir(self.pipeline_id)
            image_path = session_dir / "step_0_initial.png"
            image.save(image_path)

            # Update current image
            self.current_image_path = image_path

            node.status = NodeStatus.COMPLETED
            node.end_time = time.time()
            node.data = {
                "model": config.t2i_model,
                "image_path": str(image_path),
                "generation_time": node.end_time - node.start_time,
                # Send base64 for frontend display
                "image_base64": self._image_to_base64(image),
            }

            # Save to history
            self._record_node_history(
                "t2i_generator",
                output_data=node.data,
                status="completed",
                end_time=node.end_time
            )

            # Store initial image path in history
            if self.history:
                self.history.initial_image_path = str(image_path)

            logger.info(f"✓ T2I generation complete: {image_path}")
            await manager.broadcast_node_update("t2i_generator", "completed", node.data)

        except Exception as e:
            logger.error(f"T2I generation failed: {e}", exc_info=True)
            node.status = NodeStatus.ERROR
            node.error = str(e)
            await manager.broadcast_error(str(e), "t2i_generator")
            raise

    async def _execute_vlm_detector(self, config: PipelineConfig, iteration: int) -> float:
        """Execute VLM detector node with REAL Qwen3-VL."""
        node = self.nodes["vlm_detector"]
        node.status = NodeStatus.PROCESSING
        node.start_time = time.time()

        await manager.broadcast_node_update(
            "vlm_detector",
            "processing",
            {"iteration": iteration}
        )

        try:
            # Get component manager
            from services.component_manager import get_component_manager
            comp_mgr = get_component_manager(config.country)

            # Get current image
            current_image = self._get_current_image()
            if not current_image or not current_image.exists():
                raise RuntimeError("No image available for VLM analysis")

            # Get VLM detector (heavy, cached)
            logger.info("Getting VLM detector...")
            vlm = comp_mgr.get_vlm_detector(config.load_in_4bit)

            # Run detection in thread pool
            loop = asyncio.get_event_loop()

            def detect():
                # Get previous scores for iteration context
                prev_cultural = None
                prev_prompt = None
                if iteration > 0 and len(self.iteration_history) > 0:
                    prev_iter = self.iteration_history[-1]
                    prev_cultural = prev_iter.get("cultural_score")
                    prev_prompt = prev_iter.get("prompt_score")

                # Get scores with iteration context
                cultural_score, prompt_score = vlm.score_cultural_quality(
                    current_image,
                    config.prompt,
                    config.country,
                    None,  # editing_prompt
                    iteration_number=iteration,
                    previous_cultural_score=prev_cultural,
                    previous_prompt_score=prev_prompt,
                )

                # Get detailed issues with iteration context
                issues = vlm.detect(
                    current_image,
                    config.prompt,
                    config.country,
                    None,  # editing_prompt
                    config.category,
                    iteration_number=iteration,
                    previous_cultural_score=prev_cultural,
                    previous_prompt_score=prev_prompt,
                )

                return cultural_score, prompt_score, issues

            cultural_score, prompt_score, issues = await loop.run_in_executor(
                self.executor, detect
            )

            node.status = NodeStatus.COMPLETED
            node.end_time = time.time()

            # Count severe issues and adjust scores if needed
            severe_issues = []
            moderate_issues = []
            minor_issues = []

            for issue in issues:
                severity = 5  # default
                if isinstance(issue, dict):
                    severity = issue.get('severity', 5)

                if severity >= 8:
                    severe_issues.append(issue)
                elif severity >= 5:
                    moderate_issues.append(issue)
                else:
                    minor_issues.append(issue)

            # Original scores from VLM
            original_cultural_score = cultural_score
            original_prompt_score = prompt_score

            # Apply penalty for severe issues
            num_severe = len(severe_issues)
            if num_severe > 0:
                # Penalty: reduce score by 0.5 for each severe issue
                penalty = num_severe * 0.5
                adjusted_cultural_score = max(1.0, min(cultural_score, 10.0 - penalty))

                # If there's a big discrepancy (score >= 8 but has severe issues), apply stronger penalty
                if cultural_score >= 8.0 and num_severe >= 2:
                    adjusted_cultural_score = min(adjusted_cultural_score, 7.0)
                    logger.warning(f"⚠ Score adjustment: {cultural_score:.1f} → {adjusted_cultural_score:.1f} "
                                 f"(found {num_severe} severe issues)")

                cultural_score = adjusted_cultural_score

            # Detect score mismatch
            score_mismatch = original_cultural_score >= 8.0 and num_severe >= 2

            # Extract issue descriptions for preview (display only)
            issues_preview = []
            for i in issues[:3]:
                if isinstance(i, str):
                    issues_preview.append(i[:100])
                elif isinstance(i, dict):
                    issues_preview.append(i.get('description', str(i))[:100])
                else:
                    issues_preview.append(str(i)[:100])

            # Determine which issues are new vs remaining from previous iteration
            previous_issues = []
            if iteration > 0 and len(self.iteration_history) > 0:
                prev_iter = self.iteration_history[-1]
                previous_issues = prev_iter.get("vlm_issues", [])

            # Track issue progression
            remaining_issues = []
            fixed_issues = []
            new_issues = []

            # Simple heuristic: compare issue descriptions
            prev_issue_descs = set()
            if previous_issues:
                for pi in previous_issues:
                    if isinstance(pi, dict):
                        prev_issue_descs.add(pi.get('description', str(pi))[:200])
                    else:
                        prev_issue_descs.add(str(pi)[:200])

            current_issue_descs = set()
            for issue in issues:
                if isinstance(issue, dict):
                    desc = issue.get('description', str(issue))[:200]
                else:
                    desc = str(issue)[:200]
                current_issue_descs.add(desc)

                if desc in prev_issue_descs:
                    remaining_issues.append(issue)
                else:
                    new_issues.append(issue)

            # Issues that were fixed: in previous but not in current
            for prev_desc in prev_issue_descs:
                if prev_desc not in current_issue_descs:
                    fixed_issues.append(prev_desc)

            # Store current iteration data
            node.data = {
                "cultural_score": cultural_score,
                "prompt_score": prompt_score,
                "issues": issues,  # Full issue list
                "issue_count": len(issues),
                "iteration": iteration,
                # Preview for node display
                "issues_preview": issues_preview,
                # Issue progression tracking
                "remaining_issues": remaining_issues,
                "fixed_issues": fixed_issues,
                "new_issues": new_issues,
                "fixed_count": len(fixed_issues),
                "remaining_count": len(remaining_issues),
                "new_count": len(new_issues),
                # Severity breakdown
                "severe_count": len(severe_issues),
                "moderate_count": len(moderate_issues),
                "minor_count": len(minor_issues),
                "severe_issues": severe_issues,
                # Score adjustment tracking
                "original_cultural_score": original_cultural_score,
                "score_adjusted": original_cultural_score != cultural_score,
                "score_mismatch": score_mismatch,
                # Full iteration history for display
                "iteration_history": self.iteration_history,
            }

            logger.info(f"✓ VLM detection complete: Cultural={cultural_score:.1f}, Prompt={prompt_score:.1f}, "
                       f"Fixed={len(fixed_issues)}, Remaining={len(remaining_issues)}, New={len(new_issues)}")

            # Save to history
            self._record_node_history(
                "vlm_detector",
                output_data=node.data,
                iteration=iteration,
                vlm_analysis={
                    "cultural_score": cultural_score,
                    "prompt_score": prompt_score,
                    "issues": issues,
                    "issue_count": len(issues),
                    "iteration": iteration,
                },
                status="completed",
                end_time=node.end_time
            )

            await manager.broadcast_node_update("vlm_detector", "completed", node.data)

            # Add to scores history for tracking progression
            if self.history:
                self.history.scores_history.append({
                    "iteration": iteration,
                    "cultural_score": cultural_score,
                    "prompt_score": prompt_score,
                    "issue_count": len(issues),
                    "fixed_count": len(fixed_issues),
                    "remaining_count": len(remaining_issues),
                    "new_count": len(new_issues),
                    "timestamp": time.time(),
                })
                logger.debug(f"Added to scores_history: iteration {iteration}")

            # Analyze data gaps and create jobs if needed
            # Only do this on first iteration to avoid creating duplicate jobs
            if iteration == 0 and issues:
                await self._analyze_and_create_jobs(config, issues)

            return cultural_score

        except Exception as e:
            logger.error(f"VLM detection failed: {e}", exc_info=True)
            node.status = NodeStatus.ERROR
            node.error = str(e)
            await manager.broadcast_error(str(e), "vlm_detector")
            raise

    async def _execute_reference_selection(self, config: PipelineConfig):
        """Execute reference selection with REAL CLIP RAG."""
        try:
            # Get component manager
            from services.component_manager import get_component_manager
            comp_mgr = get_component_manager(config.country)

            # Get current image and VLM issues
            current_image = self._get_current_image()
            vlm_data = self.nodes["vlm_detector"].data
            issues = vlm_data.get("issues", [])

            # Text KB Query (lightweight, can be simplified for now)
            text_kb_node = self.nodes["text_kb_query"]
            text_kb_node.status = NodeStatus.PROCESSING
            await manager.broadcast_node_update("text_kb_query", "processing", {})

            # Prepare KB results for frontend
            kb_results = []
            for i, issue in enumerate(issues[:5]):  # Top 5 issues
                kb_results.append({
                    "text": issue if isinstance(issue, str) else str(issue),
                    "score": 1.0 - (i * 0.1),  # Decreasing relevance
                })

            text_kb_node.status = NodeStatus.COMPLETED
            text_kb_node.data = {
                "results_count": len(issues),
                "kb_results": kb_results,
            }

            # Save to history
            self._record_node_history(
                "text_kb_query",
                output_data=text_kb_node.data,
                text_rag={"results": kb_results},
                status="completed"
            )

            await manager.broadcast_node_update("text_kb_query", "completed", text_kb_node.data)

            # CLIP RAG Search
            clip_node = self.nodes["clip_rag_search"]
            clip_node.status = NodeStatus.PROCESSING
            await manager.broadcast_node_update("clip_rag_search", "processing", {})

            logger.info("Getting CLIP RAG...")
            clip_rag = comp_mgr.get_clip_rag()

            # Search for similar images
            loop = asyncio.get_event_loop()

            def search_clip():
                return clip_rag.retrieve_similar_images(
                    image_path=Path(current_image),
                    k=10,
                    category=None  # No category filter for now
                )

            clip_results = await loop.run_in_executor(self.executor, search_clip)

            # Prepare search results for frontend
            search_results = []
            if clip_results:
                for result in clip_results[:10]:  # Top 10 results
                    # CLIP RAG returns: {'image_path', 'similarity', 'category', 'metadata'}
                    search_results.append({
                        "path": str(result.get("image_path", "")),
                        "score": float(result.get("similarity", 0.0)),
                        "category": result.get("category", "unknown"),
                        "metadata": result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {},
                    })

            clip_node.status = NodeStatus.COMPLETED
            clip_node.data = {
                "results_count": len(clip_results) if clip_results else 0,
                "search_results": search_results,
            }

            # Save to history
            self._record_node_history(
                "clip_rag_search",
                output_data=clip_node.data,
                clip_rag={"results": search_results},
                status="completed"
            )

            await manager.broadcast_node_update("clip_rag_search", "completed", clip_node.data)

            # Reference Selector
            ref_node = self.nodes["reference_selector"]
            ref_node.status = NodeStatus.PROCESSING
            await manager.broadcast_node_update("reference_selector", "processing", {})

            logger.info("Getting reference selector...")
            selector = comp_mgr.get_reference_selector()

            # Select best reference
            def select_ref():
                return selector.select_best_reference(
                    str(current_image),
                    issues,
                    config.category,
                    k=10,
                )

            reference = await loop.run_in_executor(self.executor, select_ref)

            ref_node.status = NodeStatus.COMPLETED
            if reference:
                ref_node.data = {
                    "selected": Path(reference["image_path"]).name if "image_path" in reference else None,
                    "score": reference.get("total_score", 0),
                    "reason": reference.get("reason", ""),
                    # Send reference image path for frontend display
                    "selected_path": reference.get("image_path"),
                }
                logger.info(f"✓ Selected reference: {ref_node.data['selected']} (score: {ref_node.data['score']:.3f})")
            else:
                ref_node.data = {
                    "selected": None,
                    "score": 0,
                    "reason": "No suitable reference found",
                }
                logger.warning("No reference image selected")

            await manager.broadcast_node_update("reference_selector", "completed", ref_node.data)

            # Save to history
            self._record_node_history(
                "reference_selector",
                output_data=ref_node.data,
                selected_reference={
                    "selected": ref_node.data.get("selected"),
                    "score": ref_node.data.get("score"),
                    "reason": ref_node.data.get("reason"),
                },
                status="completed"
            )

        except Exception as e:
            logger.error(f"Reference selection failed: {e}", exc_info=True)
            ref_node = self.nodes["reference_selector"]
            ref_node.status = NodeStatus.ERROR
            ref_node.error = str(e)
            await manager.broadcast_error(str(e), "reference_selector")
            raise

    async def _execute_prompt_adapter(self, config: PipelineConfig, iteration: int):
        """Execute prompt adapter with REAL adapter."""
        node = self.nodes["prompt_adapter"]
        node.status = NodeStatus.PROCESSING
        await manager.broadcast_node_update("prompt_adapter", "processing", {})

        try:
            # Get component manager
            from services.component_manager import get_component_manager
            from ccub2_agent.modules.prompt_adapter import EditingContext
            comp_mgr = get_component_manager(config.country)

            # Get prompt adapter
            prompt_adapter = comp_mgr.get_prompt_adapter()

            # Get data from previous nodes
            vlm_data = self.nodes["vlm_detector"].data
            ref_data = self.nodes["reference_selector"].data

            # Get issues (keep as full Dicts, don't stringify!)
            all_issues = vlm_data.get("issues", [])

            # Get iteration context
            previous_iteration_issues = []
            fixed_issues = vlm_data.get("fixed_issues", [])
            remaining_issues = vlm_data.get("remaining_issues", [])
            new_issues = vlm_data.get("new_issues", [])
            previous_editing_prompt = None

            if iteration > 0 and len(self.iteration_history) > 0:
                prev_iter = self.iteration_history[-1]
                previous_iteration_issues = prev_iter.get("vlm_issues", [])
                previous_editing_prompt = prev_iter.get("editing_prompt")

            # PROGRESSIVE PROMPT IMPROVEMENT: Focus on unfixed issues
            # Iteration 0: Fix all detected issues
            # Iteration 1+: Focus on remaining + new issues only
            if iteration == 0:
                issues = all_issues  # Fix everything on first iteration
                logger.info(f"Iteration {iteration}: Addressing all {len(issues)} issues")
            else:
                # Focus on issues that weren't fixed + new issues discovered
                issues = remaining_issues + new_issues
                logger.info(f"Iteration {iteration}: Focusing on {len(remaining_issues)} remaining + {len(new_issues)} new issues "
                           f"(ignoring {len(fixed_issues)} fixed issues)")

                # If all issues are resolved, use the original issues as fallback
                if len(issues) == 0 and len(all_issues) > 0:
                    issues = all_issues
                    logger.warning(f"No remaining/new issues detected, using all {len(issues)} issues as fallback")

            # SEQUENTIAL ISSUE FIXING: Select ONLY top 1 issue by severity for focused improvement
            if len(issues) > 1:
                sorted_issues = sorted(
                    issues,
                    key=lambda x: x.get('severity', 5) if isinstance(x, dict) else 5,
                    reverse=True
                )
                selected_issue = sorted_issues[0]
                severity = selected_issue.get('severity', 5) if isinstance(selected_issue, dict) else 5
                issues = [selected_issue]
                logger.info(f"Sequential fix: Focusing on TOP 1 issue (severity {severity}) for better results")
            elif len(issues) == 1:
                logger.info(f"Iteration {iteration}: Addressing 1 remaining issue")
            else:
                logger.info(f"Iteration {iteration}: No issues to address")

            # Extract cultural elements from reference data or CLIP metadata
            cultural_elements = ref_data.get("reason", "")
            if not cultural_elements:
                # If Reference Selector didn't provide context, use CLIP metadata
                clip_data = self.nodes.get("clip_rag_search", {}).data
                search_results = clip_data.get("search_results", [])
                if search_results:
                    # Extract enhanced descriptions from top matches
                    descriptions = []
                    for result in search_results[:3]:  # Top 3 references
                        metadata = result.get("metadata", {})
                        enhanced = metadata.get("description_enhanced", "")
                        if enhanced:
                            descriptions.append(enhanced)

                    if descriptions:
                        cultural_elements = " ".join(descriptions)
                        logger.info(f"✓ Added focused cultural guidance ({len(descriptions)} features)")
                    else:
                        logger.warning("No enhanced descriptions found in CLIP metadata")

            # Build EditingContext with full structured data + iteration context
            editing_context = EditingContext(
                original_prompt=config.prompt,
                detected_issues=issues,  # Pass full Dict list, not stringified
                cultural_elements=cultural_elements,
                reference_images=[ref_data.get("selected_path")] if ref_data.get("selected_path") else None,
                country=config.country,
                category=config.category or "traditional_clothing",
                preserve_identity=True,
                # Iteration tracking
                iteration_number=iteration,
                previous_iteration_issues=previous_iteration_issues if previous_iteration_issues else None,
                fixed_issues=fixed_issues if fixed_issues else None,
                remaining_issues=remaining_issues if remaining_issues else None,
                previous_editing_prompt=previous_editing_prompt,
            )

            # Create universal instruction
            base_instruction = "Improve cultural accuracy"
            if issues:
                # Extract description from first issue for display (limit to 1 for focused editing)
                issue_descriptions = []
                for i in issues[:1]:
                    if isinstance(i, dict):
                        desc = i.get('description', str(i))[:500]
                    else:
                        desc = str(i)[:500]
                    issue_descriptions.append(desc)
                base_instruction += f". Fix: {' '.join(issue_descriptions)}"

            # Use the proper adapter to generate model-specific prompt
            adapted_prompt = prompt_adapter.adapt(
                universal_instruction=base_instruction,
                model_type=config.i2i_model,
                context=editing_context
            )

            # Record prompt flow in history
            # Extract issue descriptions for history (limit to 200 chars for storage)
            issue_history = []
            for i in issues[:3]:
                if isinstance(i, dict):
                    desc = i.get('description', str(i))[:200]
                else:
                    desc = str(i)[:200]
                issue_history.append(desc)

            self._record_node_history(
                "prompt_adapter",
                prompt_flow=[
                    PromptFlowStep(
                        step="original",
                        prompt=config.prompt,
                        metadata={"source": "user_input"}
                    ),
                    PromptFlowStep(
                        step="issues_detected",
                        prompt=base_instruction,
                        metadata={
                            "issues": issue_history,
                            "issue_count": len(issues)
                        }
                    ),
                    PromptFlowStep(
                        step="adapted",
                        prompt=adapted_prompt,
                        metadata={"target_model": config.i2i_model}
                    ),
                ],
                end_time=time.time(),
                status="completed",
                output_data=node.data
            )

            node.status = NodeStatus.COMPLETED
            node.data = {
                "adapted_prompt": adapted_prompt,
                "model": config.i2i_model,
                "original_prompt": config.prompt,
                "issues_addressed": [
                    i.get('description', str(i))[:100] if isinstance(i, dict) else str(i)[:100]
                    for i in issues[:5]
                ]
            }
            await manager.broadcast_node_update("prompt_adapter", "completed", node.data)

        except Exception as e:
            logger.error(f"Prompt adaptation failed: {e}", exc_info=True)
            node.status = NodeStatus.ERROR
            node.error = str(e)
            await manager.broadcast_error(str(e), "prompt_adapter")
            # Continue anyway with default prompt

    async def _execute_i2i_editor(self, config: PipelineConfig, iteration: int):
        """Execute I2I editor with REAL adapter."""
        node = self.nodes["i2i_editor"]
        node.status = NodeStatus.PROCESSING
        node.start_time = time.time()

        try:
            # Get component manager
            from services.component_manager import get_component_manager
            comp_mgr = get_component_manager(config.country)

            # Get current image
            current_image = self._get_current_image()
            if not current_image or not current_image.exists():
                raise RuntimeError("No image available for I2I editing")

            # Get editing prompt
            prompt_data = self.nodes["prompt_adapter"].data
            editing_prompt = prompt_data.get("adapted_prompt", "Improve cultural accuracy")

            # Get reference (optional)
            ref_data = self.nodes["reference_selector"].data
            reference_path = ref_data.get("reference_path")

            # If Reference Selector failed, fallback to CLIP RAG Search results
            reference_paths = []
            if not reference_path:
                logger.warning("Reference Selector found no suitable reference, using CLIP RAG results directly")
                clip_data = self.nodes.get("clip_rag_search", {}).data
                search_results = clip_data.get("search_results", [])
                if search_results:
                    # Use top-3 results from CLIP RAG
                    reference_paths = [result["path"] for result in search_results[:3] if result.get("path")]
                    logger.info(f"Using {len(reference_paths)} reference images from CLIP RAG (similarities: {[f'{r['score']:.1%}' for r in search_results[:3]]})")
            else:
                reference_paths = [reference_path]

            # Get I2I adapter
            logger.info(f"Getting I2I adapter: {config.i2i_model}")
            i2i_adapter = comp_mgr.get_i2i_adapter(config.i2i_model)

            # Run I2I editing in thread pool
            loop = asyncio.get_event_loop()

            # Progress callback for real-time updates
            def progress_callback(current_step, total_steps):
                # Schedule broadcast in async loop
                import time
                eta = (total_steps - current_step) * 2.5  # ~2.5s per step estimate
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast_progress(
                        "i2i_editor",
                        current_step,
                        total_steps,
                        eta_seconds=eta,
                        message=f"Editing image (iter {iteration}, {current_step}/{total_steps})"
                    ),
                    loop
                )

            def edit():
                # Check if pipeline was stopped
                if not self.running:
                    logger.info("I2I editing cancelled")
                    return None

                # Load images
                from PIL import Image
                current_img = Image.open(current_image)

                # Load reference images (single or multiple)
                ref_imgs = []
                for ref_path in reference_paths:
                    if ref_path and Path(ref_path).exists():
                        ref_imgs.append(Image.open(ref_path))

                # Use first reference for single-reference models, or pass all for multi-reference support
                ref_img = ref_imgs[0] if ref_imgs else None

                logger.info(f"Editing with {len(ref_imgs)} reference image(s)")

                # Build reference metadata from CLIP results if available
                ref_metadata = None
                if ref_img and len(reference_paths) > 0:
                    # Get metadata from CLIP RAG results
                    clip_data = self.nodes.get("clip_rag_search", {}).data
                    search_results = clip_data.get("search_results", [])
                    if search_results:
                        # Use metadata from first reference
                        ref_metadata = search_results[0].get("metadata", {})
                        logger.info(f"Using CLIP metadata: category={ref_metadata.get('category')}, similarity={search_results[0].get('score', 0):.1%}")
                elif ref_img:
                    # Fallback to Reference Selector metadata
                    ref_metadata = ref_data

                # Adaptive CFG scale: Higher when using reference to prevent copying
                # Reference image mode: CFG 12.0-15.0 to prioritize instruction over reference
                # No reference mode: CFG 7.0 for balanced editing
                cfg_scale = 14.0 if ref_img else 7.0

                # Edit with stronger parameters for visible changes
                return i2i_adapter.edit(
                    image=current_img,
                    instruction=editing_prompt,
                    reference_image=ref_img,
                    reference_metadata=ref_metadata,
                    strength=0.9,  # Increased from 0.7 for stronger edits (note: not used by Qwen)
                    true_cfg_scale=cfg_scale,  # Adaptive: 14.0 with reference, 7.0 without
                    num_inference_steps=50,  # Increased from 40 for better quality
                    progress_callback=progress_callback,
                )

            # Start processing notification
            await manager.broadcast_node_update(
                "i2i_editor",
                "processing",
                {"iteration": iteration, "status": "editing"}
            )

            edited_image = await loop.run_in_executor(self.executor, edit)

            # Check if editing was cancelled
            if edited_image is None:
                logger.info("I2I editing was cancelled, stopping pipeline")
                return

            # Save edited image
            session_dir = get_session_dir(self.pipeline_id)
            edited_path = session_dir / f"step_{iteration + 1}_edited.png"
            edited_image.save(edited_path)

            # Update current image
            self.current_image_path = edited_path

            node.status = NodeStatus.COMPLETED
            node.end_time = time.time()

            # Get issues that were addressed
            vlm_data = self.nodes["vlm_detector"].data
            issues = vlm_data.get("issues", [])

            # Extract issue descriptions for display
            issues_addressed = []
            for i in issues[:5]:
                if isinstance(i, dict):
                    issues_addressed.append(i.get('description', str(i))[:100])
                else:
                    issues_addressed.append(str(i)[:100])

            # Record this iteration's complete data FIRST
            iteration_data = {
                "iteration": iteration,
                "timestamp": time.time(),
                "image_path": str(edited_path),
                "vlm_cultural_score": vlm_data.get("cultural_score", 0),
                "vlm_prompt_score": vlm_data.get("prompt_score", 0),
                "vlm_issues": issues,
                "vlm_issue_count": len(issues),
                "vlm_severe_count": vlm_data.get("severe_count", 0),
                "vlm_moderate_count": vlm_data.get("moderate_count", 0),
                "vlm_minor_count": vlm_data.get("minor_count", 0),
                "fixed_issues": vlm_data.get("fixed_issues", []),
                "remaining_issues": vlm_data.get("remaining_issues", []),
                "new_issues": vlm_data.get("new_issues", []),
                "editing_prompt": editing_prompt,
                "editing_time": node.end_time - node.start_time,
            }
            self.iteration_history.append(iteration_data)

            # NOW update node.data with complete iteration history
            node.data = {
                "iteration": iteration,
                "output_image": str(edited_path),
                "image_base64": self._image_to_base64(edited_image),
                "editing_time": node.end_time - node.start_time,
                "editing_prompt": editing_prompt,  # Show what edits were applied
                "issues_addressed": issues_addressed if issues else [],  # Show what was addressed
                # Include full iteration history including current iteration
                "iteration_history": self.iteration_history,
            }

            # Save to history
            self._record_node_history(
                "i2i_editor",
                output_data=node.data,
                iteration=iteration,
                editing_prompt=editing_prompt,
                issues_fixed=len(vlm_data.get("fixed_issues", [])),
                editing_params={
                    "editing_time": node.end_time - node.start_time,
                },
                status="completed",
                end_time=node.end_time
            )

            logger.info(f"✓ I2I editing complete (iteration {iteration}): {edited_path}")
            logger.info(f"  Iteration history now has {len(self.iteration_history)} entries")
            await manager.broadcast_node_update("i2i_editor", "completed", node.data)

        except Exception as e:
            logger.error(f"I2I editing failed: {e}", exc_info=True)
            node.status = NodeStatus.ERROR
            node.error = str(e)
            await manager.broadcast_error(str(e), "i2i_editor")
            raise

    async def _execute_output_node(self):
        """Execute output node"""
        node = self.nodes["output"]
        node.status = NodeStatus.PROCESSING
        await manager.broadcast_node_update("output", "processing", {})

        # Get final scores from last VLM detection
        vlm_data = self.nodes["vlm_detector"].data
        final_cultural_score = vlm_data.get("cultural_score", 0)
        final_prompt_score = vlm_data.get("prompt_score", 0)

        node.status = NodeStatus.COMPLETED
        node.data = {
            "final_cultural_score": final_cultural_score,
            "final_prompt_score": final_prompt_score,
            "iterations": self.state.current_iteration,
            "final_image": str(self.current_image_path) if self.current_image_path else None,
        }
        logger.info(f"✓ Pipeline complete! Final score: {final_cultural_score:.1f}/10")
        await manager.broadcast_node_update("output", "completed", node.data)

        # Save to history
        self._record_node_history(
            "output",
            output_data=node.data,
            status="completed"
        )

    def _image_to_base64(self, image_input) -> str:
        """
        Convert PIL Image or path to base64 string for frontend display.

        Args:
            image_input: PIL Image object or path to image file

        Returns:
            Base64-encoded JPEG string
        """
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to JPEG for smaller size
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()

        return base64.b64encode(img_bytes).decode('utf-8')

    def _get_current_image(self) -> Optional[Path]:
        """
        Get path to the current working image.

        Returns most recent image from pipeline (edited or initial).
        """
        return self.current_image_path

    def _cleanup_gpu(self):
        """Clean up GPU memory after pipeline execution."""
        import gc
        import torch

        logger.info("Cleaning up GPU memory...")

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ GPU memory cleared")

    async def _analyze_and_create_jobs(self, config: PipelineConfig, issues: List[Dict]):
        """
        Analyze data gaps from VLM issues and create collection jobs if needed.

        Args:
            config: Pipeline configuration
            issues: List of issues detected by VLM
        """
        if not issues:
            logger.info("No issues detected, skipping job creation")
            return

        logger.info(f"Analyzing {len(issues)} issues for data gaps...")

        try:
            # Get component manager
            from services.component_manager import get_component_manager
            comp_mgr = get_component_manager(config.country)

            # Get gap analyzer and job creator
            loop = asyncio.get_event_loop()

            def analyze_gaps():
                gap_analyzer = comp_mgr.get_gap_analyzer()
                return gap_analyzer.analyze(issues, config.country)

            # Analyze gaps in thread pool
            gaps = await loop.run_in_executor(self.executor, analyze_gaps)

            if not gaps:
                logger.info("No data gaps identified")
                await manager.broadcast({
                    "type": "job_creation",
                    "status": "no_gaps",
                    "message": "Data coverage is sufficient",
                    "timestamp": time.time()
                })
                return

            logger.info(f"Found {len(gaps)} data gaps")

            # Prepare job proposals
            proposals = []
            for i, gap in enumerate(gaps):
                proposals.append({
                    "id": f"{self.pipeline_id}_gap_{i}",
                    "category": gap["category"],
                    "subcategory": gap.get("subcategory", ""),
                    "keywords": gap.get("keywords", []),
                    "priority": gap.get("priority", 1),
                    "reason": gap.get("description", f"Missing data for {gap['category']}"),
                    "country": config.country,
                    "points": 50 + (gap.get("priority", 1) * 5),
                    "target_count": 50,
                })

            # Broadcast job proposals for user approval
            logger.info(f"Sending {len(proposals)} job proposals for user approval")
            await manager.broadcast({
                "type": "job_proposal",
                "proposals": proposals,
                "pipeline_id": self.pipeline_id,
                "message": f"Found {len(proposals)} data gap(s). Review and approve job creation.",
                "timestamp": time.time()
            })

            # Store proposals for later approval
            self.pending_job_proposals = proposals

            logger.info("Job creation complete: proposals sent, awaiting user approval")

            # OLD CODE - Now handled via API when user approves:
            # created_jobs = []
            # for gap in gaps:
            #     job_creator.create_job(...)
            #     await manager.broadcast({"type": "job_creation", "status": "created", ...})

            # Continue pipeline without waiting for job approval
            return

            # Placeholder for old broadcast (will be removed)
            await manager.broadcast({
                        "type": "job_creation",
                        "status": "skipped",
                        "gap": {
                            "category": gap["category"],
                            "subcategory": gap["subcategory"],
                        },
                        "message": f"Job for {gap['category']} already exists or creation failed",
                        "timestamp": time.time()
                    })

            logger.info(f"Job creation complete: {len(created_jobs)} jobs created")

        except Exception as e:
            logger.error(f"Failed to analyze gaps or create jobs: {e}", exc_info=True)
            await manager.broadcast({
                "type": "job_creation",
                "status": "error",
                "error": str(e),
                "message": "Failed to create data collection jobs",
                "timestamp": time.time()
            })

    async def stop(self, force: bool = True):
        """
        Stop pipeline execution with proper cleanup

        Args:
            force: If True, cancel immediately. If False, wait for current operation to finish gracefully.
        """
        logger.info(f"Stopping pipeline (force={force})...")

        # 1. Set running flag to prevent new operations
        self.running = False
        self.state.status = PipelineStatus.IDLE

        # 2. Cancel main task if running
        if self.main_task and not self.main_task.done():
            if force:
                logger.info("Cancelling main pipeline task...")
                self.main_task.cancel()
                try:
                    await self.main_task
                except asyncio.CancelledError:
                    logger.info("✓ Main task cancelled successfully")
            else:
                # Wait for current iteration to finish gracefully (max 30s)
                logger.info("Waiting for current operation to finish...")
                try:
                    await asyncio.wait_for(self.main_task, timeout=30.0)
                    logger.info("✓ Pipeline stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for graceful stop, forcing cancellation...")
                    self.main_task.cancel()
                    try:
                        await self.main_task
                    except asyncio.CancelledError:
                        pass

        # 3. GPU cleanup - free VRAM
        logger.info("Cleaning up GPU memory...")
        from services.component_manager import cleanup_all
        cleanup_all()

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("✓ GPU memory cleared")

        # 4. Reset state
        self.main_task = None

        logger.info("✓ Pipeline stopped and GPU cleaned")

    def is_running(self) -> bool:
        """Check if pipeline is currently running"""
        return self.running

    def get_state(self) -> PipelineState:
        """Get current pipeline state"""
        return self.state

    def get_node_data(self, node_id: str) -> Optional[NodeData]:
        """Get data for a specific node"""
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeData]:
        """Get all nodes"""
        return list(self.nodes.values())

    def get_node_outputs(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get outputs from a specific node"""
        node = self.nodes.get(node_id)
        if node:
            return node.data
        return None


# Global pipeline runner instance
pipeline_runner = PipelineRunner()
