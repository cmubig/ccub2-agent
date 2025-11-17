"""
Component Manager for CCUB2 Agent

Manages initialization and lifecycle of CCUB2 components:
- VLM Detector (Qwen3-VL)
- CLIP Image RAG
- Reference Selector
- Prompt Adapter
- T2I and I2I Adapters

Uses lazy loading to save memory.
"""

import sys
from pathlib import Path
from typing import Optional
import logging

# Add ccub2_agent to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_country_config, MODELS

logger = logging.getLogger(__name__)


class ComponentManager:
    """
    Manages CCUB2 components with lazy loading and caching.

    This class initializes heavy ML models on-demand and caches them
    to avoid repeated loading. It also provides cleanup methods
    to free GPU memory.
    """

    def __init__(self, country: str = "korea"):
        """
        Initialize component manager.

        Args:
            country: Country code (e.g., "korea", "japan")
        """
        self.country = country
        self.config = get_country_config(country)

        # Cached components (lazy initialization)
        self._vlm_detector: Optional[any] = None
        self._clip_rag: Optional[any] = None
        self._reference_selector: Optional[any] = None
        self._prompt_adapter: Optional[any] = None
        self._current_t2i_model: Optional[str] = None
        self._current_i2i_model: Optional[str] = None
        self._t2i_adapter: Optional[any] = None
        self._i2i_adapter: Optional[any] = None

        # Job creation components (lazy)
        self._gap_analyzer: Optional[any] = None
        self._job_creator: Optional[any] = None

        logger.info(f"Component manager initialized for country: {country}")

    def get_vlm_detector(self, load_in_4bit: bool = True):
        """
        Get VLM cultural detector (Qwen3-VL-8B).

        This is a HEAVY model (~8-16GB VRAM). Uses caching to avoid
        repeated loading.

        Args:
            load_in_4bit: Use 4-bit quantization to save VRAM

        Returns:
            VLMCulturalDetector instance
        """
        if self._vlm_detector is None:
            logger.info("Initializing VLM Detector (Qwen3-VL-8B)...")
            logger.info(f"  - Text index: {self.config['text_index']}")
            logger.info(f"  - CLIP index: {self.config['clip_index']}")
            logger.info(f"  - 4-bit mode: {load_in_4bit}")

            try:
                from ccub2_agent.modules.vlm_detector import create_vlm_detector

                self._vlm_detector = create_vlm_detector(
                    model_name=MODELS["vlm"],
                    index_dir=self.config["text_index"],
                    clip_index_dir=self.config["clip_index"],
                    load_in_4bit=load_in_4bit,
                    debug=False,
                )

                logger.info("✓ VLM Detector initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize VLM Detector: {e}")
                raise

        return self._vlm_detector

    def get_clip_rag(self):
        """
        Get CLIP Image RAG for visual similarity search.

        Returns:
            CLIPImageRAG instance
        """
        if self._clip_rag is None:
            logger.info("Initializing CLIP Image RAG...")
            logger.info(f"  - Index dir: {self.config['clip_index']}")

            try:
                from ccub2_agent.modules.clip_image_rag import create_clip_rag

                self._clip_rag = create_clip_rag(
                    model_name=MODELS["clip"],
                    index_dir=str(self.config["clip_index"]),
                    device="auto",
                )

                logger.info("✓ CLIP RAG initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize CLIP RAG: {e}")
                raise

        return self._clip_rag

    def get_reference_selector(self):
        """
        Get reference image selector (CLIP + keyword matching).

        Returns:
            ReferenceImageSelector instance
        """
        if self._reference_selector is None:
            logger.info("Initializing Reference Selector...")

            try:
                # Reference selector needs CLIP RAG
                clip_rag = self.get_clip_rag()

                from ccub2_agent.modules.reference_selector import create_reference_selector

                self._reference_selector = create_reference_selector(clip_rag)

                logger.info("✓ Reference Selector initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Reference Selector: {e}")
                raise

        return self._reference_selector

    def get_prompt_adapter(self):
        """
        Get universal prompt adapter for model-specific optimization.

        Returns:
            UniversalPromptAdapter instance
        """
        if self._prompt_adapter is None:
            logger.info("Initializing Prompt Adapter...")

            try:
                from ccub2_agent.modules.prompt_adapter import get_prompt_adapter

                self._prompt_adapter = get_prompt_adapter()

                logger.info("✓ Prompt Adapter initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Prompt Adapter: {e}")
                raise

        return self._prompt_adapter

    def get_t2i_adapter(self, model_type: str):
        """
        Get T2I adapter for initial image generation.

        NOTE: T2I adapters are NOT cached because users may switch models.

        Args:
            model_type: Model type ("sdxl", "flux", "sd35", "gemini")

        Returns:
            ImageEditingAdapter instance
        """
        logger.info(f"Creating T2I adapter: {model_type}")

        try:
            from ccub2_agent.adapters.image_editing_adapter import create_adapter

            # Create new adapter (don't cache T2I)
            adapter = create_adapter(
                model_type=model_type,
                t2i_model=model_type,
            )

            logger.info(f"✓ T2I adapter created: {model_type}")
            return adapter

        except Exception as e:
            logger.error(f"Failed to create T2I adapter: {e}")
            raise

    def get_i2i_adapter(self, model_type: str):
        """
        Get I2I adapter for image editing.

        Args:
            model_type: Model type ("qwen", "flux", "sdxl", "sd35")

        Returns:
            ImageEditingAdapter instance
        """
        # Cache I2I adapter (heavy model)
        if self._i2i_adapter is None or self._current_i2i_model != model_type:
            logger.info(f"Creating I2I adapter: {model_type}")

            try:
                from ccub2_agent.adapters.image_editing_adapter import create_adapter

                self._i2i_adapter = create_adapter(model_type=model_type)
                self._current_i2i_model = model_type

                logger.info(f"✓ I2I adapter created: {model_type}")

            except Exception as e:
                logger.error(f"Failed to create I2I adapter: {e}")
                raise

        return self._i2i_adapter

    def get_gap_analyzer(self):
        """
        Get data gap analyzer.

        Returns:
            DataGapAnalyzer instance
        """
        if self._gap_analyzer is None:
            logger.info("Initializing Data Gap Analyzer...")

            try:
                from ccub2_agent.modules.gap_analyzer import DataGapAnalyzer
                from ccub2_agent.modules.country_pack import CountryDataPack

                # Load country pack
                country_pack = CountryDataPack(self.country)
                self._gap_analyzer = DataGapAnalyzer(country_pack)

                logger.info("✓ Data Gap Analyzer initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Data Gap Analyzer: {e}")
                raise

        return self._gap_analyzer

    def get_job_creator(self):
        """
        Get agent job creator.

        Returns:
            AgentJobCreator instance
        """
        if self._job_creator is None:
            logger.info("Initializing Agent Job Creator...")

            try:
                from ccub2_agent.modules.agent_job_creator import AgentJobCreator

                # Initialize without firebase config for now (will use default)
                self._job_creator = AgentJobCreator()

                logger.info("✓ Agent Job Creator initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Agent Job Creator: {e}")
                raise

        return self._job_creator

    def cleanup(self):
        """
        Cleanup all components and free GPU memory.

        Call this when pipeline is done or when switching countries.
        """
        import gc
        import torch

        logger.info("Cleaning up components...")

        # Delete components
        if self._vlm_detector:
            del self._vlm_detector
            self._vlm_detector = None

        if self._i2i_adapter:
            del self._i2i_adapter
            self._i2i_adapter = None

        if self._t2i_adapter:
            del self._t2i_adapter
            self._t2i_adapter = None

        if self._clip_rag:
            del self._clip_rag
            self._clip_rag = None

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ GPU memory cleared")

        logger.info("✓ Cleanup complete")


# Global component manager instance
_component_manager: Optional[ComponentManager] = None


def get_component_manager(country: str = "korea") -> ComponentManager:
    """
    Get or create global component manager instance.

    Args:
        country: Country code

    Returns:
        ComponentManager instance
    """
    global _component_manager

    if _component_manager is None or _component_manager.country != country:
        if _component_manager is not None:
            # Cleanup old manager before creating new one
            _component_manager.cleanup()

        _component_manager = ComponentManager(country)

    return _component_manager


def cleanup_all():
    """
    Cleanup global component manager.

    Call this on server shutdown or when switching countries.
    """
    global _component_manager

    if _component_manager is not None:
        _component_manager.cleanup()
        _component_manager = None
