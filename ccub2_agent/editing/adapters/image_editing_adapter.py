"""
Model-Agnostic Image Editing Adapter

Provides unified interface for different I2I models:
- Qwen-Image-Edit (I2I)
- Qwen-Image (T2I only, text rendering specialist)
- SDXL with ControlNet
- Flux with ControlNet
- Stable Diffusion 3.5 Medium (T2I + I2I)
- Gemini 2.5 Flash Image (Nano Banana) (T2I + I2I, API-based)

Key features:
- Unified edit() interface
- Reference image support
- Instruction-based editing
- Model-specific optimizations
- API key management for cloud models
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from abc import ABC, abstractmethod
import logging
import os

from PIL import Image
import torch

logger = logging.getLogger(__name__)


class BaseImageEditor(ABC):
    """Base class for all image editing models."""

    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.kwargs = kwargs

    def _get_device(self, device: str) -> str:
        """Get device (auto/cuda/cpu)."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @abstractmethod
    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """
        Edit image based on instruction with optional cultural guidance.

        Args:
            image: Input image
            instruction: Editing instruction
            reference_image: Reference image path (optional)
            reference_metadata: RAG metadata with cultural knowledge (optional)
            strength: Edit strength 0-1
            **kwargs: Model-specific parameters

        Returns:
            Edited image
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> Image.Image:
        """
        Generate image from text prompt.

        Args:
            prompt: Text prompt
            width: Image width
            height: Image height
            **kwargs: Model-specific parameters

        Returns:
            Generated image
        """
        pass

    def _load_image(self, image: Union[Image.Image, Path, str]) -> Image.Image:
        """Load image from path or return PIL Image."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(image).convert('RGB')


class QwenImageEditor(BaseImageEditor):
    """Qwen-Image-Edit-2509 adapter."""

    # T2I model IDs (can be overridden via kwargs)
    T2I_MODEL_IDS = {
        "flux": "black-forest-labs/FLUX.1-dev",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    }

    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit-2509", device: str = "auto", t2i_model: str = "sd35", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.t2i_model = t2i_model
        # Allow custom T2I model IDs
        self.t2i_model_ids = kwargs.get('t2i_model_ids', self.T2I_MODEL_IDS.copy())
        self._init_model()

    def _init_model(self):
        """Initialize Qwen model."""
        from diffusers import QwenImageEditPlusPipeline

        logger.info(f"Loading Qwen Image Editor Plus (2509): {self.model_name}")

        # Use bfloat16 for better memory efficiency (if CUDA available)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )

        # Use CPU offload to save GPU memory (needed when VLM is also loaded)
        if self.device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
            logger.info("✓ Qwen model loaded with sequential CPU offload")
        else:
            self.pipe = self.pipe.to(self.device)

        logger.info(f"✓ Qwen model loaded on {self.device} with {dtype}")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> Image.Image:
        """
        Edit image with Qwen using TEXT-ONLY cultural guidance.

        IMPORTANT: Reference images are NOT passed to the model to preserve original composition.
        Only text metadata (descriptions, key_features) from RAG is used for cultural guidance.

        Args:
            image: Input image to edit
            instruction: Editing instruction
            reference_image: Reference image path (used for metadata lookup only, NOT passed to model)
            reference_metadata: RAG metadata with cultural knowledge (descriptions, key_features)
            strength: Editing strength (not directly used by Qwen pipeline)
            progress_callback: Optional callback for progress updates (step, total_steps)
            **kwargs: Additional parameters
        """
        image = self._load_image(image)

        # TEXT-ONLY CULTURAL GUIDANCE: Use metadata descriptions, NOT the reference image
        # This preserves original composition while adding cultural context
        if reference_metadata is not None:
            # Extract issue keywords to filter relevant cultural info
            issue_keywords = self._extract_issue_keywords(instruction) if instruction else set()

            # Get the most detailed description
            full_description = (
                reference_metadata.get('description_enhanced') or
                reference_metadata.get('description') or
                reference_metadata.get('caption', '')
            )

            # Smart filtering: Extract only relevant sentences
            relevant_description = self._filter_relevant_sentences(
                full_description, issue_keywords
            ) if issue_keywords and full_description else full_description

            # Extract only relevant key features (matching issue keywords)
            all_key_features = reference_metadata.get('key_features', [])
            relevant_features = [
                feat for feat in all_key_features
                if any(keyword in feat.lower() for keyword in issue_keywords)
            ] if issue_keywords else all_key_features[:5]  # Max 5 features

            # Build focused cultural context from TEXT ONLY
            if relevant_description or relevant_features:
                reference_context = "\n\n[CULTURAL REFERENCE - Apply these authentic elements]:"

                if relevant_description:
                    # Use more description for better guidance (up to 300 chars)
                    truncated_desc = relevant_description[:300] + "..." if len(relevant_description) > 300 else relevant_description
                    reference_context += f"\n{truncated_desc}"

                if relevant_features:
                    features_str = ', '.join(relevant_features[:5])  # Max 5
                    reference_context += f"\nKey authentic features: {features_str}"

                instruction = instruction + reference_context
                logger.info(f"✓ Added TEXT-ONLY cultural guidance ({len(relevant_features)} features, {len(relevant_description)} chars)")
                if issue_keywords:
                    logger.debug(f"Issue keywords: {list(issue_keywords)[:5]}")
            else:
                logger.warning("⚠ No relevant cultural information found in metadata")

            if reference_image is not None:
                logger.info(f"ℹ Reference image path provided but NOT passed to model (text-only mode)")
        elif reference_image is not None:
            logger.warning("⚠ Reference image provided but no metadata - skipping (text-only mode requires metadata)")

        # SINGLE IMAGE MODE ONLY: Preserve original composition
        image_input = image

        num_steps = kwargs.get('num_inference_steps', 40)

        # CFG scale: Moderate value for balanced editing
        # 3.0 for subtle, composition-preserving edits
        default_cfg = 3.0

        params = {
            'image': image_input,
            'prompt': instruction,
            'true_cfg_scale': kwargs.get('true_cfg_scale', default_cfg),
            'guidance_scale': kwargs.get('guidance_scale', 1.0),
            'negative_prompt': kwargs.get('negative_prompt', ' '),
            'num_inference_steps': num_steps,
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        # Add progress callback if provided
        if progress_callback is not None:
            def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
                progress_callback(step_index + 1, num_steps)
                return callback_kwargs
            params['callback_on_step_end'] = diffusers_callback

        with torch.inference_mode():
            output = self.pipe(**params)

        return output.images[0]

    def _extract_issue_keywords(self, instruction: str) -> set:
        """Extract keywords from issues to filter relevant cultural info."""
        # Common stopwords to exclude
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'image', 'fix', 'improve', 'needs',
            'improvement', 'accuracy', 'cultural', 'traditional', 'authentic'
        }

        # Extract words (3+ chars, not stopwords)
        words = instruction.lower().split()
        keywords = {
            word.strip('.,!?():;')
            for word in words
            if len(word) > 2 and word.lower() not in stopwords
        }

        return keywords

    def _extract_reference_description(self, ref_img, reference_metadata: Optional[Dict] = None) -> str:
        """
        Extract detailed visual description from reference image metadata.

        This provides text-based visual guidance instead of passing the image directly,
        preventing the model from copying the reference.

        Args:
            ref_img: Reference PIL Image (not used, kept for consistency)
            reference_metadata: Metadata dict with descriptions and features

        Returns:
            Detailed text description of the reference image
        """
        description_parts = []

        if reference_metadata:
            # Extract enhanced description (preferred)
            enhanced_desc = reference_metadata.get('description_enhanced')
            if enhanced_desc:
                description_parts.append(enhanced_desc)
            else:
                # Fallback to regular description
                regular_desc = reference_metadata.get('description', '')
                if regular_desc:
                    description_parts.append(regular_desc)

            # Add key features
            key_features = reference_metadata.get('key_features', [])
            if key_features:
                features_text = "Key visual elements: " + ", ".join(key_features[:5])
                description_parts.append(features_text)

            # Add category context
            category = reference_metadata.get('category', '')
            if category:
                category_display = category.replace('_', ' ').title()
                description_parts.insert(0, f"Reference type: {category_display}")

        # Combine all parts
        if description_parts:
            full_description = "\n".join(description_parts)
            logger.debug(f"Extracted reference description: {len(full_description)} chars")
            return full_description
        else:
            logger.warning("No description found in reference metadata")
            return ""

    def _filter_relevant_sentences(self, text: str, keywords: set, max_sentences: int = 2) -> str:
        if not text or not keywords:
            return text

        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]

        scored = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in keywords if kw in sent_lower)
            if score > 0:
                scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sent for _, sent in scored[:max_sentences]]

        return '. '.join(top_sentences) + '.' if top_sentences else text[:200]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, progress_callback: Optional[callable] = None, **kwargs) -> Image.Image:
        """
        Generate image from prompt (T2I mode).

        Note: Qwen-Image-Edit is I2I only, so we use a separate T2I model.
        """
        logger.info(f"Qwen-Image-Edit is I2I only. Using {self.t2i_model.upper()} for T2I generation...")
        from diffusers import DiffusionPipeline, FluxPipeline

        # Select T2I model
        if self.t2i_model == "flux":
            model_id = self.t2i_model_ids.get("flux", "black-forest-labs/FLUX.1-dev")
            logger.info(f"Loading Flux for T2I: {model_id}")
            t2i_pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
        else:  # default to sdxl
            model_id = self.t2i_model_ids.get("sdxl", "stabilityai/stable-diffusion-xl-base-1.0")
            logger.info(f"Loading SDXL for T2I: {model_id}")
            t2i_pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )

        if self.device == "cuda":
            t2i_pipe.enable_sequential_cpu_offload()  # Sequential for max memory efficiency
        else:
            t2i_pipe = t2i_pipe.to(self.device)

        # Reduce resolution for memory efficiency
        width = min(width, 768)
        height = min(height, 768)

        num_steps = kwargs.get('num_inference_steps', 25 if self.t2i_model == "sdxl" else 28)

        params = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'num_inference_steps': num_steps,
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        # Add progress callback if provided
        if progress_callback is not None:
            def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
                progress_callback(step_index + 1, num_steps)
                return callback_kwargs
            params['callback_on_step_end'] = diffusers_callback

        with torch.inference_mode():
            output = t2i_pipe(**params)

        # Clean up T2I pipe
        del t2i_pipe
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return output.images[0]


class SDXLControlNetEditor(BaseImageEditor):
    """SDXL with ControlNet adapter."""

    # Default ControlNet model (can be overridden)
    DEFAULT_CONTROLNET = "diffusers/controlnet-canny-sdxl-1.0"

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.controlnet_model = kwargs.get('controlnet_model', self.DEFAULT_CONTROLNET)
        self._init_model()

    def _init_model(self):
        """Initialize SDXL with ControlNet."""
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

        logger.info(f"Loading SDXL ControlNet: {self.model_name}")

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Load SDXL pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_name,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        logger.info(f"✓ SDXL ControlNet loaded on {self.device}")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        import cv2
        import numpy as np

        image = self._load_image(image)

        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        edges = Image.fromarray(edges)

        logger.info(f"✓ SDXL ControlNet: Using Canny edge for structure preservation")

        prompt = instruction
        if reference_image is not None:
            ref_img = self._load_image(reference_image)
            prompt = f"{instruction}. Style and details similar to reference image provided."
            logger.info(f"✓ SDXL: Reference guidance via text prompt (ControlNet scale={strength:.2f})")
        else:
            logger.info(f"✓ SDXL: Single image editing with Canny ControlNet (scale={strength:.2f})")

        params = {
            'prompt': prompt,
            'image': edges,
            'num_inference_steps': kwargs.get('num_inference_steps', 30),
            'controlnet_conditioning_scale': strength,
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        output = self.pipe(**params)
        return output.images[0]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
        """Generate image from prompt."""
        # Use base SDXL for generation
        from diffusers import DiffusionPipeline

        base_pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        output = base_pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=kwargs.get('num_inference_steps', 30),
            generator=torch.manual_seed(kwargs.get('seed', 42)),
        )

        return output.images[0]


class FluxControlNetEditor(BaseImageEditor):
    """Flux with ControlNet adapter."""

    # Default ControlNet model (can be overridden)
    DEFAULT_CONTROLNET = "InstantX/FLUX.1-dev-Controlnet-Canny"

    def __init__(self, model_name: str = "black-forest-labs/FLUX.1-dev", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.controlnet_model = kwargs.get('controlnet_model', self.DEFAULT_CONTROLNET)
        self.t2i_model = kwargs.get('t2i_model', 'flux')
        self._init_model()

    def _init_model(self):
        """Initialize Flux model."""
        from diffusers import FluxControlNetPipeline, FluxControlNetModel

        logger.info(f"Loading Flux ControlNet: {self.model_name}")

        controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            token=True,
        )

        self.pipe = FluxControlNetPipeline.from_pretrained(
            self.model_name,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            token=True,
        )

        if self.device == "cuda":
            self.pipe.enable_sequential_cpu_offload()
            logger.info("✓ Flux ControlNet loaded with SEQUENTIAL CPU offload (memory optimized)")
        else:
            self.pipe = self.pipe.to(self.device)
            logger.info(f"✓ Flux ControlNet loaded on {self.device}")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        import cv2
        import numpy as np

        image = self._load_image(image)

        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges = Image.fromarray(edges_rgb)

        logger.info(f"✓ FLUX ControlNet: Using Canny edge for structure preservation")

        prompt = instruction
        if reference_image is not None:
            prompt = f"{instruction}. Use style and cultural elements from reference."
            logger.info(f"✓ FLUX: Reference guidance via text prompt (ControlNet strength={strength:.2f})")
        else:
            logger.info(f"✓ FLUX: Single image editing with Canny ControlNet (strength={strength:.2f})")

        params = {
            'prompt': prompt,
            'control_image': edges,
            'num_inference_steps': kwargs.get('num_inference_steps', 28),
            'controlnet_conditioning_scale': strength,
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        output = self.pipe(**params)
        return output.images[0]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
        """Generate image from prompt using specified T2I model."""
        if self.t2i_model == 'sd35':
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium",
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=True,
            )

            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)

            output = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=kwargs.get('num_inference_steps', 40),
                guidance_scale=kwargs.get('guidance_scale', 4.5),
                generator=torch.manual_seed(kwargs.get('seed', 42)),
            )
        elif self.t2i_model == 'sdxl':
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=True,
            )

            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)

            output = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=kwargs.get('num_inference_steps', 40),
                guidance_scale=kwargs.get('guidance_scale', 7.5),
                generator=torch.manual_seed(kwargs.get('seed', 42)),
            )
        else:
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                token=True,
            )

            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)

            output = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=kwargs.get('num_inference_steps', 28),
                generator=torch.manual_seed(kwargs.get('seed', 42)),
            )

        return output.images[0]


class SD35Editor(BaseImageEditor):
    """Stable Diffusion 3.5 Medium adapter (T2I + I2I)."""

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-3.5-medium", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._init_model()

    def _init_model(self):
        """Initialize SD3.5 Medium model."""
        from diffusers import StableDiffusion3Pipeline

        logger.info(f"Loading SD3.5 Medium: {self.model_name}")

        # Use bfloat16 for better memory efficiency
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )

        # Enable SEQUENTIAL CPU offload for maximum memory efficiency
        if self.device == "cuda":
            logger.info("Enabling SEQUENTIAL CPU offload for SD3.5...")
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        logger.info(f"✓ SD3.5 Medium loaded on {self.device} with {dtype}")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> Image.Image:
        image = self._load_image(image)

        prompt = instruction
        if reference_image is not None:
            ref_img = self._load_image(reference_image)
            prompt = f"{instruction}. Apply cultural accuracy and styling similar to reference image."
            logger.info(f"✓ SD3.5: Using text-based reference guidance (strength={strength:.2f})")
        else:
            logger.info(f"✓ SD3.5: Single image editing (strength={strength:.2f})")

        num_steps = kwargs.get('num_inference_steps', 28)

        params = {
            'prompt': prompt,
            'image': image,
            'strength': strength,
            'num_inference_steps': num_steps,
            'guidance_scale': kwargs.get('guidance_scale', 7.0),
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        # Add progress callback if provided
        if progress_callback is not None:
            def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
                progress_callback(step_index + 1, num_steps)
                return callback_kwargs
            params['callback_on_step_end'] = diffusers_callback

        with torch.inference_mode():
            output = self.pipe(**params)

        return output.images[0]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, progress_callback: Optional[callable] = None, **kwargs) -> Image.Image:
        """Generate image from prompt (T2I mode)."""
        # SD3.5 excels at natural language prompts
        # Use descriptive, concise prompts

        num_steps = kwargs.get('num_inference_steps', 28)

        params = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'num_inference_steps': num_steps,
            'guidance_scale': kwargs.get('guidance_scale', 7.0),
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        # Add progress callback if provided
        if progress_callback is not None:
            def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
                progress_callback(step_index + 1, num_steps)
                return callback_kwargs
            params['callback_on_step_end'] = diffusers_callback

        with torch.inference_mode():
            output = self.pipe(**params)

        return output.images[0]


class QwenT2IEditor(BaseImageEditor):
    """Qwen-Image (T2I only) - specialized for text rendering."""

    def __init__(self, model_name: str = "Qwen/Qwen-Image", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._init_model()

    def _init_model(self):
        """Initialize Qwen-Image T2I model."""
        from diffusers import DiffusionPipeline

        logger.info(f"Loading Qwen-Image T2I: {self.model_name}")

        # Use bfloat16 for better quality and memory efficiency
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )

        # Enable SEQUENTIAL CPU offload for maximum memory efficiency
        if self.device == "cuda":
            logger.info("Enabling SEQUENTIAL CPU offload for Qwen-Image...")
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        # Qwen-Image specific prompt magic for better quality
        self.positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.",
            "zh": ", 超清，4K，电影级构图.",
        }

        logger.info(f"✓ Qwen-Image T2I loaded on {self.device} with {dtype}")

    def _detect_language(self, prompt: str) -> str:
        """Detect if prompt is primarily Chinese or English."""
        import re
        # Simple heuristic: check if more than 30% of characters are Chinese
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', prompt))
        total_chars = len(prompt.strip())
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return "zh"
        return "en"

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        raise NotImplementedError(
            "Qwen-Image is a text-to-image model only and does not support image editing. "
            "Use 'qwen' (Qwen-Image-Edit) for I2I editing instead."
        )

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
        """Generate image from prompt with Qwen-Image's text rendering capabilities."""

        # Detect language and add appropriate magic prompt
        lang = self._detect_language(prompt)
        enhanced_prompt = prompt + self.positive_magic[lang]

        # Determine aspect ratio from width/height
        # Qwen-Image supports: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }

        # Find closest aspect ratio
        target_ratio = width / height
        closest_ratio = "1:1"
        min_diff = float('inf')

        for ratio_name, (w, h) in aspect_ratios.items():
            ratio_val = w / h
            diff = abs(ratio_val - target_ratio)
            if diff < min_diff:
                min_diff = diff
                closest_ratio = ratio_name

        width, height = aspect_ratios[closest_ratio]

        logger.info(f"Qwen-Image generating ({closest_ratio}): {enhanced_prompt[:80]}...")

        with torch.inference_mode():
            output = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=kwargs.get('negative_prompt', ' '),  # empty string as default
                width=width,
                height=height,
                num_inference_steps=kwargs.get('num_inference_steps', 50),
                true_cfg_scale=kwargs.get('true_cfg_scale', 4.0),
                generator=torch.manual_seed(kwargs.get('seed', 42)),
            )

        return output.images[0]


class GeminiImageEditor(BaseImageEditor):
    """Gemini 2.5 Flash Image (Nano Banana) adapter."""

    def __init__(self, model_name: str = "gemini-2.5-flash-image", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._check_api_key()
        self._init_model()

    def _check_api_key(self):
        """Check if GOOGLE_API_KEY is set."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "\n" + "="*80 + "\n"
                "GOOGLE_API_KEY not found!\n\n"
                "To use Gemini image generation, you need to set your API key:\n\n"
                "1. Get your API key from: https://aistudio.google.com/apikey\n"
                "2. Set the environment variable:\n"
                "   export GOOGLE_API_KEY='your-api-key-here'\n\n"
                "Or add it to your ~/.bashrc or ~/.zshrc file.\n"
                + "="*80
            )
        logger.info("✓ GOOGLE_API_KEY found")

    def _init_model(self):
        """Initialize Gemini client."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Google Generative AI SDK not installed. Install with:\n"
                "pip install google-genai"
            )

        logger.info(f"Loading Gemini Image: {self.model_name}")

        self.client = genai.Client()
        self.types = types

        logger.info(f"✓ Gemini Image ready")

    def _optimize_prompt_for_gemini(self, prompt: str, is_editing: bool = False) -> str:
        """
        Optimize prompt for Gemini's narrative understanding.

        Gemini excels at descriptive, narrative prompts rather than keyword lists.
        """
        if is_editing:
            # For editing: be specific about changes while preserving context
            return f"Using the provided image, {prompt}. Ensure the change integrates naturally with the original style, lighting, and composition."
        else:
            # For generation: Add photorealistic details if not already present
            if "photorealistic" not in prompt.lower() and "photo" not in prompt.lower():
                # Add camera/photography context for realism
                return f"A photorealistic image of {prompt}. Captured with professional photography, natural lighting, sharp focus, and rich detail."
            return prompt

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        from io import BytesIO

        image = self._load_image(image)

        optimized_instruction = self._optimize_prompt_for_gemini(instruction, is_editing=True)

        contents = [optimized_instruction, image]

        if reference_image is not None:
            ref_img = self._load_image(reference_image)
            contents.append(ref_img)
            contents[0] = f"{instruction}. Use styling and cultural elements from the additional reference image provided."

        # Configure aspect ratio
        aspect_ratio = kwargs.get('aspect_ratio', '1:1')

        config = self.types.GenerateContentConfig(
            response_modalities=['Image'],  # Only return image, no text
            image_config=self.types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )
        )

        logger.info(f"Gemini editing with prompt: {contents[0][:100]}...")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                edited_image = Image.open(BytesIO(part.inline_data.data))
                return edited_image

        raise RuntimeError("No image generated in Gemini response")

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
        """Generate image from prompt (text-to-image)."""
        from io import BytesIO

        # Optimize prompt for Gemini
        optimized_prompt = self._optimize_prompt_for_gemini(prompt, is_editing=False)

        # Determine aspect ratio from width/height
        if width == height:
            aspect_ratio = '1:1'
        elif width > height:
            ratio = width / height
            if abs(ratio - 16/9) < 0.1:
                aspect_ratio = '16:9'
            elif abs(ratio - 3/2) < 0.1:
                aspect_ratio = '3:2'
            elif abs(ratio - 4/3) < 0.1:
                aspect_ratio = '4:3'
            else:
                aspect_ratio = '16:9'  # Default wide
        else:
            ratio = height / width
            if abs(ratio - 16/9) < 0.1:
                aspect_ratio = '9:16'
            elif abs(ratio - 3/2) < 0.1:
                aspect_ratio = '2:3'
            elif abs(ratio - 4/3) < 0.1:
                aspect_ratio = '3:4'
            else:
                aspect_ratio = '9:16'  # Default tall

        aspect_ratio = kwargs.get('aspect_ratio', aspect_ratio)

        config = self.types.GenerateContentConfig(
            response_modalities=['Image'],
            image_config=self.types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )
        )

        logger.info(f"Gemini generating with prompt: {optimized_prompt[:100]}...")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[optimized_prompt],
            config=config,
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                generated_image = Image.open(BytesIO(part.inline_data.data))
                return generated_image

        raise RuntimeError("No image generated in Gemini response")


class Flux2Editor(BaseImageEditor):
    """FLUX.2-dev adapter - Latest SOTA model for I2I editing."""

    def __init__(self, model_name: str = "black-forest-labs/FLUX.2-dev", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._init_model()

    def _init_model(self):
        """Initialize FLUX.2 model."""
        from diffusers import Flux2Pipeline

        logger.info(f"Loading FLUX.2-dev: {self.model_name}")

        # Use bfloat16 for efficiency
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.pipe = Flux2Pipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            token=True,  # Use HF token for gated model
        )

        # Enable memory optimization
        if self.device == "cuda":
            logger.info("Enabling CPU offload for FLUX.2...")
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        logger.info(f"✓ FLUX.2-dev loaded on {self.device} with {dtype}")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> Image.Image:
        """
        Edit image with FLUX.2-dev using TEXT-ONLY cultural guidance.

        IMPORTANT: Reference images are NOT passed to the model to preserve original composition.
        Only text metadata from RAG is used for cultural guidance.
        """
        image = self._load_image(image)

        # SINGLE IMAGE MODE ONLY: Preserve original composition
        image_list = [image]
        logger.info(f"✓ FLUX.2: Single image editing (text-only cultural guidance)")

        # TEXT-ONLY cultural context from metadata
        if reference_metadata:
            desc = reference_metadata.get('description_enhanced') or reference_metadata.get('description', '')
            key_features = reference_metadata.get('key_features', [])

            if desc or key_features:
                cultural_context = "\n\n[CULTURAL REFERENCE - Apply these authentic elements]:"
                if desc:
                    cultural_context += f"\n{desc[:300]}"
                if key_features:
                    cultural_context += f"\nKey features: {', '.join(key_features[:5])}"
                instruction = instruction + cultural_context
                logger.info(f"✓ Added TEXT-ONLY cultural guidance")

        if reference_image is not None:
            logger.info(f"ℹ Reference image path provided but NOT passed to model (text-only mode)")

        num_steps = kwargs.get('num_inference_steps', 50)

        params = {
            'prompt': instruction,
            'image': image_list,
            'num_inference_steps': num_steps,
            'guidance_scale': kwargs.get('guidance_scale', 4.0),
            'generator': torch.Generator(device='cpu').manual_seed(kwargs.get('seed', 42)),
        }

        # Add progress callback if provided
        if progress_callback is not None:
            def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
                progress_callback(step_index + 1, num_steps)
                return callback_kwargs
            params['callback_on_step_end'] = diffusers_callback

        with torch.inference_mode():
            output = self.pipe(**params)

        return output.images[0]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, progress_callback: Optional[callable] = None, **kwargs) -> Image.Image:
        """Generate image from prompt (T2I mode)."""
        num_steps = kwargs.get('num_inference_steps', 50)

        params = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'num_inference_steps': num_steps,
            'guidance_scale': kwargs.get('guidance_scale', 4.0),
            'generator': torch.Generator(device='cpu').manual_seed(kwargs.get('seed', 42)),
        }

        # Add progress callback if provided
        if progress_callback is not None:
            def diffusers_callback(pipe, step_index, timestep, callback_kwargs):
                progress_callback(step_index + 1, num_steps)
                return callback_kwargs
            params['callback_on_step_end'] = diffusers_callback

        with torch.inference_mode():
            output = self.pipe(**params)

        return output.images[0]


class ImageEditingAdapter:
    """
    Model-agnostic image editing adapter.

    Automatically selects and wraps the appropriate model.
    """

    SUPPORTED_MODELS = {
        'qwen': QwenImageEditor,
        'qwen-t2i': QwenT2IEditor,
        'sdxl': SDXLControlNetEditor,
        'flux': FluxControlNetEditor,
        'flux2': Flux2Editor,
        'sd35': SD35Editor,
        'gemini': GeminiImageEditor,
    }

    def __init__(self, model_type: str = 'qwen', model_name: Optional[str] = None, device: str = "auto", t2i_model: str = "sd35", **kwargs):
        """
        Initialize adapter.

        Args:
            model_type: I2I model type ('qwen', 'qwen-t2i', 'sdxl', 'flux', 'sd35', 'gemini')
            model_name: Optional model name (uses default if None)
            device: Device to use (not applicable for 'gemini')
            t2i_model: T2I model for Qwen ('sdxl', 'flux', or 'qwen-t2i')
                Note: 'qwen-t2i' is T2I only, excellent for text rendering
            **kwargs: Model-specific parameters
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(self.SUPPORTED_MODELS.keys())}")

        editor_class = self.SUPPORTED_MODELS[model_type]

        # Pass t2i_model to Qwen and Flux
        if model_type in ['qwen', 'flux']:
            if model_name:
                self.editor = editor_class(model_name=model_name, device=device, t2i_model=t2i_model, **kwargs)
            else:
                self.editor = editor_class(device=device, t2i_model=t2i_model, **kwargs)
        else:
            if model_name:
                self.editor = editor_class(model_name=model_name, device=device, **kwargs)
            else:
                self.editor = editor_class(device=device, **kwargs)

        self.model_type = model_type
        self.t2i_model = t2i_model

        # Log with clear role distinction
        if model_type in ['qwen', 'flux']:
            logger.info(f"✓ ImageEditingAdapter initialized:")
            logger.info(f"   - I2I Editing Model: {model_type} (for iterative improvements)")
            logger.info(f"   - T2I Generation Model: {t2i_model} (for initial image only)")
        else:
            logger.info(f"✓ ImageEditingAdapter initialized: {model_type} (T2I + I2I capable)")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        reference_metadata: Optional[Dict[str, Any]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        return self.editor.edit(image, instruction, reference_image, reference_metadata, strength, **kwargs)

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
        """Generate image from prompt."""
        return self.editor.generate(prompt, width, height, **kwargs)


def create_adapter(model_type: str = 'qwen', t2i_model: str = "sd35", **kwargs) -> ImageEditingAdapter:
    """
    Factory function to create image editing adapter.

    Args:
        model_type: I2I model type ('qwen', 'qwen-t2i', 'sdxl', 'flux', 'sd35', or 'gemini')
        t2i_model: T2I model for Qwen ('sdxl', 'flux', 'qwen-t2i', default: 'sdxl')
            Note: 'sd35', 'gemini', and 'qwen-t2i' have built-in T2I
        **kwargs: Model-specific parameters

    Returns:
        ImageEditingAdapter instance

    Notes:
        - 'gemini' requires GOOGLE_API_KEY environment variable
        - 'qwen-t2i' is T2I only, specialized for text rendering (especially Chinese)
        - Install google-genai: pip install google-genai
    """
    return ImageEditingAdapter(model_type=model_type, t2i_model=t2i_model, **kwargs)
