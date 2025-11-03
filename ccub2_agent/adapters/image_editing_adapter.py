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
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """
        Edit image based on instruction.

        Args:
            image: Input image to edit
            instruction: Text instruction for editing
            reference_image: Optional reference image for style/content
            strength: Edit strength (0-1)
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

    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit-2509", device: str = "auto", t2i_model: str = "sdxl", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.t2i_model = t2i_model
        # Allow custom T2I model IDs
        self.t2i_model_ids = kwargs.get('t2i_model_ids', self.T2I_MODEL_IDS.copy())
        self._init_model()

    def _init_model(self):
        """Initialize Qwen model."""
        from diffusers import QwenImageEditPlusPipeline

        logger.info(f"Loading Qwen Image Editor: {self.model_name}")

        # Use bfloat16 for better memory efficiency (if CUDA available)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )

        # Enable SEQUENTIAL CPU offload for maximum memory efficiency
        if self.device == "cuda":
            logger.info("Enabling SEQUENTIAL CPU offload for maximum memory efficiency...")
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        logger.info(f"✓ Qwen model loaded on {self.device} with {dtype}")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Edit image with Qwen."""
        image = self._load_image(image)

        # Qwen supports multi-image input
        images = [image]
        if reference_image is not None:
            ref_img = self._load_image(reference_image)
            images.append(ref_img)
            instruction = f"Edit the first image based on this instruction: {instruction}. Use the second image as a reference for cultural accuracy."

        # Default parameters (optimized for memory)
        params = {
            'image': images,
            'prompt': instruction,
            'true_cfg_scale': kwargs.get('true_cfg_scale', 4.0),
            'negative_prompt': kwargs.get('negative_prompt', ' '),
            'num_inference_steps': kwargs.get('num_inference_steps', 15),  # Reduced from 40
            'guidance_scale': kwargs.get('guidance_scale', 1.0),
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        with torch.inference_mode():
            output = self.pipe(**params)

        return output.images[0]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
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

        with torch.inference_mode():
            output = t2i_pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=kwargs.get('num_inference_steps', 25 if self.t2i_model == "sdxl" else 28),
                generator=torch.manual_seed(kwargs.get('seed', 42)),
            )

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
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Edit image with SDXL ControlNet."""
        import cv2
        import numpy as np

        image = self._load_image(image)

        # Extract canny edges for ControlNet
        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        edges = Image.fromarray(edges)

        # If reference image provided, blend features
        prompt = instruction
        if reference_image is not None:
            ref_img = self._load_image(reference_image)
            prompt = f"{instruction}. Style and details similar to reference image provided."
            # In practice, use IP-Adapter or other methods to inject reference

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
        self._init_model()

    def _init_model(self):
        """Initialize Flux model."""
        from diffusers import FluxControlNetPipeline, FluxControlNetModel

        logger.info(f"Loading Flux ControlNet: {self.model_name}")

        # Load Flux ControlNet
        controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        self.pipe = FluxControlNetPipeline.from_pretrained(
            self.model_name,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        logger.info(f"✓ Flux ControlNet loaded on {self.device}")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Edit image with Flux ControlNet."""
        import cv2
        import numpy as np

        image = self._load_image(image)

        # Extract canny edges
        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        edges = Image.fromarray(edges)

        prompt = instruction
        if reference_image is not None:
            prompt = f"{instruction}. Use style and cultural elements from reference."

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
        """Generate image from prompt."""
        from diffusers import FluxPipeline

        flux_pipe = FluxPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        output = flux_pipe(
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
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Edit image with SD3.5 (img2img mode)."""
        image = self._load_image(image)

        # SD3.5 has excellent prompt understanding - use natural language
        prompt = instruction
        if reference_image is not None:
            ref_img = self._load_image(reference_image)
            prompt = f"{instruction}. Apply cultural accuracy and styling similar to reference image."

        # SD3.5 img2img parameters
        params = {
            'prompt': prompt,
            'image': image,
            'strength': strength,
            'num_inference_steps': kwargs.get('num_inference_steps', 28),
            'guidance_scale': kwargs.get('guidance_scale', 7.0),
            'generator': torch.manual_seed(kwargs.get('seed', 42)),
        }

        with torch.inference_mode():
            output = self.pipe(**params)

        return output.images[0]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
        """Generate image from prompt (T2I mode)."""
        # SD3.5 excels at natural language prompts
        # Use descriptive, concise prompts

        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=kwargs.get('num_inference_steps', 28),
                guidance_scale=kwargs.get('guidance_scale', 7.0),
                generator=torch.manual_seed(kwargs.get('seed', 42)),
            )

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
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Qwen-Image is T2I only - editing not supported."""
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
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Edit image with Gemini (text-and-image-to-image)."""
        from io import BytesIO

        image = self._load_image(image)

        # Optimize instruction for Gemini's conversational style
        optimized_instruction = self._optimize_prompt_for_gemini(instruction, is_editing=True)

        # Build content list
        contents = [optimized_instruction, image]

        # If reference image provided, add it to context
        if reference_image is not None:
            ref_img = self._load_image(reference_image)
            # Gemini can handle multiple images for style transfer/composition
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
        'sd35': SD35Editor,
        'gemini': GeminiImageEditor,
    }

    def __init__(self, model_type: str = 'qwen', model_name: Optional[str] = None, device: str = "auto", t2i_model: str = "sdxl", **kwargs):
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

        # Pass t2i_model to Qwen
        if model_type == 'qwen':
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

        logger.info(f"✓ ImageEditingAdapter initialized with {model_type} (T2I: {t2i_model if model_type == 'qwen' else 'N/A'})")

    def edit(
        self,
        image: Union[Image.Image, Path, str],
        instruction: str,
        reference_image: Optional[Union[Image.Image, Path, str]] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Image.Image:
        """Edit image based on instruction."""
        return self.editor.edit(image, instruction, reference_image, strength, **kwargs)

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, **kwargs) -> Image.Image:
        """Generate image from prompt."""
        return self.editor.generate(prompt, width, height, **kwargs)


def create_adapter(model_type: str = 'qwen', t2i_model: str = "sdxl", **kwargs) -> ImageEditingAdapter:
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
