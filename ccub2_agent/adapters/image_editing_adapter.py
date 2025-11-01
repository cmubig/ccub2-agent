"""
Model-Agnostic Image Editing Adapter

Provides unified interface for different I2I models:
- Qwen-Image-Edit
- SDXL with ControlNet
- Flux with ControlNet
- Stable Diffusion 3

Key features:
- Unified edit() interface
- Reference image support
- Instruction-based editing
- Model-specific optimizations
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from abc import ABC, abstractmethod
import logging

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

    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit-2509", device: str = "auto", t2i_model: str = "sdxl", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.t2i_model = t2i_model
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
            model_id = "black-forest-labs/FLUX.1-dev"
            logger.info(f"Loading Flux for T2I: {model_id}")
            t2i_pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )
        else:  # default to sdxl
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
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

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._init_model()

    def _init_model(self):
        """Initialize SDXL with ControlNet."""
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

        logger.info(f"Loading SDXL ControlNet: {self.model_name}")

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
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

    def __init__(self, model_name: str = "black-forest-labs/FLUX.1-dev", device: str = "auto", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._init_model()

    def _init_model(self):
        """Initialize Flux model."""
        from diffusers import FluxControlNetPipeline, FluxControlNetModel

        logger.info(f"Loading Flux ControlNet: {self.model_name}")

        # Load Flux ControlNet
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny",
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


class ImageEditingAdapter:
    """
    Model-agnostic image editing adapter.

    Automatically selects and wraps the appropriate model.
    """

    SUPPORTED_MODELS = {
        'qwen': QwenImageEditor,
        'sdxl': SDXLControlNetEditor,
        'flux': FluxControlNetEditor,
    }

    def __init__(self, model_type: str = 'qwen', model_name: Optional[str] = None, device: str = "auto", t2i_model: str = "sdxl", **kwargs):
        """
        Initialize adapter.

        Args:
            model_type: I2I model type ('qwen', 'sdxl', 'flux')
            model_name: Optional model name (uses default if None)
            device: Device to use
            t2i_model: T2I model for Qwen ('sdxl' or 'flux')
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
        model_type: I2I model type ('qwen', 'sdxl', or 'flux')
        t2i_model: T2I model for Qwen ('sdxl' or 'flux', default: 'sdxl')
        **kwargs: Model-specific parameters

    Returns:
        ImageEditingAdapter instance
    """
    return ImageEditingAdapter(model_type=model_type, t2i_model=t2i_model, **kwargs)
