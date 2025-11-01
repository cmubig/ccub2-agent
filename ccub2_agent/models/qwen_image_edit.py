"""
Qwen-Image-Edit-2509 wrapper for image generation and editing.

Qwen-Image-Edit-2509 is the state-of-the-art model for image editing.
https://huggingface.co/Qwen/Qwen-Image-Edit-2509
"""

from pathlib import Path
from typing import Optional, Union
import logging

from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline

logger = logging.getLogger(__name__)


class QwenImageEditor:
    """
    Wrapper for Qwen-Image-Edit-2509 model.

    Provides both:
    - Text-to-Image generation (for initial images)
    - Image-to-Image editing (for iterative refinement)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-Image-Edit-2509",
        device: str = "auto",
        torch_dtype=torch.float16,
    ):
        """
        Initialize Qwen Image Editor.

        Args:
            model_name: Model name from HuggingFace
            device: Device to use (auto/cuda/cpu)
            torch_dtype: Torch dtype for model
        """
        logger.info(f"Initializing Qwen Image Editor: {model_name}")

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {torch_dtype}")

        # Load pipeline
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            logger.info("  Enabled attention slicing")

        logger.info("✓ Qwen Image Editor loaded successfully")

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate image from text prompt (T2I).

        Args:
            prompt: Text description
            width: Output width
            height: Output height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        logger.info(f"Generating image: '{prompt[:60]}...'")

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        image = result.images[0]
        logger.info(f"  Generated: {image.size}")

        return image

    def edit(
        self,
        prompt: str,
        image: Union[Image.Image, Path, str],
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Edit existing image based on text prompt (I2I).

        Args:
            prompt: Editing instruction
            image: Input image (PIL Image or path)
            strength: Editing strength (0-1, higher = more change)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for editing
            seed: Random seed

        Returns:
            Edited PIL Image
        """
        logger.info(f"Editing image: '{prompt[:60]}...'")

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
            logger.info(f"  Loaded image: {image.size}")

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Edit
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        edited_image = result.images[0]
        logger.info(f"  Edited: {edited_image.size}")

        return edited_image

    def batch_edit(
        self,
        prompts: list[str],
        images: list[Union[Image.Image, Path]],
        **kwargs,
    ) -> list[Image.Image]:
        """
        Edit multiple images in batch.

        Args:
            prompts: List of editing prompts
            images: List of input images
            **kwargs: Additional arguments for edit()

        Returns:
            List of edited images
        """
        if len(prompts) != len(images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match images ({len(images)})")

        results = []
        for prompt, image in zip(prompts, images):
            edited = self.edit(prompt=prompt, image=image, **kwargs)
            results.append(edited)

        return results


class ImageGeneratorWrapper:
    """
    Unified interface for image generation.

    Wraps Qwen-Image-Edit for use in iterative editing pipeline.
    """

    def __init__(self, qwen_editor: QwenImageEditor):
        """
        Initialize wrapper.

        Args:
            qwen_editor: QwenImageEditor instance
        """
        self.editor = qwen_editor

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """Generate image from prompt."""
        return self.editor.generate(prompt=prompt, **kwargs)

    def edit(self, prompt: str, image_path: Path, **kwargs) -> Image.Image:
        """Edit image based on prompt."""
        return self.editor.edit(prompt=prompt, image=image_path, **kwargs)


def create_qwen_editor(
    model_name: str = "Qwen/Qwen-Image-Edit-2509",
    device: str = "auto",
    torch_dtype=torch.float16,
) -> QwenImageEditor:
    """
    Factory function to create Qwen Image Editor.

    Args:
        model_name: Model name from HuggingFace
        device: Device (auto/cuda/cpu)
        torch_dtype: Torch dtype

    Returns:
        QwenImageEditor instance
    """
    return QwenImageEditor(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )


def create_image_generator_wrapper(
    qwen_editor: Optional[QwenImageEditor] = None,
) -> ImageGeneratorWrapper:
    """
    Create unified image generator wrapper.

    Args:
        qwen_editor: QwenImageEditor instance (creates new if None)

    Returns:
        ImageGeneratorWrapper instance
    """
    if qwen_editor is None:
        qwen_editor = create_qwen_editor()

    return ImageGeneratorWrapper(qwen_editor)
