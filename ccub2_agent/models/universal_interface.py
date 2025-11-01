"""
Universal I2I interface for model-agnostic operation.
"""

from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseI2IModel(ABC):
    """Abstract base class for I2I models."""

    @abstractmethod
    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Any:
        """Edit image based on prompt."""
        pass

    @abstractmethod
    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """Generate image from text."""
        pass


class FluxKontextWrapper(BaseI2IModel):
    """Wrapper for FLUX.1 Kontext (dev)"""

    def __init__(self):
        logger.info("Loading FLUX.1 Kontext model...")
        # TODO: Implement actual model loading
        # from diffusers import FluxKontextPipeline
        # self.pipe = FluxKontextPipeline.from_pretrained(...)
        self.pipe = None  # Placeholder

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Any:
        """Edit with FLUX Kontext."""
        guidance_scale = kwargs.get("guidance", 7.5)
        num_steps = kwargs.get("steps", 30)

        # TODO: Implement actual editing
        logger.info(f"FLUX editing: {prompt}")
        return image  # Placeholder

    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """Generate with FLUX."""
        # TODO: Implement T2I
        logger.info(f"FLUX T2I: {prompt}")
        return None  # Placeholder


class HiDreamWrapper(BaseI2IModel):
    """Wrapper for HiDream-E1.1"""

    def __init__(self):
        logger.info("Loading HiDream model...")
        self.model = None  # Placeholder

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Any:
        """Edit with HiDream."""
        cfg_scale = kwargs.get("guidance", 7.5)

        logger.info(f"HiDream editing: {prompt}")
        return image  # Placeholder

    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """Generate with HiDream."""
        logger.info(f"HiDream T2I: {prompt}")
        return None  # Placeholder


class SD35Wrapper(BaseI2IModel):
    """Wrapper for Stable Diffusion 3.5 Medium"""

    def __init__(self):
        logger.info("Loading SD3.5 model...")
        self.pipe = None  # Placeholder

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Any:
        """Edit with SD3.5."""
        logger.info(f"SD3.5 editing: {prompt}")
        return image  # Placeholder

    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """Generate with SD3.5."""
        logger.info(f"SD3.5 T2I: {prompt}")
        return None  # Placeholder


class QwenEditWrapper(BaseI2IModel):
    """Wrapper for Qwen-Image-Edit"""

    def __init__(self):
        logger.info("Loading Qwen-Image-Edit model...")
        self.model = None  # Placeholder

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Any:
        """Edit with Qwen."""
        logger.info(f"Qwen editing: {prompt}")
        return image  # Placeholder

    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """Generate with Qwen."""
        logger.info(f"Qwen T2I: {prompt}")
        return None  # Placeholder


class NextStepWrapper(BaseI2IModel):
    """Wrapper for NextStep-1-Large-Edit"""

    def __init__(self):
        logger.info("Loading NextStep model...")
        self.model = None  # Placeholder

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Any:
        """Edit with NextStep."""
        logger.info(f"NextStep editing: {prompt}")
        return image  # Placeholder

    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """Generate with NextStep."""
        logger.info(f"NextStep T2I: {prompt}")
        return None  # Placeholder


class UniversalI2IInterface:
    """
    Universal interface for ANY I2I model.

    This enables model-agnostic operation - just change the model name
    and everything else works the same!
    """

    # Registry of supported models
    _MODEL_REGISTRY: Dict[str, Callable[[], BaseI2IModel]] = {
        "flux-kontext": lambda: FluxKontextWrapper(),
        "flux": lambda: FluxKontextWrapper(),  # Alias
        "hidream": lambda: HiDreamWrapper(),
        "sd3.5": lambda: SD35Wrapper(),
        "sd35": lambda: SD35Wrapper(),  # Alias
        "qwen-edit": lambda: QwenEditWrapper(),
        "qwen": lambda: QwenEditWrapper(),  # Alias
        "nextstep": lambda: NextStepWrapper(),
    }

    def __init__(self, model_name: str):
        """
        Initialize with any supported model.

        Args:
            model_name: Name of the I2I model
        """
        if model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Available models: {list(self._MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        logger.info(f"Initializing UniversalI2IInterface with '{model_name}'")
        self.model = self._MODEL_REGISTRY[model_name]()

    @classmethod
    def register_model(cls, name: str, wrapper_class: Callable[[], BaseI2IModel]):
        """
        Register a new model wrapper.

        This allows extending to new models without modifying core code!

        Example:
            >>> class MyNewModel(BaseI2IModel):
            ...     def edit(self, image, prompt, **kwargs):
            ...         # Custom implementation
            ...         pass
            >>>
            >>> UniversalI2IInterface.register_model("my-new-model", MyNewModel)
        """
        cls._MODEL_REGISTRY[name] = wrapper_class
        logger.info(f"Registered new model: {name}")

    def edit(
        self,
        image: Any,
        prompt: str,
        reference_images: Optional[List] = None,
        **kwargs,
    ) -> Any:
        """
        Universal edit interface.

        Args:
            image: Input image
            prompt: Edit instruction
            reference_images: Optional reference images (if model supports)
            **kwargs: Model-specific parameters

        Returns:
            Edited image
        """
        return self.model.edit(image, prompt, reference_images, **kwargs)

    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """
        Universal T2I interface.

        Args:
            prompt: Text prompt
            **kwargs: Model-specific parameters

        Returns:
            Generated image
        """
        return self.model.text_to_image(prompt, **kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available models."""
        return list(cls._MODEL_REGISTRY.keys())
