"""
Universal I2I interface - IMPLEMENTATION VERSION
실제 모델 로딩 코드 포함

이 파일은 실제 모델을 로드하는 구현 버전입니다.
프로덕션 환경에서는 이 파일을 universal_interface.py로 교체하세요.
"""

from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseI2IModel(ABC):
    """Abstract base class for I2I models."""

    def __init__(self):
        """Initialize base model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def optimize_memory(self, pipe):
        """
        Apply memory optimizations to pipeline.
        모델 불가지론적 메모리 최적화.
        """
        if not torch.cuda.is_available():
            return pipe

        try:
            # Try CPU offload first (best for large models)
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
                logger.info("✓ Enabled CPU offload")
            elif hasattr(pipe, 'enable_sequential_cpu_offload'):
                pipe.enable_sequential_cpu_offload()
                logger.info("✓ Enabled sequential CPU offload")
            else:
                # Fallback: move to device
                pipe.to(self.device)

            # Additional optimizations
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
                logger.info("✓ Enabled attention slicing")

            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
                logger.info("✓ Enabled VAE slicing")

        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            # Fallback to basic device placement
            pipe.to(self.device)

        return pipe

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
        super().__init__()  # Initialize base class
        logger.info("Loading FLUX.1 Kontext model...")
        try:
            from diffusers import FluxKontextPipeline

            self.pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev",
                torch_dtype=self.dtype,
            )

            # 공통 메모리 최적화 사용
            self.pipe = self.optimize_memory(self.pipe)

            logger.info("✓ FLUX.1 Kontext loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FLUX.1 Kontext: {e}")
            raise

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Image.Image:
        """Edit with FLUX Kontext."""
        guidance_scale = kwargs.get("guidance", 7.5)
        num_steps = kwargs.get("steps", 30)

        result = self.pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        return result.images[0]

    def text_to_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate with FLUX (uses edit mode without input image)."""
        result = self.pipe(
            prompt=prompt,
            guidance_scale=kwargs.get("guidance", 7.5),
            num_inference_steps=kwargs.get("steps", 30),
        )
        return result.images[0]


class HiDreamWrapper(BaseI2IModel):
    """Wrapper for HiDream-E1.1"""

    def __init__(self):
        super().__init__()
        logger.info("Loading HiDream-E1.1 model...")
        try:
            from diffusers import DiffusionPipeline

            self.pipe = DiffusionPipeline.from_pretrained(
                "HiDream-ai/HiDream-E1-1",
                torch_dtype=self.dtype,
                use_safetensors=True,
            )

            # 공통 메모리 최적화 사용
            self.pipe = self.optimize_memory(self.pipe)

            logger.info("✓ HiDream-E1.1 loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load HiDream: {e}")
            logger.info("Installing flash-attention may help: pip install flash-attn")
            raise

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Image.Image:
        """Edit with HiDream."""
        result = self.pipe(
            image=image,
            prompt=prompt,
            height=kwargs.get("height", 1024),
            width=kwargs.get("width", 1024),
        )
        return result.images[0]

    def text_to_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate with HiDream."""
        result = self.pipe(
            prompt=prompt,
            height=kwargs.get("height", 1024),
            width=kwargs.get("width", 1024),
        )
        return result.images[0]


class SD35Wrapper(BaseI2IModel):
    """Wrapper for Stable Diffusion 3.5 Medium"""

    def __init__(self):
        super().__init__()
        logger.info("Loading Stable Diffusion 3.5 Medium...")
        try:
            from diffusers import (
                StableDiffusion3Pipeline,
                StableDiffusion3Img2ImgPipeline,
            )

            # T2I 파이프라인
            self.pipe_t2i = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium", torch_dtype=self.dtype
            )

            # I2I 파이프라인 (편집용)
            self.pipe_i2i = StableDiffusion3Img2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium", torch_dtype=self.dtype
            )

            # 공통 메모리 최적화 사용
            self.pipe_t2i = self.optimize_memory(self.pipe_t2i)
            self.pipe_i2i = self.optimize_memory(self.pipe_i2i)

            logger.info("✓ SD3.5 Medium loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SD3.5: {e}")
            raise

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Image.Image:
        """Edit with SD3.5 (Img2Img)."""
        result = self.pipe_i2i(
            image=image,
            prompt=prompt,
            strength=kwargs.get("strength", 0.75),
            guidance_scale=kwargs.get("guidance", 7.5),
            num_inference_steps=kwargs.get("steps", 50),
        )
        return result.images[0]

    def text_to_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate with SD3.5."""
        result = self.pipe_t2i(
            prompt=prompt,
            guidance_scale=kwargs.get("guidance", 4.5),
            num_inference_steps=kwargs.get("steps", 28),
        )
        return result.images[0]


class QwenEditWrapper(BaseI2IModel):
    """Wrapper for Qwen-Image-Edit"""

    def __init__(self):
        logger.info("Loading Qwen-Image-Edit-2509...")
        try:
            # Qwen-Image-Edit은 특별한 설치가 필요할 수 있음
            from qwen_image import QwenImageEditPipeline

            self.pipe = QwenImageEditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.float16
            )
            self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            logger.info("✓ Qwen-Image-Edit loaded successfully")
        except ImportError:
            logger.error(
                "qwen-image package not found. Install with: pip install qwen-image"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen-Image-Edit: {e}")
            raise

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Image.Image:
        """Edit with Qwen."""
        edit_mode = kwargs.get("edit_mode", "both")  # "semantic", "appearance", "both"

        result = self.pipe(
            image=image,
            prompt=prompt,
            edit_mode=edit_mode,
            preserve_style=kwargs.get("preserve_style", True),
        )
        return result.images[0]

    def text_to_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate with Qwen."""
        result = self.pipe(prompt=prompt)
        return result.images[0]


class NextStepWrapper(BaseI2IModel):
    """Wrapper for NextStep-1-Large-Edit"""

    def __init__(self):
        logger.info("Loading NextStep-1-Large-Edit...")
        try:
            from diffusers import NextStepPipeline

            self.pipe = NextStepPipeline.from_pretrained(
                "stepfun-ai/NextStep-1-Large-Edit", torch_dtype=torch.float16
            )
            self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            logger.info("✓ NextStep-1-Large-Edit loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NextStep: {e}")
            raise

    def edit(
        self, image: Any, prompt: str, reference_images: Optional[List] = None, **kwargs
    ) -> Image.Image:
        """Edit with NextStep (returns 2 images, we return the first)."""
        result = self.pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=kwargs.get("steps", 50),
        )
        # NextStep returns 2 images, return the first one
        return result.images[0]

    def text_to_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate with NextStep."""
        result = self.pipe(
            prompt=prompt, num_inference_steps=kwargs.get("steps", 50)
        )
        return result.images[0]


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

        try:
            self.model = self._MODEL_REGISTRY[model_name]()
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            logger.info(
                "\nTroubleshooting tips:"
                "\n1. Check MODEL_SOURCES.md for installation instructions"
                "\n2. Ensure you have enough GPU memory (see requirements)"
                "\n3. Install required packages: pip install diffusers transformers accelerate"
            )
            raise

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


# 편의 함수
def create_interface(
    model_name: str, use_cpu: bool = False, optimize_memory: bool = True
) -> UniversalI2IInterface:
    """
    편리한 인터페이스 생성 함수.

    Args:
        model_name: 모델 이름
        use_cpu: CPU 사용 강제 (기본: GPU 자동 감지)
        optimize_memory: 메모리 최적화 활성화

    Returns:
        UniversalI2IInterface 인스턴스
    """
    if use_cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    interface = UniversalI2IInterface(model_name)

    if optimize_memory and torch.cuda.is_available():
        logger.info("Applying memory optimizations...")
        # 최적화는 각 wrapper에서 자동 적용됨

    return interface
