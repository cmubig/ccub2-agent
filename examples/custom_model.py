#!/usr/bin/env python3
"""
Custom Model Example: Add your own T2I or I2I model to CCUB2-Agent.

This example demonstrates how to integrate a custom image generation
or editing model using the universal adapter pattern.
"""

import sys
from pathlib import Path
from typing import List, Optional
from PIL import Image
import torch

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.adapters.image_editing_adapter import BaseImageEditor


class CustomTextToImageModel(BaseImageEditor):
    """
    Example custom T2I model implementation.

    Replace this with your actual model logic.
    """

    def __init__(self, model_path: str = None, load_in_4bit: bool = False):
        """
        Initialize your custom model.

        Args:
            model_path: Path to model weights or HuggingFace model ID
            load_in_4bit: Enable 4-bit quantization for memory efficiency
        """
        self.model_path = model_path or "your-username/your-model-name"
        self.load_in_4bit = load_in_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load your model here
        print(f"Loading custom T2I model: {self.model_path}")
        # self.model = YourModelClass.from_pretrained(self.model_path)
        # self.model.to(self.device)
        print("✓ Custom model loaded")

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        """
        Generate image from text prompt.

        Args:
            prompt: Text description of desired image
            width: Output image width
            height: Output image height

        Returns:
            Generated PIL Image
        """
        print(f"Generating image with custom T2I model...")
        print(f"  Prompt: {prompt}")
        print(f"  Size: {width}x{height}")

        # Your generation logic here
        # Example placeholder: create a solid color image
        image = Image.new("RGB", (width, height), color=(100, 150, 200))

        # Actual implementation would be something like:
        # with torch.no_grad():
        #     output = self.model(
        #         prompt=prompt,
        #         width=width,
        #         height=height,
        #     )
        # image = output.images[0]

        print("✓ Image generated")
        return image

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        reference_images: Optional[List[Image.Image]] = None,
    ) -> Image.Image:
        """
        Edit existing image based on prompt and optional references.

        Args:
            image: Input image to edit
            prompt: Editing instruction
            reference_images: Optional reference images for guidance

        Returns:
            Edited PIL Image
        """
        # Most T2I models don't support editing
        # You can either:
        # 1. Return NotImplementedError
        # 2. Implement a basic edit using img2img if your model supports it

        raise NotImplementedError("Custom T2I model does not support editing")


class CustomImageEditingModel(BaseImageEditor):
    """
    Example custom I2I model implementation.

    This shows how to integrate an image editing model.
    """

    def __init__(self, model_path: str = None, load_in_4bit: bool = False):
        """
        Initialize your custom editing model.

        Args:
            model_path: Path to model weights or HuggingFace model ID
            load_in_4bit: Enable 4-bit quantization
        """
        self.model_path = model_path or "your-username/your-editing-model"
        self.load_in_4bit = load_in_4bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading custom I2I editing model: {self.model_path}")
        # self.model = YourEditingModel.from_pretrained(self.model_path)
        # self.processor = YourProcessor.from_pretrained(self.model_path)
        # self.model.to(self.device)
        print("✓ Custom editing model loaded")

    def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
        """
        Most I2I models don't support T2I generation.

        You can either raise NotImplementedError or implement basic T2I
        by starting with a noise image.
        """
        raise NotImplementedError("Custom I2I model does not support T2I generation")

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        reference_images: Optional[List[Image.Image]] = None,
    ) -> Image.Image:
        """
        Edit image using your custom model.

        Args:
            image: Input image to edit
            prompt: Editing instruction
            reference_images: Optional reference images for guidance

        Returns:
            Edited PIL Image
        """
        print(f"Editing image with custom I2I model...")
        print(f"  Prompt: {prompt}")
        print(f"  References: {len(reference_images) if reference_images else 0}")

        # Your editing logic here
        # Example placeholder: return the original image
        edited_image = image.copy()

        # Actual implementation might look like:
        # inputs = self.processor(
        #     images=image,
        #     text=prompt,
        #     reference_images=reference_images,
        #     return_tensors="pt"
        # ).to(self.device)
        #
        # with torch.no_grad():
        #     output = self.model(**inputs)
        #
        # edited_image = self.processor.decode(output)

        print("✓ Image edited")
        return edited_image


def register_custom_model():
    """
    Register your custom model with the adapter factory.

    After registration, you can use it like any built-in model:
    create_adapter(model_type="my_custom_t2i")
    """
    from ccub2_agent.adapters.image_editing_adapter import IMAGE_EDITOR_REGISTRY

    # Register T2I model
    IMAGE_EDITOR_REGISTRY["my_custom_t2i"] = CustomTextToImageModel

    # Register I2I model
    IMAGE_EDITOR_REGISTRY["my_custom_i2i"] = CustomImageEditingModel

    print("✓ Custom models registered")
    print("  Available: my_custom_t2i, my_custom_i2i")


def demo_custom_model():
    """
    Demonstrate using a custom model in the full pipeline.
    """
    print("=" * 80)
    print("Custom Model Integration Demo")
    print("=" * 80)
    print()

    # Register custom models
    register_custom_model()
    print()

    # Now you can use them like any other model
    from ccub2_agent.adapters.image_editing_adapter import create_adapter

    print("Creating custom T2I adapter...")
    t2i_adapter = create_adapter(model_type="my_custom_t2i")
    print("✓ Custom T2I adapter created")
    print()

    print("Generating image with custom T2I model...")
    image = t2i_adapter.generate(
        prompt="A traditional Korean hanbok",
        width=512,
        height=512,
    )
    image.save("output/custom_t2i_example.png")
    print("✓ Image saved to: output/custom_t2i_example.png")
    print()

    print("Creating custom I2I adapter...")
    i2i_adapter = create_adapter(model_type="my_custom_i2i")
    print("✓ Custom I2I adapter created")
    print()

    print("Editing image with custom I2I model...")
    edited_image = i2i_adapter.edit(
        image=image,
        prompt="Add more vibrant colors to the hanbok",
    )
    edited_image.save("output/custom_i2i_example.png")
    print("✓ Edited image saved to: output/custom_i2i_example.png")
    print()

    print("=" * 80)
    print("✓ Custom model demo completed!")
    print("=" * 80)


def integration_checklist():
    """
    Print checklist for integrating a new model.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       Custom Model Integration Checklist                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. Implement BaseImageEditor interface:
   ✓ Inherit from ccub2_agent.adapters.image_editing_adapter.BaseImageEditor
   ✓ Implement generate() method (for T2I models)
   ✓ Implement edit() method (for I2I models)
   ✓ Add __init__() with load_in_4bit support

2. Model Loading:
   ✓ Load model in __init__()
   ✓ Handle GPU/CPU device placement
   ✓ Support 4-bit quantization if possible
   ✓ Cache model to avoid reloading

3. Prompt Handling:
   ✓ Accept prompt as string
   ✓ Adapt prompt format to your model's requirements
   ✓ Handle long prompts gracefully (truncate or split)

4. Reference Images (for I2I):
   ✓ Accept Optional[List[Image.Image]]
   ✓ Decide how to use references (concatenate, average, etc.)
   ✓ Handle case when references=None

5. Image Processing:
   ✓ Accept PIL Image.Image objects
   ✓ Return PIL Image.Image objects
   ✓ Handle resizing if needed
   ✓ Ensure consistent color space (RGB)

6. Error Handling:
   ✓ Handle CUDA out of memory errors
   ✓ Provide helpful error messages
   ✓ Log warnings for unsupported features

7. Registration:
   ✓ Register model in IMAGE_EDITOR_REGISTRY
   ✓ Choose a unique model_type name
   ✓ Document usage in docstrings

8. Testing:
   ✓ Test generate() with various prompts
   ✓ Test edit() with different images
   ✓ Test with references if applicable
   ✓ Test on GPU and CPU
   ✓ Test with 4-bit quantization

9. Documentation:
   ✓ Add docstrings to all methods
   ✓ Document model requirements (VRAM, dependencies)
   ✓ Provide usage example
   ✓ Update README.md with new model

10. Optional Enhancements:
    ○ Add model-specific parameters
    ○ Support negative prompts
    ○ Add style/strength controls
    ○ Implement batch processing

Example Usage After Integration:
    from ccub2_agent.adapters.image_editing_adapter import create_adapter

    adapter = create_adapter(model_type="your_model_name")
    image = adapter.generate("your prompt")
    edited = adapter.edit(image, "editing instruction")

For more details, see:
- ccub2_agent/adapters/image_editing_adapter.py (base classes)
- ccub2_agent/adapters/qwen_image_editor.py (example implementation)
- CONTRIBUTING.md (development guidelines)
    """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Custom model integration example")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with placeholder custom models",
    )
    parser.add_argument(
        "--checklist",
        action="store_true",
        help="Show integration checklist",
    )

    args = parser.parse_args()

    if args.checklist:
        integration_checklist()
    elif args.demo:
        demo_custom_model()
    else:
        # Show both by default
        integration_checklist()
        print("\n\n")
        demo_custom_model()
