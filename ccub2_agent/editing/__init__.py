"""
Image Editing System

Model-agnostic image-to-image editing with multiple I2I models.
"""

from .adapters.image_editing_adapter import ImageEditingAdapter, create_adapter
from .pipelines.iterative_editing import IterativeEditingPipeline, EditingStep, EditingResult

__all__ = [
    "ImageEditingAdapter",
    "create_adapter",
    "IterativeEditingPipeline",
    "EditingStep",
    "EditingResult",
]
