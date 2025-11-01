"""
Model-agnostic adapters for image generation and editing.
"""

from .image_editing_adapter import ImageEditingAdapter, create_adapter

__all__ = ['ImageEditingAdapter', 'create_adapter']
