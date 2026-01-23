"""
Cultural Bias Detection System

Detects cultural inaccuracies and biases in generated images using VLM.
"""

from .vlm_detector import VLMCulturalDetector, create_vlm_detector

__all__ = [
    "VLMCulturalDetector",
    "create_vlm_detector",
]
