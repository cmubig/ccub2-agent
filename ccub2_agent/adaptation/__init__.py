"""
Prompt Adaptation System

Adapts universal editing instructions to model-specific formats.
"""

from .prompt_adapter import UniversalPromptAdapter, EditingContext, get_prompt_adapter

__all__ = [
    "UniversalPromptAdapter",
    "EditingContext",
    "get_prompt_adapter",
]
