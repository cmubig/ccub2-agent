"""
Model-Agnostic Prompt Adapter

Automatically adapts editing instructions to model-specific prompt formats
while maintaining cultural accuracy requirements.

Supported models:
- FLUX.1 Kontext (context-preserving edits)
- Qwen Image Edit (text rendering, semantic understanding)
- Stable Diffusion 3.5 (balanced, versatile)
- HiDream (artistic style)
- NextStep (realistic edits)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EditingContext:
    """Context for editing operation."""
    original_prompt: str
    detected_issues: List[Dict]
    cultural_elements: str  # From RAG
    reference_images: Optional[List[str]]
    country: str
    category: str
    preserve_identity: bool = True


class UniversalPromptAdapter:
    """
    Model-agnostic prompt adapter that automatically formats instructions
    according to each model's optimal prompting style.

    Philosophy: One universal instruction → Multiple model-specific prompts
    """

    def __init__(self):
        self.model_configs = self._load_model_configs()
        logger.info("Initialized UniversalPromptAdapter with support for 6+ models")

    def adapt(
        self,
        universal_instruction: str,
        model_type: str,
        context: Optional[EditingContext] = None
    ) -> str:
        """
        Convert universal editing instruction to model-specific prompt.

        Args:
            universal_instruction: Generic editing instruction
            model_type: Target model (flux, qwen, sd35, hidream, nextstep)
            context: Optional cultural and editing context

        Returns:
            Model-optimized prompt string
        """
        if model_type not in self.model_configs:
            logger.warning(f"Unknown model {model_type}, using generic format")
            return universal_instruction

        config = self.model_configs[model_type]
        adapter_fn = getattr(self, f"_adapt_{model_type}", None)

        if adapter_fn:
            return adapter_fn(universal_instruction, context)
        else:
            return self._adapt_generic(universal_instruction, config, context)

    def _load_model_configs(self) -> Dict:
        """Load model-specific configuration."""
        return {
            "flux": {
                "name": "FLUX.1 Kontext Dev",
                "strengths": ["context_preservation", "character_consistency"],
                "max_tokens": 512,
                "requires_explicit_preservation": True,
                "style": "instruction_based"
            },
            "qwen": {
                "name": "Qwen Image Edit 2509",
                "strengths": ["text_rendering", "semantic_understanding", "multilingual"],
                "optimal_steps": 50,
                "cfg_scale": 4.0,
                "style": "detailed_specific",
                "supports_languages": ["en", "zh", "ko", "ja", "it"]
            },
            "sd35": {
                "name": "Stable Diffusion 3.5 Medium",
                "strengths": ["balanced", "versatile", "quality_modifiers"],
                "cfg_scale": 7.0,
                "style": "structured_tags"
            },
            "hidream": {
                "name": "HiDream-E1.1",
                "strengths": ["artistic", "style_transfer", "high_detail"],
                "cfg_scale": 7.5,
                "style": "artistic_descriptive"
            },
            "nextstep": {
                "name": "NextStep-1-Large-Edit",
                "strengths": ["realistic", "photographic", "fine_detail"],
                "cfg_scale": 7.0,
                "style": "natural_language"
            }
        }

    def _adapt_flux(self, instruction: str, context: Optional[EditingContext]) -> str:
        """
        FLUX Kontext format: [Action] + [Preservation] + [Details]

        FLUX excels at context preservation and character consistency.
        Key: Explicit preservation statements are CRITICAL.
        """
        # Extract cultural changes
        cultural_mods = self._extract_cultural_modifications(instruction, context)

        # FLUX formula: Simple action + Explicit preservation
        prompt_parts = []

        # 1. Main action (brief and clear)
        if cultural_mods:
            action = f"Modify the {context.category if context else 'clothing'}: {cultural_mods}"
        else:
            action = instruction.split('.')[0]  # First sentence

        prompt_parts.append(action)

        # 2. Preservation (FLUX CRITICAL!)
        if context and context.preserve_identity:
            preservation = (
                "while maintaining the exact same person, face, facial features, "
                "pose, body proportions, background, and overall composition"
            )
            prompt_parts.append(preservation)

        # 3. Cultural specifics (FLUX has 512 token limit)
        if context and context.cultural_elements:
            key_terms = self._extract_keywords(context.cultural_elements, max_words=30)
            if key_terms:
                prompt_parts.append(f"Ensure authentic {context.country} style: {key_terms}")

        final_prompt = " ".join(prompt_parts)

        # Enforce token limit
        if len(final_prompt.split()) > 512:
            final_prompt = " ".join(final_prompt.split()[:512])
            logger.warning("FLUX prompt truncated to 512 tokens")

        logger.debug(f"FLUX prompt: {final_prompt}")
        return final_prompt

    def _adapt_qwen(self, instruction: str, context: Optional[EditingContext]) -> str:
        """
        Qwen format: Detailed, specific, structured

        Qwen excels at:
        - Text rendering (use quotes for text)
        - Complex semantic understanding
        - Preserving specific attributes
        """
        prompt_parts = []

        # 1. Main subject and action
        if context:
            subject = f"the {context.category} in this {context.country} image"
        else:
            subject = "the subject"

        prompt_parts.append(f"Modify {subject}.")

        # 2. Specific changes (Qwen loves details!)
        if context and context.detected_issues:
            for issue in context.detected_issues[:3]:
                desc = issue.get('description', '')
                if desc:
                    fix = self._issue_to_specific_fix(desc, context)
                    prompt_parts.append(fix)
        else:
            prompt_parts.append(instruction)

        # 3. Cultural requirements (structured)
        if context and context.cultural_elements:
            cultural_desc = context.cultural_elements
            prompt_parts.append(f"Cultural requirements: {cultural_desc}")

        # 4. Preservation directives (Qwen-style: specific attributes)
        if context and context.preserve_identity:
            preservation = (
                "Retain the original: facial identity, skin tone, eye color, "
                "hair style, body type, pose, hand positions, background environment, "
                "and lighting setup. Only modify the specified cultural elements."
            )
            prompt_parts.append(preservation)

        # 5. Quality enhancement (Qwen magic prompt)
        prompt_parts.append("Maintain high detail, realistic textures, and cultural authenticity.")

        final_prompt = " ".join(prompt_parts)
        logger.debug(f"Qwen prompt: {final_prompt}")
        return final_prompt

    def _adapt_sd35(self, instruction: str, context: Optional[EditingContext]) -> str:
        """
        SD 3.5 format: Structured with quality tags

        Formula: [Source] + [Transformation] + [Preservation] + [Style] + [Quality]
        """
        prompt_parts = []

        # 1. Source acknowledgment
        if context:
            prompt_parts.append(f"Transform this {context.country} {context.category} image:")

        # 2. Transformation
        prompt_parts.append(instruction)

        # 3. Preservation
        if context and context.preserve_identity:
            prompt_parts.append("preserving facial identity and expression,")

        # 4. Style/Cultural
        if context and context.cultural_elements:
            style_desc = self._extract_keywords(context.cultural_elements, max_words=20)
            prompt_parts.append(f"in authentic {context.country} style: {style_desc},")

        # 5. Quality modifiers (SD loves these!)
        quality_tags = [
            "highly detailed",
            "realistic textures",
            "proper lighting",
            "professional photography",
            "8K resolution",
            "masterpiece quality"
        ]
        prompt_parts.append(", ".join(quality_tags[:4]))

        final_prompt = " ".join(prompt_parts)
        logger.debug(f"SD3.5 prompt: {final_prompt}")
        return final_prompt

    def _adapt_hidream(self, instruction: str, context: Optional[EditingContext]) -> str:
        """
        HiDream format: Artistic, style-focused

        HiDream excels at artistic transformations and style transfer.
        """
        prompt_parts = []

        # 1. Artistic framing
        if context:
            prompt_parts.append(f"Artistically enhance the {context.category} to reflect authentic {context.country} aesthetics.")

        # 2. Specific changes (artistic language)
        prompt_parts.append(instruction.replace("Change", "Transform").replace("Fix", "Refine"))

        # 3. Cultural style
        if context and context.cultural_elements:
            prompt_parts.append(f"Embody traditional {context.country} artistic principles: {context.cultural_elements[:100]}")

        # 4. Artistic quality
        prompt_parts.append("High artistic detail, harmonious composition, authentic cultural representation.")

        final_prompt = " ".join(prompt_parts)
        logger.debug(f"HiDream prompt: {final_prompt}")
        return final_prompt

    def _adapt_nextstep(self, instruction: str, context: Optional[EditingContext]) -> str:
        """
        NextStep format: Natural language, photorealistic

        NextStep prefers natural language instructions.
        """
        prompt_parts = []

        # Natural language style
        if context:
            prompt_parts.append(f"Please modify the {context.category} in this photograph to be more culturally accurate for {context.country}.")

        prompt_parts.append(instruction)

        if context and context.preserve_identity:
            prompt_parts.append("Keep everything else exactly the same, especially the person's face and overall appearance.")

        if context and context.cultural_elements:
            prompt_parts.append(f"The result should look like authentic {context.country} {context.category} as seen in real life.")

        final_prompt = " ".join(prompt_parts)
        logger.debug(f"NextStep prompt: {final_prompt}")
        return final_prompt

    def _adapt_generic(self, instruction: str, config: Dict, context: Optional[EditingContext]) -> str:
        """Fallback for unknown models."""
        return instruction

    # Helper methods

    def _extract_cultural_modifications(self, instruction: str, context: Optional[EditingContext]) -> str:
        """Extract what needs to be culturally modified."""
        if not context or not context.detected_issues:
            return instruction

        modifications = []
        for issue in context.detected_issues[:3]:
            desc = issue.get('description', '')
            if desc:
                if "missing" in issue.get('type', ''):
                    mod = desc.replace("Missing", "Add").replace("missing", "add")
                elif "incorrect" in issue.get('type', ''):
                    mod = desc.replace("Incorrect", "Fix").replace("incorrect", "fix")
                else:
                    mod = desc
                modifications.append(mod)

        return ", ".join(modifications) if modifications else instruction

    def _issue_to_specific_fix(self, issue_desc: str, context: EditingContext) -> str:
        """Convert issue description to specific fix instruction."""
        if context.category == "traditional_clothing":
            if "collar" in issue_desc.lower():
                return f"Replace the collar with traditional {context.country} garment collar style."
            elif "fabric" in issue_desc.lower() or "tight" in issue_desc.lower():
                return f"Make the fabric more flowing and layered, matching traditional {context.country} textile style."
            elif "waist" in issue_desc.lower():
                return f"Adjust the waistline to authentic {context.country} traditional placement."

        return issue_desc.replace("not", "make it").replace("lacks", "add")

    def _extract_keywords(self, text: str, max_words: int = 20) -> str:
        """Extract key terms from cultural context."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."


# Singleton instance
_adapter = None

def get_prompt_adapter() -> UniversalPromptAdapter:
    """Get or create the global prompt adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = UniversalPromptAdapter()
    return _adapter
