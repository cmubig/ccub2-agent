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
    # Iteration tracking for focused prompting
    iteration_number: int = 0
    previous_iteration_issues: Optional[List[Dict]] = None
    fixed_issues: Optional[List] = None
    remaining_issues: Optional[List[Dict]] = None
    previous_editing_prompt: Optional[str] = None


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
        Qwen format: Simple, sequential, actionable

        KEY: Qwen works best with 1-2 SIMPLE fixes at a time, not long analysis.
        """
        prompt_parts = []

        # 1. Main subject and action
        if context:
            subject = f"the {context.category} in this {context.country} image"
        else:
            subject = "the subject"

        prompt_parts.append(f"Modify {subject}.")

        # 2. Sequential fixes (MAX 2 simple instructions)
        if context and context.detected_issues:
            # Extract TOP 2 actionable fixes from VLM analysis
            detailed_issues = [i for i in context.detected_issues if i.get('is_detailed')]
            if detailed_issues:
                desc = detailed_issues[0].get('description', '')
                if desc:
                    # Convert VLM analysis to simple numbered fixes
                    simple_fixes = self._extract_simple_fixes(desc, max_fixes=2)
                    if simple_fixes:
                        prompt_parts.append(simple_fixes)
                    else:
                        # Last resort: convert first sentence to instruction
                        # FIXED: Increased from 150 to 500 chars to avoid truncation
                        first_issue = desc.split('\n')[0][:500]
                        prompt_parts.append(self._to_simple_instruction(first_issue, context))
            else:
                # Generic issues - convert to specific fixes (limit to 1 for focused editing)
                for issue in context.detected_issues[:1]:
                    desc = issue.get('description', '')
                    if desc:
                        fix = self._issue_to_specific_fix(desc, context)
                        prompt_parts.append(fix)
        else:
            # Fallback to raw instruction (simplified)
            simple_inst = instruction.split('.')[0]  # First sentence only
            prompt_parts.append(simple_inst)

        # 3. Preservation (CRITICAL for Qwen)
        if context and context.preserve_identity:
            preservation = (
                "Retain the original: facial identity, skin tone, eye color, "
                "hair style, body type, pose, hand positions, background environment, "
                "and lighting setup. Only modify the specified cultural elements."
            )
            prompt_parts.append(preservation)

        # 4. Quality directive
        prompt_parts.append("Maintain high detail, realistic textures, and cultural authenticity.")

        final_prompt = " ".join(prompt_parts)
        logger.debug(f"Qwen prompt ({len(final_prompt)} chars): {final_prompt[:200]}...")
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
        """
        Extract what needs to be culturally modified.
        Uses iteration-aware logic to focus on remaining issues.
        """
        if not context or not context.detected_issues:
            return instruction

        # Use iteration-aware instruction if we have iteration context
        if context.iteration_number > 0 and context.remaining_issues:
            return self._generate_iteration_aware_instruction(context)

        # First iteration or no iteration context: use detected issues (limit to 1 for focused editing)
        modifications = []
        for issue in context.detected_issues[:1]:
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
        issue_lower = issue_desc.lower()

        # If description is already specific and long, use it directly
        if len(issue_desc) > 80 and not issue_lower.endswith("?"):
            return issue_desc

        # Handle generic question-style descriptions using RAG context
        if "incorrect or missing" in issue_lower or "wrong" in issue_lower:
            if "traditional elements" in issue_lower:
                # Try to extract specific elements from RAG cultural_elements
                if context.cultural_elements:
                    specific_elements = self._extract_key_cultural_elements(context.cultural_elements, context.category)
                    if specific_elements:
                        return f"Add and correct: {', '.join(specific_elements[:3])}"

                return f"Add and correct traditional {context.country} {context.category} elements"
            elif "colors" in issue_lower or "shapes" in issue_lower or "proportions" in issue_lower:
                # Try to extract specific color/shape info from RAG
                if context.cultural_elements:
                    color_info = self._extract_color_info(context.cultural_elements)
                    if color_info:
                        return f"Correct colors and proportions: {color_info}"

                return f"Correct the colors, shapes, and proportions to match authentic {context.country} {context.category}"
            elif "elements from other cultures" in issue_lower:
                return f"Remove non-{context.country} cultural elements"

        # Category-specific fixes
        if context.category == "traditional_clothing":
            if "collar" in issue_lower:
                return f"Replace the collar with traditional {context.country} garment collar style"
            elif "fabric" in issue_lower or "tight" in issue_lower:
                return f"Make the fabric more flowing and layered, matching traditional {context.country} textile style"
            elif "waist" in issue_lower:
                return f"Adjust the waistline to authentic {context.country} traditional placement"

        # Generic transformations
        fix = issue_desc.replace("not", "make it").replace("lacks", "add")
        fix = fix.replace("incorrect", "correct").replace("missing", "add")
        fix = fix.replace("wrong", "fix")

        # Make it imperative if not already
        if not any(fix.startswith(word) for word in ["Add", "Correct", "Fix", "Change", "Remove", "Replace", "Make", "Adjust"]):
            fix = f"Fix: {fix}"

        return fix

    def _extract_key_cultural_elements(self, cultural_text: str, category: str) -> List[str]:
        """Extract specific cultural elements from RAG context."""
        elements = []
        lower_text = cultural_text.lower()

        # Extract key terms based on category
        if category == "traditional_clothing":
            keywords = ["collar", "sleeve", "waist", "fabric", "pattern", "embroidery", "color", "layers"]
            for keyword in keywords:
                if keyword in lower_text:
                    # Try to extract context around the keyword
                    idx = lower_text.find(keyword)
                    snippet = cultural_text[max(0, idx-20):min(len(cultural_text), idx+60)]
                    # Clean and add
                    cleaned = snippet.strip().split('.')[0].strip()
                    if len(cleaned) > 10 and len(cleaned) < 80:
                        elements.append(cleaned)

        return elements[:5]  # Return up to 5 specific elements

    def _extract_color_info(self, cultural_text: str) -> str:
        """Extract color information from RAG context."""
        lower_text = cultural_text.lower()

        # Look for color mentions
        color_keywords = ["red", "blue", "green", "yellow", "white", "black", "pink", "vibrant", "bright", "pastel"]
        found_colors = [color for color in color_keywords if color in lower_text]

        if found_colors:
            return f"use traditional colors like {', '.join(found_colors[:3])}"

        return ""

    def _generate_iteration_aware_instruction(self, context: Optional[EditingContext]) -> str:
        """
        Generate instructions that focus on UNFIXED issues from previous iterations.

        Logic:
        - Iteration 0: Address all detected issues
        - Iteration 1+: Focus on remaining_issues, acknowledge fixed_issues
        """
        if not context:
            return "Improve cultural accuracy"

        # First iteration: address all issues
        if context.iteration_number == 0 or not context.remaining_issues:
            issues_to_fix = context.detected_issues
            prefix = ""
        else:
            # Later iterations: focus on what's still wrong
            issues_to_fix = context.remaining_issues

            # Build acknowledgment of progress
            if context.fixed_issues and len(context.fixed_issues) > 0:
                prefix = f"Good progress on previous iteration. "
            else:
                prefix = "Previous attempt did not fix the issues. "

            prefix += "NOW focus on these remaining problems: "

        # Extract specific fixes from remaining issues (limit to 1 for focused editing)
        modifications = []
        for issue in issues_to_fix[:1]:  # Top 1 issue for sequential fixing
            if isinstance(issue, dict):
                desc = issue.get('description', '')
                severity = issue.get('severity', 5)

                # Prioritize severe issues
                if severity >= 8:
                    fix = self._issue_to_specific_fix(desc, context)
                    modifications.append(f"**CRITICAL**: {fix}")
                elif severity >= 5:
                    fix = self._issue_to_specific_fix(desc, context)
                    modifications.append(fix)
            else:
                modifications.append(str(issue)[:100])

        if modifications:
            return prefix + "; ".join(modifications)
        else:
            return f"Perfect the {context.country} {context.category} cultural accuracy"

    def _extract_top_fixes(self, vlm_analysis: str, max_fixes: int = 2) -> str:
        """
        Extract top N actionable fixes from VLM's detailed analysis.

        VLM gives numbered lists like:
        1. **Problem**: description
        2. **Problem**: description

        We extract the top N and simplify to SHORT instructions.
        """
        import re

        # Find numbered items
        numbered_items = re.findall(r'\d+\.\s+\*\*([^*]+)\*\*:\s*([^.]+\.)', vlm_analysis)

        if not numbered_items:
            # Try alternative format
            numbered_items = re.findall(r'\d+\.\s+([^:]+):\s*([^.]+\.)', vlm_analysis)

        if numbered_items and len(numbered_items) >= max_fixes:
            # Convert to simple instructions
            fixes = []
            for i, (problem, description) in enumerate(numbered_items[:max_fixes], 1):
                # Simplify: "Incorrect X" -> "Fix X"
                simple = self._simplify_fix(problem, description)
                fixes.append(f"{i}. {simple}")

            return " ".join(fixes)

        # Fallback: take first 200 chars
        return vlm_analysis[:200].strip()

    def _simplify_fix(self, problem: str, description: str) -> str:
        """
        Convert VLM problem to simple I2I instruction.
        Uses VLM's specific description directly - works for ANY cultural element
        (food, architecture, clothing, nature, festivals, etc.)
        """
        # Use the VLM's specific description as-is
        # The VLM already provides detailed, category-specific guidance
        full_text = f"{problem}. {description}".strip()

        # Remove only VLM filler phrases, keep all cultural details
        full_text = full_text.replace("The problem is ", "")
        full_text = full_text.replace("The issue is ", "")

        return full_text[:800].strip()

    def _extract_simple_fixes(self, vlm_analysis: str, max_fixes: int = 2) -> str:
        """
        Extract simple actionable fixes from VLM analysis.

        Handles both formats:
        - "1. Problem description"
        - "1. **Problem**: description"

        Returns: "1. Simple fix 2. Simple fix"
        """
        import re

        # Try to find numbered items (both formats)
        # Format 1: "1. **Title**: description"
        bold_items = re.findall(r'(\d+)\.\s+\*\*([^*]+)\*\*[:\s]+([^.\n]+)', vlm_analysis)

        # Format 2: "1. Regular text..."
        if not bold_items:
            # Capture full multi-line items (including bullet points) until next numbered item
            plain_items = re.findall(r'(\d+)\.\s+([^\d]+?)(?=\n\d+\.|$)', vlm_analysis, re.DOTALL)
            if plain_items:
                # Convert to (num, problem, desc) format
                bold_items = [(num, text.strip(), text) for num, text in plain_items]

        if bold_items and len(bold_items) >= 1:
            fixes = []
            for i, item in enumerate(bold_items[:max_fixes], 1):
                if len(item) == 3:
                    num, problem, desc = item
                else:
                    num, combined = item[0], item[1]
                    problem = desc = combined

                # Convert to simple fix
                simple = self._problem_to_simple_fix(problem, desc)
                fixes.append(f"{i}. {simple}")

            return " ".join(fixes)

        # Fallback: extract first meaningful sentence
        sentences = [s.strip() for s in vlm_analysis.split('.') if len(s.strip()) > 20]
        if sentences:
            return self._to_simple_instruction(sentences[0])

        return ""

    def _problem_to_simple_fix(self, problem: str, description: str = "") -> str:
        """
        Convert VLM problem description to actionable instruction.
        Uses the VLM's specific analysis directly - no pattern matching or hardcoding.
        """
        # Use the full description from VLM as-is (it's already specific and actionable)
        full_text = f"{problem}. {description}".strip()

        # Remove redundant phrases but keep the specific cultural details
        full_text = full_text.replace("The image shows ", "")
        full_text = full_text.replace("This image ", "")
        full_text = full_text.replace("I notice that ", "")

        # Return the VLM's specific description (up to 800 chars to preserve bullet points)
        return full_text[:800].strip()

    def _to_simple_instruction(self, text: str, context: Optional[EditingContext] = None) -> str:
        """Convert any text to simple editing instruction."""
        # Remove common VLM phrases
        text = text.replace("The image shows", "").replace("The garment", "garment")
        text = text.replace("is incorrect", "").replace("is wrong", "")
        text = text.strip()

        # Make it imperative
        if not text.startswith(("Change", "Fix", "Add", "Remove", "Transform", "Make")):
            if context and context.category:
                text = f"Fix the {context.category}: {text}"
            else:
                text = f"Fix: {text}"

        return text[:100].strip()

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
