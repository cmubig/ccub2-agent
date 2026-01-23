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
        FLUX.2 format: DETAILED, SPECIFIC

        FLUX.2 supports up to 32K tokens. Use VLM ACTION instructions.
        Priority: VLM ACTION instructions > Knowledge base > Generic instructions
        """
        prompt_parts = []

        # 1. Main action (short)
        if context:
            prompt_parts.append(f"Edit this {context.country} {context.category} image:")
        else:
            prompt_parts.append("Edit this image:")

        # 2. VLM ANALYSIS - PRIORITY #1: Extract ACTION instructions from Problem-Action pairs
        vlm_action_found = False
        if context and context.detected_issues:
            for issue in context.detected_issues:
                desc = issue.get('description', '')

                # Skip score-based generic descriptions
                if 'needs improvement' in desc and '/10)' in desc:
                    continue

                # NEW: Extract ACTION lines from VLM Problem-Action pairs
                actions = self._extract_actions_from_vlm(desc)
                if actions:
                    for action in actions[:2]:  # Top 2 actions
                        prompt_parts.append(action)
                    vlm_action_found = True
                    break

                # Fallback: Use long specific descriptions directly if no ACTION format
                elif len(desc) > 50 and not self._is_empty_instruction(desc):
                    if not any(desc.startswith(verb) for verb in ['Replace', 'Add', 'Remove', 'Change', 'Transform']):
                        desc = f"Fix: {desc}"
                    prompt_parts.append(desc)
                    vlm_action_found = True
                    break

        # 3. KNOWLEDGE BASE - PRIORITY #2
        if context and context.cultural_elements and not vlm_action_found:
            cultural_text = context.cultural_elements.strip()
            if len(cultural_text) > 50:
                prompt_parts.append(f"Apply authentic {context.country} style: {cultural_text}")

        # 4. Fallback
        if not vlm_action_found and (not context or not context.cultural_elements):
            prompt_parts.append(instruction)

        # 5. Preservation - Category-aware
        if context and context.preserve_identity:
            people_categories = {'fashion', 'clothing', 'event', 'wedding', 'people', 'portrait'}
            category_lower = (context.category or '').lower()

            if any(cat in category_lower for cat in people_categories):
                prompt_parts.append("KEEP: same person, face, pose, background.")
            else:
                prompt_parts.append("KEEP: same composition, layout, structure. Do NOT add/remove subjects.")

        final_prompt = " ".join(prompt_parts)
        logger.info(f"FLUX prompt ({len(final_prompt)} chars, {len(final_prompt.split())} words)")
        logger.info(f"Full prompt: {final_prompt}")
        return final_prompt

    def _adapt_qwen(self, instruction: str, context: Optional[EditingContext]) -> str:
        """
        Qwen format: DETAILED, SPECIFIC, ACTIONABLE

        Qwen-Image-Edit-2509 supports up to 32K tokens, optimal ~200 tokens (~800 chars).
        Priority: VLM ACTION instructions > Knowledge base > Generic instructions
        """
        prompt_parts = []

        # 1. Main action (short)
        if context:
            prompt_parts.append(f"Edit this {context.country} {context.category} image:")
        else:
            prompt_parts.append("Edit this image:")

        # 2. VLM ANALYSIS - PRIORITY #1: Extract ACTION instructions from Problem-Action pairs
        vlm_action_found = False
        if context and context.detected_issues:
            for issue in context.detected_issues:
                desc = issue.get('description', '')

                # Skip score-based generic descriptions
                if 'needs improvement' in desc and '/10)' in desc:
                    continue

                # NEW: Extract ACTION lines from VLM Problem-Action pairs
                actions = self._extract_actions_from_vlm(desc)
                if actions:
                    for action in actions[:2]:  # Top 2 actions
                        prompt_parts.append(action)
                    vlm_action_found = True
                    break

                # Fallback: Use long specific descriptions directly if no ACTION format
                elif len(desc) > 50 and not self._is_empty_instruction(desc):
                    # Try to convert problem to action if it's not already actionable
                    if not any(desc.startswith(verb) for verb in ['Replace', 'Add', 'Remove', 'Change', 'Transform']):
                        desc = f"Fix: {desc}"
                    prompt_parts.append(desc)
                    vlm_action_found = True
                    break

        # 3. KNOWLEDGE BASE - PRIORITY #2 (if VLM didn't give specifics)
        if context and context.cultural_elements and not vlm_action_found:
            cultural_text = context.cultural_elements.strip()
            if len(cultural_text) > 50:
                prompt_parts.append(f"Apply authentic {context.country} style: {cultural_text}")

        # 4. Fallback to original instruction if nothing specific
        if not vlm_action_found and (not context or not context.cultural_elements):
            prompt_parts.append(instruction)

        # 5. Preservation - Category-aware (keep short)
        if context and context.preserve_identity:
            people_categories = {'fashion', 'clothing', 'event', 'wedding', 'people', 'portrait'}
            category_lower = (context.category or '').lower()

            if any(cat in category_lower for cat in people_categories):
                prompt_parts.append("KEEP: same person, face, pose, background.")
            else:
                prompt_parts.append("KEEP: same composition, layout, structure. Do NOT add/remove subjects.")

        final_prompt = " ".join(prompt_parts)

        # Log full prompt for debugging
        logger.info(f"Qwen prompt ({len(final_prompt)} chars, {len(final_prompt.split())} words)")
        logger.info(f"Full prompt: {final_prompt}")
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

    def _extract_actions_from_vlm(self, vlm_description: str) -> List[str]:
        """
        Extract ACTION instructions from VLM's Problem-Action pairs.

        VLM generates output like:
        PROBLEM: The collar is wrong
        ACTION: Replace the collar with traditional "dongjeong" collar

        PROBLEM: Western plating style
        ACTION: Add traditional side dishes "banchan"

        We extract ONLY the ACTION lines.
        """
        import re

        actions = []

        # Pattern 1: "ACTION: ..." lines
        action_matches = re.findall(r'ACTION:\s*(.+?)(?=\n|PROBLEM:|$)', vlm_description, re.IGNORECASE | re.DOTALL)
        for match in action_matches:
            action = match.strip()
            if len(action) > 20:  # Skip too short actions
                # Clean up the action
                action = action.split('\n')[0].strip()  # Take only first line
                actions.append(action)

        # Pattern 2: Lines starting with action verbs (fallback)
        if not actions:
            action_verbs = ['Replace', 'Add', 'Remove', 'Change', 'Transform', 'Use']
            lines = vlm_description.split('\n')
            for line in lines:
                line = line.strip()
                # Remove leading "- " or "1. " etc
                line = re.sub(r'^[\-\d\.\*]+\s*', '', line)
                if any(line.startswith(verb) for verb in action_verbs) and len(line) > 20:
                    actions.append(line)

        return actions[:3]  # Max 3 actions

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

        # HANDLE GENERIC SCORE-BASED DESCRIPTIONS like "Cultural accuracy needs improvement (3/10)"
        if "needs improvement" in issue_lower and "/10)" in issue_lower:
            # This is a generic VLM score, need to generate specific instructions from RAG context
            if context.cultural_elements and len(context.cultural_elements) > 50:
                # Extract specific cultural guidance from RAG context
                return self._generate_specific_fix_from_context(context)
            else:
                # No RAG context available, use category-based generic fix
                return self._generate_category_based_fix(context)

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

    def _generate_specific_fix_from_context(self, context: EditingContext) -> str:
        """Generate specific fix instructions using RAG cultural context."""
        if not context.cultural_elements:
            return self._generate_category_based_fix(context)

        # Parse cultural context to extract actionable items
        context_text = context.cultural_elements
        fix_parts = []

        # Extract key information from context
        lines = context_text.split('\n')
        for line in lines[:5]:  # Use top 5 lines of context
            line = line.strip()
            if len(line) > 20 and not line.startswith('['):
                # Convert informational text to actionable instruction
                if any(keyword in line.lower() for keyword in ['should', 'must', 'traditional', 'authentic', 'typical']):
                    fix_parts.append(line[:200])
                    break

        if fix_parts:
            return f"Apply authentic {context.country} {context.category} style: {fix_parts[0]}"
        else:
            return self._generate_category_based_fix(context)

    def _generate_category_based_fix(self, context: EditingContext) -> str:
        """Generate category-specific fix when no RAG context available."""
        category = context.category.lower() if context.category else "general"
        country = context.country

        category_fixes = {
            "food": f"Transform to authentic {country} cuisine with traditional ingredients, plating style, and serving presentation",
            "traditional_clothing": f"Replace with authentic {country} traditional garments including proper fabric, colors, and structural elements",
            "architecture": f"Modify architectural elements to reflect authentic {country} building style, materials, and decorative features",
            "art": f"Apply traditional {country} artistic style, techniques, and cultural motifs",
            "festival": f"Add authentic {country} festival elements, decorations, and cultural symbols",
            "wedding": f"Transform to traditional {country} wedding style with appropriate attire and ceremonial elements",
            "funeral": f"Apply traditional {country} funeral/memorial customs and appropriate styling",
            "wildlife": f"Ensure accurate representation of {country}'s native species and natural habitat",
            "landmark": f"Modify to accurately represent {country}'s iconic landmarks and architectural heritage",
        }

        # Find matching category
        for key, fix in category_fixes.items():
            if key in category:
                return fix

        # Default fallback
        return f"Transform to authentic {country} {category} style with culturally accurate elements and traditional characteristics"

    def _is_empty_instruction(self, text: str) -> bool:
        """
        Check if the instruction is essentially empty or says "no issues".

        These patterns confuse I2I models and cause image degradation.
        """
        if not text:
            return True

        lower_text = text.lower()
        empty_patterns = [
            "no specific incorrect",
            "no incorrect elements",
            "there are no",
            "nothing to fix",
            "no issues",
            "no problems",
            "looks correct",
            "appears correct",
            "is correct",
            "are correct",
            "no changes needed",
            "no modifications needed",
        ]

        return any(pattern in lower_text for pattern in empty_patterns)

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
        """
        Extract key terms from cultural context.

        CHANGED: Prioritize culturally significant terms, remove filler words.
        This gives I2I models SPECIFIC guidance instead of generic terms.
        """
        # If text is short enough, return as-is
        if len(text.split()) <= max_words:
            return text

        # Remove common filler phrases that don't help I2I models
        cleaned = text
        filler_phrases = [
            "it is important to note that",
            "it should be noted that",
            "generally speaking",
            "in most cases",
            "typically",
            "usually",
            "often",
            "sometimes",
            "however",
            "therefore",
            "furthermore",
            "additionally",
            "for example",
        ]
        for phrase in filler_phrases:
            cleaned = cleaned.replace(phrase, "")

        # Split into sentences and prioritize
        sentences = [s.strip() for s in cleaned.split('.') if s.strip()]

        # Collect words, prioritizing:
        # 1. First sentence (most important context)
        # 2. Sentences with specific cultural terms (capitalized words)
        # 3. Sentences with concrete nouns

        priority_sentences = []

        # First sentence is always high priority
        if sentences:
            priority_sentences.append(sentences[0])

        # Add sentences with capitalized words (proper nouns - cultural terms)
        for sent in sentences[1:]:
            words = sent.split()
            # Check for capitalized words (excluding first word)
            if any(w[0].isupper() for w in words[1:] if w):
                priority_sentences.append(sent)
                if len(' '.join(priority_sentences).split()) >= max_words:
                    break

        # Join priority sentences
        result = '. '.join(priority_sentences)

        # If still too long, truncate to max_words
        words = result.split()
        if len(words) > max_words:
            result = ' '.join(words[:max_words])
            # Try to end at sentence boundary
            if '.' in result:
                last_period = result.rfind('.')
                if last_period > len(result) * 0.7:  # At least 70% of text
                    result = result[:last_period + 1]

        return result.strip()


# Singleton instance
_adapter = None

def get_prompt_adapter() -> UniversalPromptAdapter:
    """Get or create the global prompt adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = UniversalPromptAdapter()
    return _adapter
