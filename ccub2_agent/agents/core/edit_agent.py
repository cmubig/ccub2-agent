"""
Edit Agent - Model-agnostic I2I editing.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...adaptation.prompt_adapter import UniversalPromptAdapter, EditingContext
from ...editing.adapters.image_editing_adapter import create_adapter

logger = logging.getLogger(__name__)


class EditAgent(BaseAgent):
    """
    Executes model-specific prompt adaptation and I2I editing.

    Responsibilities:
    - Adapt prompts for specific I2I models
    - Execute I2I editing with references
    - Preserve non-cultural attributes
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.prompt_adapter = UniversalPromptAdapter()
        self.i2i_model = "qwen"  # Default, can be configured

    def _build_knowledge_context(self, item_knowledge: Optional[Dict]) -> str:
        """Build cultural context string from structured knowledge."""
        if not item_knowledge:
            return ""
        parts = []
        if item_knowledge.get("key_characteristics"):
            parts.append(item_knowledge["key_characteristics"])
        if item_knowledge.get("cultural_elements"):
            parts.append(f"Authentic elements: {item_knowledge['cultural_elements']}")
        if item_knowledge.get("colors_patterns"):
            parts.append(f"Correct colors/patterns: {item_knowledge['colors_patterns']}")
        if item_knowledge.get("materials_textures"):
            parts.append(f"Materials: {item_knowledge['materials_textures']}")
        if item_knowledge.get("common_mistakes"):
            mistakes = item_knowledge["common_mistakes"]
            if isinstance(mistakes, list):
                mistakes = "; ".join(mistakes[:3])
            parts.append(f"Avoid: {mistakes}")
        return "\n".join(parts)

    def _build_specific_instruction(self, issues: List[Dict], knowledge_ctx: str) -> str:
        """Build a specific editing instruction from issues + knowledge."""
        instruction_parts = []

        # Extract concrete issues
        for issue in issues:
            if isinstance(issue, dict):
                desc = issue.get("description", "")
                if desc:
                    instruction_parts.append(desc)

        if instruction_parts:
            issues_text = ". ".join(instruction_parts[:3])  # Top 3 issues
            instruction = f"Fix these cultural issues: {issues_text}"
        else:
            instruction = "Improve cultural accuracy and authenticity"

        # Add knowledge as ground truth for what "correct" looks like
        if knowledge_ctx:
            instruction += f"\n\nAuthentic reference:\n{knowledge_ctx}"

        return instruction

    def _get_adaptive_strength(self, cultural_score: float) -> float:
        """
        Fix 5: Calculate adaptive edit strength based on current cultural score.

        Lower scores need stronger edits, higher scores need gentler preservation.

        Args:
            cultural_score: Current cultural score (1-10)

        Returns:
            Edit strength (0.0-1.0)
        """
        if cultural_score <= 4:
            return 0.65  # Severe issues → strong editing
        elif cultural_score <= 6:
            return 0.55  # Moderate issues → medium editing
        else:
            return 0.40  # Minor issues → gentle editing

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Edit image using I2I model.

        Args:
            input_data: {
                "image_path": str,
                "prompt": str,
                "issues": List[Dict],
                "references": List[str] (optional),
                "country": str,
                "category": str (optional),
                "item_knowledge": Dict (optional, Fix 7),
                "model": str (optional),
                "cultural_score": float (optional, for adaptive strength),
                "iteration_number": int (optional, Fix 7)
            }

        Returns:
            AgentResult with edited image path
        """
        try:
            image_path = Path(input_data["image_path"])
            prompt = input_data["prompt"]
            issues = input_data.get("issues", [])
            references = input_data.get("references", [])
            model_type = input_data.get("model", self.i2i_model)
            item_knowledge = input_data.get("item_knowledge")
            category = input_data.get("category", self.config.category or "traditional_clothing")
            # Fix 5 & 7: New parameters
            cultural_score = input_data.get("cultural_score", 5.0)  # Default to mid-range
            iteration_number = input_data.get("iteration_number", 0)

            # Fix 7: Build cultural context from structured knowledge
            knowledge_ctx = self._build_knowledge_context(item_knowledge)
            if knowledge_ctx:
                logger.info(f"Edit agent using structured knowledge ({len(knowledge_ctx)} chars)")

            # Build specific instruction from issues + knowledge
            specific_instruction = self._build_specific_instruction(issues, knowledge_ctx)

            # Create editing context
            context = EditingContext(
                original_prompt=prompt,
                detected_issues=issues,
                cultural_elements=knowledge_ctx,
                reference_images=references,
                country=input_data.get("country", self.config.country),
                category=category,
            )

            # Adapt prompt
            editing_prompt = self.prompt_adapter.adapt(
                universal_instruction=specific_instruction,
                model_type=model_type,
                context=context
            )

            # Create I2I adapter
            i2i_adapter = create_adapter(model_type=model_type)

            # Load image
            from PIL import Image
            image = Image.open(image_path)

            # Load reference image if provided
            ref_image = None
            if references and len(references) > 0:
                ref_path = Path(references[0])
                if ref_path.exists():
                    ref_image = Image.open(ref_path)

            # Fix 5: Calculate adaptive strength based on cultural score
            adaptive_strength = self._get_adaptive_strength(cultural_score)
            logger.info(f"Using adaptive strength {adaptive_strength:.2f} for score {cultural_score:.1f} (iteration {iteration_number})")

            # Edit with adaptive strength
            edited_image = i2i_adapter.edit(
                image=image,
                instruction=editing_prompt,
                reference_image=ref_image,
                strength=adaptive_strength
            )

            # Save edited image
            output_path = self.config.output_dir / f"edited_{image_path.stem}.png" if self.config.output_dir else Path(f"edited_{image_path.stem}.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            edited_image.save(output_path)

            return AgentResult(
                success=True,
                data={
                    "output_image": str(output_path),
                    "editing_prompt": editing_prompt,
                    "model": model_type
                },
                message="Image edited successfully"
            )

        except Exception as e:
            logger.error(f"Edit execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Edit error: {str(e)}"
            )
