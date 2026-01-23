"""
Edit Agent - Model-agnostic I2I editing.
"""

from typing import Dict, Any, Optional
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
                "model": str (optional)
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
            
            # Create editing context
            context = EditingContext(
                original_prompt=prompt,
                detected_issues=issues,
                cultural_elements="",  # Can be enhanced with RAG
                reference_images=references,
                country=input_data.get("country", self.config.country),
                category=self.config.category or "traditional_clothing"
            )
            
            # Adapt prompt
            editing_prompt = self.prompt_adapter.adapt(
                universal_instruction="Improve cultural accuracy",
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
            
            # Edit
            edited_image = i2i_adapter.edit(
                image=image,
                instruction=editing_prompt,
                reference_image=ref_image,
                strength=0.35  # Lower strength for better preservation
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
