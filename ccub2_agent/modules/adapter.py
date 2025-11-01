"""
Model-agnostic cultural correction adapter.
"""

from typing import Any, Dict, List, Optional
import logging

from .detector import CulturalDetector
from .ccub_metric import CCUBMetric
from .country_pack import CountryDataPack
from ..models.universal_interface import UniversalI2IInterface

logger = logging.getLogger(__name__)


class CulturalCorrectionAdapter:
    """
    Model-agnostic adapter for cultural bias correction.

    This adapter works with ANY I2I model through UniversalI2IInterface.
    No model training required - everything is runtime correction!
    """

    def __init__(
        self,
        model_interface: UniversalI2IInterface,
        country_pack: CountryDataPack,
        detector: CulturalDetector,
        metric: CCUBMetric,
    ):
        """
        Initialize correction adapter.

        Args:
            model_interface: Universal I2I interface (any model!)
            country_pack: Country-specific data pack
            detector: Cultural problem detector
            metric: Cultural accuracy metric
        """
        self.model = model_interface
        self.country_pack = country_pack
        self.detector = detector
        self.metric = metric

        logger.info(
            f"CulturalCorrectionAdapter initialized with model: {model_interface.model_name}"
        )

    def correct(
        self,
        image: Any,
        prompt: str,
        country: str,
        max_iterations: int = 5,
        threshold: float = 80.0,
    ) -> Dict:
        """
        Apply iterative cultural correction to an image.

        This is the core correction loop that:
        1. Detects cultural issues (VLM)
        2. Checks data availability
        3. Retrieves relevant examples
        4. Generates edit intent (LLM)
        5. Applies I2I correction
        6. Repeats until threshold met

        Args:
            image: Input image to correct
            prompt: Original prompt
            country: Target country
            max_iterations: Maximum correction iterations
            threshold: CCUB metric threshold (0-100)

        Returns:
            Dict with:
                - image: Corrected image
                - status: "corrected", "data_missing", or "threshold_not_met"
                - history: Iteration history
        """
        logger.info(f"Starting correction loop (max_iter={max_iterations})")

        history = []
        current_image = image

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # 1. Detect cultural issues
            issues = self.detector.detect(current_image, prompt, country)

            if not issues:
                logger.info("No issues detected - correction complete!")
                return {
                    "image": current_image,
                    "status": "corrected",
                    "history": history,
                }

            logger.info(f"Detected {len(issues)} issues: {issues}")

            # Check if we have relevant data
            has_data, missing_categories = self.country_pack.check_coverage(
                issues, country
            )

            if not has_data:
                logger.warning(
                    f"Missing data for categories: {missing_categories}"
                )
                return {
                    "image": current_image,
                    "status": "data_missing",
                    "missing_data": missing_categories,
                    "history": history,
                }

            # Retrieve relevant examples from country pack
            examples = self.country_pack.retrieve(issues, country, top_k=3)

            # Generate edit intent
            edit_intent = self._generate_edit_intent(
                prompt=prompt, issues=issues, examples=examples, country=country
            )

            # Apply I2I correction
            current_image = self.model.edit(
                image=current_image,
                prompt=edit_intent["enhanced_prompt"],
                reference_images=examples.get("images"),
                **edit_intent.get("model_params", {}),
            )

            # Evaluate
            score = self.metric.score(country, prompt, current_image)

            iteration_info = {
                "iteration": iteration + 1,
                "issues": issues,
                "score": score,
                "edit_intent": edit_intent,
                "examples_used": len(examples.get("data", [])),
            }
            history.append(iteration_info)

            logger.info(f"Score after iteration {iteration + 1}: {score:.2f}")

            # Check if threshold met
            if score >= threshold:
                logger.info(f"Threshold met ({score:.2f} >= {threshold})!")
                return {
                    "image": current_image,
                    "status": "corrected",
                    "history": history,
                }

        # Max iterations reached
        logger.warning(
            f"Max iterations reached. Final score: {score:.2f} < {threshold}"
        )
        return {
            "image": current_image,
            "status": "threshold_not_met",
            "history": history,
        }

    def _generate_edit_intent(
        self, prompt: str, issues: List[Dict], examples: Dict, country: str
    ) -> Dict:
        """
        Generate edit intent using LLM.

        This creates a detailed editing instruction based on:
        - Original prompt
        - Detected issues
        - Retrieved examples from country pack

        Args:
            prompt: Original prompt
            issues: Detected issues
            examples: Retrieved examples from country pack
            country: Target country

        Returns:
            Dict with:
                - enhanced_prompt: Detailed edit instruction
                - focus_areas: Areas to focus on
                - cultural_elements: Elements to add/fix
        """
        # Extract example information
        example_captions = [ex.get("caption", "") for ex in examples.get("data", [])]
        example_contexts = [
            ex.get("cultural_context", "") for ex in examples.get("data", [])
        ]

        # Construct LLM prompt
        llm_prompt = f"""You are a cultural image correction expert for {country}.

Original prompt: "{prompt}"

Detected issues:
{self._format_issues(issues)}

We have the following approved examples from {country}:
{self._format_examples(example_captions, example_contexts)}

Generate a detailed editing instruction that:
1. Preserves the original intent of "{prompt}"
2. Incorporates cultural accuracy from the examples
3. Specifically addresses each detected issue
4. Uses clear, actionable language for image editing

Respond in JSON format:
{{
  "enhanced_prompt": "Detailed editing instruction here...",
  "focus_areas": ["area1", "area2"],
  "cultural_elements_to_add": ["element1", "element2"]
}}
"""

        # TODO: Call actual LLM
        # response = call_llm(llm_prompt)
        # return parse_json(response)

        # Placeholder response
        return self._generate_edit_intent_placeholder(prompt, issues, country)

    def _format_issues(self, issues: List[Dict]) -> str:
        """Format issues for LLM prompt."""
        formatted = []
        for i, issue in enumerate(issues, 1):
            formatted.append(
                f"{i}. [{issue['type']}] {issue['category']}: {issue['description']} "
                f"(severity: {issue['severity']}/10)"
            )
        return "\n".join(formatted)

    def _format_examples(
        self, captions: List[str], contexts: List[str]
    ) -> str:
        """Format examples for LLM prompt."""
        formatted = []
        for i, (cap, ctx) in enumerate(zip(captions, contexts), 1):
            formatted.append(f"{i}. Caption: {cap}\n   Context: {ctx}")
        return "\n".join(formatted)

    def _generate_edit_intent_placeholder(
        self, prompt: str, issues: List[Dict], country: str
    ) -> Dict:
        """Placeholder edit intent generator for testing."""
        # Extract issue categories
        categories = [issue["category"] for issue in issues]

        enhanced_prompt = f"Edit this image to accurately represent {country} culture. "

        if "text" in categories:
            if country.lower() == "korea":
                enhanced_prompt += "Use proper Korean Hangul (한글) characters. "
            elif country.lower() == "japan":
                enhanced_prompt += "Use proper Japanese characters. "

        if "clothing" in categories:
            if country.lower() == "korea":
                enhanced_prompt += (
                    "Ensure traditional clothing follows authentic hanbok style "
                    "with proper jeogori (jacket) and chima (skirt). "
                )

        enhanced_prompt += f"Maintain the original intent: {prompt}"

        return {
            "enhanced_prompt": enhanced_prompt,
            "focus_areas": categories,
            "cultural_elements_to_add": [issue["description"] for issue in issues],
            "model_params": {"guidance": 8.0, "steps": 40},
        }
