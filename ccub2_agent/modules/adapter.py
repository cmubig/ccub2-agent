"""
Model-agnostic cultural correction adapter.
"""

from typing import Any, Dict, List, Optional
import json
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        llm_model_name: str = "openai/gpt-oss-20b",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize correction adapter.

        Args:
            model_interface: Universal I2I interface (any model!)
            country_pack: Country-specific data pack
            detector: Cultural problem detector
            metric: Cultural accuracy metric
            llm_model_name: LLM for edit intent generation (default: GPT-OSS-20B)
            load_in_4bit: Use 4-bit quantization for LLM
            load_in_8bit: Use 8-bit quantization for LLM
            device: Device for LLM (cuda/cpu)
        """
        self.model = model_interface
        self.country_pack = country_pack
        self.detector = detector
        self.metric = metric

        # Lazy loading for LLM (only load when needed)
        self.llm_model_name = llm_model_name
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.device = device
        self.llm_model = None
        self.llm_tokenizer = None
        self.use_chat_template = False

        logger.info(
            f"CulturalCorrectionAdapter initialized with model: {model_interface.model_name}"
        )
        logger.info(f"LLM for edit intent: {llm_model_name} (lazy loading)")

    def _load_llm(self):
        """Load LLM model for edit intent generation (lazy loading)."""
        if self.llm_model is not None:
            return

        logger.info(f"Loading LLM: {self.llm_model_name}...")

        model_kwargs = {"torch_dtype": torch.float16}
        tokenizer_kwargs = {}

        # Quantization config
        if self.load_in_4bit or self.load_in_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quant_config

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name, **tokenizer_kwargs
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name, device_map="auto", **model_kwargs
        )
        self.llm_model.eval()
        self.use_chat_template = hasattr(self.llm_tokenizer, "apply_chat_template")

        logger.info(f"✓ LLM loaded: {self.llm_model_name}")

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

        # Call actual LLM
        try:
            # Ensure LLM is loaded
            self._load_llm()

            # Invoke LLM
            response = self._invoke_llm(llm_prompt)

            # Parse JSON response
            parsed = self._parse_json_response(response)

            # Validate required fields
            if parsed and all(k in parsed for k in ["enhanced_prompt", "focus_areas", "cultural_elements_to_add"]):
                return parsed
            else:
                logger.warning("LLM response missing required fields, using placeholder")
                return self._generate_edit_intent_placeholder(prompt, issues, country)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            logger.info("Falling back to placeholder edit intent")
            return self._generate_edit_intent_placeholder(prompt, issues, country)

    def _invoke_llm(self, instruction: str) -> str:
        """Invoke LLM with the given instruction."""
        try:
            if self.use_chat_template:
                messages = [
                    {"role": "system", "content": "You are a cultural image editing expert."},
                    {"role": "user", "content": instruction},
                ]
                encoded = self.llm_tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)
            else:
                encoded = self.llm_tokenizer.encode(
                    f"system You are a cultural image editing expert. user {instruction}",
                    return_tensors="pt"
                ).to(self.device)

            with torch.inference_mode():
                outputs = self.llm_model.generate(
                    encoded,
                    attention_mask=torch.ones_like(encoded),
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )

            response = self.llm_tokenizer.decode(
                outputs[0][encoded.shape[1]:], skip_special_tokens=True
            )
            return response

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from LLM."""
        # Try to extract JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON object found in LLM response")
            return {}

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_str = json_match.group()
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
            json_str = re.sub(r"'", '"', json_str)  # Replace single quotes
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}")
                return {}

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
                enhanced_prompt += "Use proper Korean Hangul characters. "
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
