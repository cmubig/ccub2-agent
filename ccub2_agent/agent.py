"""
Main agent orchestrator for cultural bias mitigation.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .modules.adapter import CulturalCorrectionAdapter
from .modules.detector import CulturalDetector
from .modules.ccub_metric import CCUBMetric
from .modules.agent_job_creator import AgentJobCreator
from .modules.gap_analyzer import DataGapAnalyzer
from .modules.country_pack import CountryDataPack
from .models.universal_interface import UniversalI2IInterface

logger = logging.getLogger(__name__)


class CulturalAgent:
    """
    Main agent that orchestrates cultural bias detection and correction.

    This is a model-agnostic system that works with any I2I model through
    the UniversalI2IInterface.
    """

    def __init__(
        self,
        country: str,
        model: str = "flux-kontext",
        firebase_config: Optional[str] = None,
        auto_job_creation: bool = True,
        data_pack_path: Optional[Path] = None,
    ):
        """
        Initialize cultural agent.

        Args:
            country: Target country (e.g., "korea", "japan")
            model: I2I model name (any supported model)
            firebase_config: Path to Firebase config file
            auto_job_creation: Auto-create jobs when data is missing
            data_pack_path: Path to country data packs
        """
        self.country = country
        self.model_name = model
        self.auto_job_creation = auto_job_creation

        # Initialize components
        logger.info(f"Initializing CulturalAgent for {country} with {model}")

        self.model_interface = UniversalI2IInterface(model)
        self.country_pack = CountryDataPack(country, data_pack_path)
        self.detector = CulturalDetector()
        self.metric = CCUBMetric()
        self.gap_analyzer = DataGapAnalyzer(self.country_pack)

        # Initialize adapter (model-agnostic!)
        self.adapter = CulturalCorrectionAdapter(
            model_interface=self.model_interface,
            country_pack=self.country_pack,
            detector=self.detector,
            metric=self.metric,
        )

        # Initialize job creator if needed
        if auto_job_creation and firebase_config:
            self.job_creator = AgentJobCreator(firebase_config)
        else:
            self.job_creator = None

        logger.info("CulturalAgent initialized successfully")

    def generate(
        self,
        prompt: str,
        initial_image: Optional[Any] = None,
        max_iterations: int = 5,
        threshold: float = 80.0,
    ) -> Dict:
        """
        Generate culturally accurate image with automatic correction.

        Args:
            prompt: Text prompt
            initial_image: Optional initial image (if None, use T2I first)
            max_iterations: Maximum correction iterations
            threshold: CCUB metric threshold

        Returns:
            Dict containing:
                - image: Final corrected image
                - issues: Detected cultural issues
                - corrections: Correction history
                - jobs_created: Auto-created jobs (if any)
                - score: Final CCUB metric score
        """
        logger.info(f"Generating image for prompt: '{prompt}'")

        # If no initial image, generate one
        if initial_image is None:
            logger.info("No initial image provided, generating with T2I...")
            initial_image = self.model_interface.text_to_image(prompt)

        # Apply correction adapter
        result = self.adapter.correct(
            image=initial_image,
            prompt=prompt,
            country=self.country,
            max_iterations=max_iterations,
            threshold=threshold,
        )

        # Handle missing data
        if result["status"] == "data_missing":
            logger.warning(f"Missing data detected: {result['missing_data']}")

            # Analyze gaps
            gaps = self.gap_analyzer.analyze(
                issues=result["missing_data"], country=self.country
            )

            # Auto-create jobs if enabled
            jobs_created = []
            if self.auto_job_creation and self.job_creator:
                logger.info("Auto-creating data collection jobs...")
                for gap in gaps:
                    job_id = self.job_creator.create_job(
                        country=self.country,
                        category=gap["category"],
                        keywords=gap["keywords"],
                        description=gap["description"],
                    )
                    jobs_created.append(job_id)
                    logger.info(f"Created job: {job_id}")

            result["jobs_created"] = jobs_created
            result["data_gaps"] = gaps

        # Calculate final score
        final_score = self.metric.score(
            country=self.country, caption=prompt, image=result["image"]
        )

        return {
            "image": result["image"],
            "issues": result.get("history", [{}])[-1].get("issues", [])
            if result.get("history")
            else [],
            "corrections": result.get("history", []),
            "jobs_created": result.get("jobs_created", []),
            "score": final_score,
            "status": result["status"],
        }

    def evaluate(
        self, image: Any, prompt: str, return_details: bool = True
    ) -> Dict:
        """
        Evaluate an image for cultural accuracy.

        Args:
            image: Image to evaluate
            prompt: Original prompt
            return_details: Return detailed breakdown

        Returns:
            Evaluation results with score and details
        """
        # Detect issues
        issues = self.detector.detect(image, prompt, self.country)

        # Calculate score
        score = self.metric.score(self.country, prompt, image)

        result = {"score": score, "issues": issues, "country": self.country}

        if return_details:
            result["details"] = self.metric.get_detailed_scores(
                self.country, prompt, image
            )

        return result

    def update_country_pack(self, fetch_from_firebase: bool = True):
        """
        Update country data pack with new approved data.

        Args:
            fetch_from_firebase: Fetch latest approved data from Firebase
        """
        logger.info(f"Updating country pack for {self.country}")
        self.country_pack.update(fetch_from_firebase=fetch_from_firebase)
        logger.info("Country pack updated successfully")
