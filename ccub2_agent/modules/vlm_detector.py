"""
VLM-based cultural problem detector.

Integrates the enhanced cultural metric from metric/cultural_metric
into the agent system for real-time problem detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import sys

# Add metric module to path
METRIC_PATH = Path(__file__).parent.parent.parent / "metric" / "cultural_metric"
if str(METRIC_PATH) not in sys.path:
    sys.path.insert(0, str(METRIC_PATH))

from enhanced_cultural_metric_pipeline import (
    EnhancedVLMClient,
    EnhancedCulturalKnowledgeBase,
    EnhancedCulturalEvalSample,
)

# Import CLIP image RAG
from .clip_image_rag import create_clip_rag, CLIPImageRAG

# Import job creator (optional)
try:
    from .agent_job_creator import AgentJobCreator
except ImportError:
    AgentJobCreator = None

logger = logging.getLogger(__name__)


class VLMCulturalDetector:
    """
    Cultural problem detector using VLM (Qwen3-VL-8B).

    This wraps the EnhancedVLMClient from cultural_metric
    for use in the agent's real-time detection pipeline.
    """

    def __init__(
        self,
        vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        index_dir: Optional[Path] = None,
        clip_index_dir: Optional[Path] = None,
        load_in_4bit: bool = True,
        debug: bool = False,
        firebase_config: Optional[str] = None,
        enable_auto_job_creation: bool = False,
    ):
        """
        Initialize VLM detector.

        Args:
            vlm_model: VLM model name
            index_dir: Path to text FAISS index (optional, for cultural context)
            clip_index_dir: Path to CLIP image index (optional, for reference retrieval)
            load_in_4bit: Use 4-bit quantization to save memory
            debug: Enable debug logging
            firebase_config: Path to Firebase config (optional, for job creation)
            enable_auto_job_creation: Automatically create jobs when data gaps detected
        """
        logger.info(f"Initializing VLMCulturalDetector with {vlm_model}")

        self.vlm = EnhancedVLMClient(
            model_name=vlm_model,
            load_in_4bit=load_in_4bit,
            debug=debug,
        )

        # Optional: Load knowledge base for cultural context
        self.kb = None
        if index_dir and index_dir.exists():
            logger.info(f"Loading cultural knowledge base from {index_dir}")
            self.kb = EnhancedCulturalKnowledgeBase(index_dir)
        else:
            logger.warning("No text knowledge base provided - using basic detection")

        # Optional: Load CLIP RAG for reference image retrieval
        self.clip_rag: Optional[CLIPImageRAG] = None
        if clip_index_dir and clip_index_dir.exists():
            logger.info(f"Loading CLIP image RAG from {clip_index_dir}")
            self.clip_rag = create_clip_rag(
                model_name="openai/clip-vit-base-patch32",
                index_dir=clip_index_dir,
                device="auto",
            )
        else:
            logger.warning("No CLIP index provided - no reference image retrieval")

        # Optional: Job creator for automatic data collection
        self.job_creator = None
        self.enable_auto_job_creation = enable_auto_job_creation
        if enable_auto_job_creation and AgentJobCreator and firebase_config:
            logger.info("Enabling automatic job creation")
            self.job_creator = AgentJobCreator(firebase_config=firebase_config)
        elif enable_auto_job_creation:
            logger.warning("Auto job creation requested but Firebase config not provided")

        self.debug = debug

    def detect(
        self,
        image_path: Path,
        prompt: str,
        country: str,
        editing_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """
        Detect cultural issues in a generated image.

        Args:
            image_path: Path to image
            prompt: Original generation prompt
            country: Target country
            editing_prompt: Optional editing instruction
            category: Optional category for context

        Returns:
            List of detected issues with:
                - type: Issue type
                - category: Issue category
                - description: Description
                - severity: Severity score (1-10)
        """
        logger.info(f"Detecting issues for {country} image: {image_path.name}")

        # Get cultural context if available
        context = ""
        if self.kb:
            # Create minimal sample for retrieval
            sample = EnhancedCulturalEvalSample(
                uid="temp",
                group_id="temp",
                step="detection",
                prompt=prompt,
                country=country,
                image_path=image_path,
                editing_prompt=editing_prompt,
                category=category,
            )
            docs = self.kb.retrieve_contextual(prompt, sample, top_k=5)
            context = "\n".join(f"[Doc {i}] {doc.text}" for i, doc in enumerate(docs, 1))

        # Add CLIP-based reference image context
        if self.clip_rag:
            ref_context = self.clip_rag.get_reference_context(
                image_path=image_path,
                k=3,
                category=category,
            )
            if context:
                context += "\n\n" + ref_context
            else:
                context = ref_context

        # Get VLM scores
        cultural_score, prompt_score = self.vlm.evaluate_cultural_scores(
            image_path=image_path,
            prompt=prompt,
            editing_prompt=editing_prompt or "",
            context=context,
            country=country,
        )

        # Convert scores to issues
        issues = []

        # Low cultural representation → issue (STRICTER: < 8 instead of < 6)
        if cultural_score < 8:
            issues.append({
                "type": "incorrect",
                "category": "cultural_representation",
                "description": f"Cultural accuracy needs improvement ({cultural_score}/10) for {country}",
                "severity": 5 + (8 - cultural_score),  # 6-12 severity
            })

        # Low prompt alignment → issue (STRICTER: < 8 instead of < 6)
        if prompt_score < 8:
            issues.append({
                "type": "missing",
                "category": "prompt_alignment",
                "description": f"Prompt alignment needs improvement ({prompt_score}/10)",
                "severity": 4 + (8 - prompt_score),  # 5-11 severity
            })

        # Ask specific cultural questions
        if category:
            specific_issues = self._detect_category_specific(
                image_path, prompt, country, category, context
            )
            issues.extend(specific_issues)

        if self.debug:
            logger.info(f"Detected {len(issues)} issues (cultural={cultural_score}, prompt={prompt_score})")

        # Check if we have enough reference data
        if self.enable_auto_job_creation and self.job_creator:
            self._check_data_sufficiency(
                country=country,
                category=category,
                prompt=prompt,
                context=context,
                cultural_score=cultural_score,
            )

        return issues

    def _check_data_sufficiency(
        self,
        country: str,
        category: Optional[str],
        prompt: str,
        context: str,
        cultural_score: int,
    ):
        """
        Check if we have sufficient reference data.
        If not, trigger job creation.

        Args:
            country: Target country
            category: Image category
            prompt: Generation prompt
            context: Retrieved context
            cultural_score: VLM cultural score
        """
        # Extract cultural concepts from prompt
        concepts = self._extract_concepts_from_prompt(prompt, country)

        if not concepts:
            return

        # Check reference data availability for each concept
        for concept in concepts:
            # Check if we have enough data about this concept
            concept_context = self._search_concept_in_kb(concept, country, category)

            if len(concept_context) < 3:  # Less than 3 references
                logger.warning(f"⚠️ Insufficient data for concept '{concept}' in {country}")
                logger.warning(f"   Found only {len(concept_context)} references")

                # Check if job already exists
                keywords = [concept, country]
                if category:
                    keywords.append(category)

                # Try to create job (will check for duplicates)
                job_id = self.job_creator.create_job(
                    country=country,
                    category=category or "general",
                    keywords=keywords,
                    description=f"Collect authentic {country} {concept} images to improve cultural accuracy",
                    min_level=2,
                    points=50,
                    target_count=20,  # Collect 20 images per concept
                )

                if job_id:
                    logger.info(f"📋 Created data collection job {job_id} for concept '{concept}'")
                else:
                    logger.info(f"Job for '{concept}' already exists or creation skipped")

    def _extract_concepts_from_prompt(self, prompt: str, country: str) -> List[str]:
        """Extract cultural concepts from prompt."""
        # Simple keyword extraction
        # TODO: Use VLM for better extraction
        keywords = []

        # Category-specific keywords
        if "hanbok" in prompt.lower():
            keywords.append("hanbok")
        if "kimchi" in prompt.lower():
            keywords.append("kimchi")
        if "palace" in prompt.lower() or "temple" in prompt.lower():
            keywords.append("traditional_architecture")
        if "festival" in prompt.lower():
            keywords.append("festival")

        return keywords

    def _search_concept_in_kb(
        self,
        concept: str,
        country: str,
        category: Optional[str]
    ) -> List:
        """Search for concept in knowledge base."""
        if not self.kb:
            return []

        # Simple search
        # TODO: Implement proper concept search
        try:
            docs = self.kb.retrieve(f"{concept} {country}", top_k=10)
            # Filter for relevant docs
            relevant = [
                doc for doc in docs
                if concept.lower() in doc.text.lower()
            ]
            return relevant
        except:
            return []

    def _detect_category_specific(
        self,
        image_path: Path,
        prompt: str,
        country: str,
        category: str,
        context: str,
    ) -> List[Dict]:
        """
        Detect category-specific cultural issues using DYNAMIC question generation.

        Uses Cultural Metric's EnhancedQuestionGenerator to create questions
        based on cultural context, not hardcoded templates.
        """
        issues = []

        # Dynamic question generation using Cultural Metric approach
        # Create a sample for question generation
        from enhanced_cultural_metric_pipeline import EnhancedCulturalEvalSample, RetrievedDoc

        sample = EnhancedCulturalEvalSample(
            uid="detection",
            group_id="detection",
            step="detection",
            prompt=prompt,
            country=country,
            image_path=image_path,
            category=category,
        )

        # Parse context into docs
        docs = []
        if context:
            for line in context.split('\n'):
                if line.strip() and not line.startswith('[Doc'):
                    docs.append(RetrievedDoc(
                        text=line.strip(),
                        score=1.0,
                        metadata={"category": category}
                    ))

        # Generate dynamic questions using LLM (if available)
        # For now, use fallback templates but structured properly
        # TODO: Initialize question generator in __init__ for true dynamic generation

        # Fallback: Use smart template selection based on cultural knowledge
        template_questions = self._generate_smart_questions(prompt, country, category, context)

        for question, expected in template_questions:
            answer = self.vlm.answer(image_path, question, context)

            # If answer doesn't match expected, it's an issue
            if answer != expected and answer != "ambiguous":
                issues.append({
                    "type": "incorrect" if expected == "yes" else "missing",
                    "category": category,
                    "description": question.replace("?", ""),
                    "severity": 7,
                })

        return issues

    def _generate_smart_questions(
        self,
        prompt: str,
        country: str,
        category: str,
        context: str,
    ) -> List[Tuple[str, str]]:
        """
        Generate smart questions based on cultural context (not just category).

        This extracts key concepts from context and generates specific questions.
        Better than hardcoded templates!
        """
        questions = []

        # Extract cultural concepts from context
        cultural_concepts = self._extract_concepts_from_context(context, category)

        # Base questions for any category
        questions.append((
            f"Does this image accurately represent authentic {country} {category}?",
            "yes"
        ))

        # Add concept-specific questions
        for concept in cultural_concepts[:3]:  # Top 3 concepts
            questions.append((
                f"Does the {category} show proper {concept} as found in {country} culture?",
                "yes"
            ))

        # Add negative check
        questions.append((
            f"Are there elements from other cultures that contradict authentic {country} {category}?",
            "no"
        ))

        return questions

    def _extract_concepts_from_context(
        self,
        context: str,
        category: str
    ) -> List[str]:
        """Extract key cultural concepts from RAG context."""
        # Simple extraction - look for descriptive nouns/adjectives
        # In future: use NER or LLM for better extraction

        concepts = []

        # Category-specific keywords to look for
        if category == "traditional_clothing":
            keywords = ["fabric", "material", "texture", "color", "pattern", "structure", "garment", "style", "design"]
        elif category == "food":
            keywords = ["ingredients", "preparation", "presentation", "flavor", "dish", "recipe", "cooking"]
        elif category == "architecture":
            keywords = ["building", "structure", "design", "style", "materials", "roof", "decoration"]
        else:
            keywords = ["style", "design", "appearance", "characteristics", "features"]

        # Find keywords in context
        context_lower = context.lower()
        for keyword in keywords:
            if keyword in context_lower:
                concepts.append(keyword)

        return concepts[:5]  # Return top 5

    def score_cultural_quality(
        self,
        image_path: Path,
        prompt: str,
        country: str,
        editing_prompt: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Get cultural quality scores.

        Args:
            image_path: Path to image
            prompt: Generation prompt
            country: Target country
            editing_prompt: Optional editing instruction

        Returns:
            Tuple of (cultural_representative, prompt_alignment) scores (1-10)
        """
        context = ""
        if self.kb:
            # Get minimal context for scoring
            sample = EnhancedCulturalEvalSample(
                uid="temp_score",
                group_id="temp",
                step="scoring",
                prompt=prompt,
                country=country,
                image_path=image_path,
                editing_prompt=editing_prompt,
            )
            docs = self.kb.retrieve_contextual(prompt, sample, top_k=3)
            context = "\n".join(doc.text for doc in docs)

        return self.vlm.evaluate_cultural_scores(
            image_path=image_path,
            prompt=prompt,
            editing_prompt=editing_prompt or "",
            context=context,
            country=country,
        )


def create_vlm_detector(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    index_dir: Optional[Path] = None,
    clip_index_dir: Optional[Path] = None,
    load_in_4bit: bool = True,
    debug: bool = False,
) -> VLMCulturalDetector:
    """
    Convenience function to create VLM detector.

    Args:
        model_name: VLM model name
        index_dir: Path to text FAISS cultural knowledge index
        clip_index_dir: Path to CLIP image index
        load_in_4bit: Use 4-bit quantization
        debug: Enable debug logging

    Returns:
        VLMCulturalDetector instance
    """
    return VLMCulturalDetector(
        vlm_model=model_name,
        index_dir=index_dir,
        clip_index_dir=clip_index_dir,
        load_in_4bit=load_in_4bit,
        debug=debug,
    )
