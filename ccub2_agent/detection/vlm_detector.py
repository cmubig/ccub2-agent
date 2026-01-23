"""
VLM-based cultural problem detector.

Integrates the enhanced cultural metric from evaluation/metrics/cultural_metric
into the agent system for real-time problem detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import sys

# Add metric module to path
METRIC_PATH = Path(__file__).parent.parent / "evaluation" / "metrics" / "cultural_metric"
if str(METRIC_PATH) not in sys.path:
    sys.path.insert(0, str(METRIC_PATH))

from enhanced_cultural_metric_pipeline import (
    EnhancedVLMClient,
    EnhancedCulturalKnowledgeBase,
    EnhancedCulturalEvalSample,
)

# Import CLIP image RAG
from ..retrieval.clip_image_rag import create_clip_rag, CLIPImageRAG

# Import job creator (optional)
try:
    from ..data.job_creator import AgentJobCreator
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

        # Load structured cultural knowledge (korea_knowledge.json)
        self.structured_knowledge = {}
        if index_dir:
            knowledge_file = index_dir.parent.parent / "cultural_knowledge" / f"{index_dir.name}_knowledge.json"
            if knowledge_file.exists():
                import json
                with open(knowledge_file) as f:
                    data = json.load(f)
                    self.structured_knowledge = {
                        k['item_id']: k for k in data.get('knowledge', [])
                    }
                logger.info(f"Loaded {len(self.structured_knowledge)} structured knowledge entries")
            else:
                logger.warning(f"Structured knowledge not found at {knowledge_file}")

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
        iteration_number: int = 0,
        previous_cultural_score: Optional[int] = None,
        previous_prompt_score: Optional[int] = None,
    ) -> List[Dict]:
        """
        Detect cultural issues in a generated image.

        Args:
            image_path: Path to image
            prompt: Original generation prompt
            country: Target country
            editing_prompt: Optional editing instruction
            category: Optional category for context
            iteration_number: Current iteration number for context
            previous_cultural_score: Previous cultural score for comparison
            previous_prompt_score: Previous prompt score for comparison

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

        # Get VLM scores with iteration context
        cultural_score, prompt_score = self.vlm.evaluate_cultural_scores(
            image_path=image_path,
            prompt=prompt,
            editing_prompt=editing_prompt or "",
            context=context,
            country=country,
            iteration_number=iteration_number,
            previous_cultural_score=previous_cultural_score,
            previous_prompt_score=previous_prompt_score,
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
        # Dynamic keyword extraction from prompt
        # Extract cultural terms from the prompt itself
        keywords = []

        # Split prompt into words and look for potential cultural terms
        words = prompt.lower().split()

        # General category mapping (country-agnostic)
        category_keywords = {
            'architecture': ['palace', 'temple', 'building', 'shrine', 'pagoda', 'mosque', 'cathedral'],
            'food': ['food', 'dish', 'cuisine', 'meal', 'cooking'],
            'clothing': ['clothing', 'dress', 'garment', 'attire', 'costume', 'robe'],
            'festival': ['festival', 'celebration', 'ceremony', 'ritual'],
        }

        # Map prompt words to categories
        for word in words:
            for category, terms in category_keywords.items():
                if word in terms:
                    keywords.append(category)
                    break

        # Remove duplicates
        keywords = list(set(keywords))

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

        # Concept search
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
        Detect category-specific cultural issues using DETAILED VLM analysis.

        Uses our cultural knowledge base to ask VLM SPECIFIC questions.
        """
        issues = []

        # IMPROVED: Ask VLM to analyze with SPECIFIC cultural knowledge
        if context and len(context.strip()) > 50:
            # We have cultural knowledge - use it!
            detailed_analysis_prompt = self._build_detailed_analysis_prompt(
                prompt, country, category, context
            )

            # Get detailed description of problems from VLM
            try:
                from PIL import Image
                img = Image.open(image_path)

                # Use VLM's query method with context
                # FIXED: Increased from 300 to 800 tokens for complete analysis
                detailed_problems = self.vlm.query_with_context(
                    image=img,
                    query=detailed_analysis_prompt,
                    context=context,
                    max_tokens=800
                )

                # Parse the VLM's detailed response
                if detailed_problems and len(detailed_problems.strip()) > 10:
                    # FIXED: Remove duplicate/repetitive lines from VLM hallucination
                    cleaned_problems = self._deduplicate_vlm_analysis(detailed_problems.strip())

                    # Add as a comprehensive issue with details
                    issues.append({
                        "type": "incorrect",
                        "category": category,
                        "description": cleaned_problems,
                        "severity": 9,
                        "is_detailed": True,  # Flag for prompt adapter to use directly
                    })

                    if self.debug:
                        logger.info(f"VLM detailed analysis: {cleaned_problems[:200]}...")
            except Exception as e:
                logger.warning(f"Detailed VLM analysis failed: {e}, falling back to yes/no questions")

        # Only use yes/no questions as fallback if detailed analysis didn't find issues
        if not issues or not any(i.get("is_detailed") for i in issues):
            template_questions = self._generate_smart_questions(prompt, country, category, context)

            for question, expected in template_questions:
                answer = self.vlm.answer(image_path, question, context)

                # If answer doesn't match expected, it's an issue
                if answer != expected and answer != "ambiguous":
                    # Ask follow-up for specific details instead of using generic question
                    try:
                        from PIL import Image
                        img = Image.open(image_path)

                        followup_prompt = f"Based on the {country} {category} in this image, what specifically is wrong or missing? Describe the exact cultural problems you see in 2-3 sentences."
                        specific_desc = self.vlm.query_with_context(
                            image=img,
                            query=followup_prompt,
                            context=context,
                            max_tokens=300
                        )

                        # Use specific description if we got meaningful response
                        description = specific_desc.strip() if specific_desc and len(specific_desc.strip()) > 30 else question.replace("?", "")

                        if self.debug:
                            logger.info(f"Follow-up question result: {description[:150]}...")

                    except Exception as e:
                        logger.warning(f"Follow-up question failed: {e}, using generic description")
                        description = question.replace("?", "")

                    issues.append({
                        "type": "incorrect" if expected == "yes" else "missing",
                        "category": category,
                        "description": description,
                        "severity": 7,
                    })

        return issues

    def _build_detailed_analysis_prompt(
        self,
        prompt: str,
        country: str,
        category: str,
        context: str,
    ) -> str:
        """
        Build a detailed analysis prompt that extracts SPECIFIC cultural issues
        AND actionable solutions in Problem-Action pairs.

        Returns a prompt that asks VLM to describe what's wrong AND what to do.
        """
        # Extract cultural elements from context
        cultural_elements = self._extract_specific_elements_from_context(context, category, country)

        analysis_prompt = f"""Analyze this image of {country} {category} and provide SPECIFIC Problem-Action pairs.

Expected elements for authentic {country} {category}:
{cultural_elements}

Task: For each cultural issue, provide BOTH the problem AND the concrete action to fix it.

FORMAT (use exactly this format):
PROBLEM: [what is wrong]
ACTION: [specific editing instruction to fix it]

EXAMPLES:
PROBLEM: The collar is a Western-style folded collar instead of traditional stand-up collar
ACTION: Replace the collar with a traditional "dongjeong" white stand-up collar

PROBLEM: The dish uses Western plating style with sauce drizzle
ACTION: Add traditional Korean side dishes "banchan", use brass chopsticks, serve in ceramic bowls

PROBLEM: The roof has Western-style shingles instead of traditional tiles
ACTION: Replace with curved Korean "giwa" roof tiles in dark gray/black color

RULES:
- Each ACTION must be a specific editing instruction (start with: Replace, Add, Remove, Change, Transform)
- Do NOT use vague terms like "improve", "enhance", "adjust aesthetics"
- Use SPECIFIC cultural item names in quotes (e.g., "hanbok", "kimchi", "pagoda")
- Maximum 3 Problem-Action pairs, focus on the most important issues

Now analyze this image:"""

        return analysis_prompt

    def _extract_specific_elements_from_context(
        self,
        context: str,
        category: str,
        country: str,
    ) -> str:
        """
        Extract specific cultural elements using STRUCTURED KNOWLEDGE.

        Priority:
        1. Structured knowledge (korea_knowledge.json) - detailed, categorized
        2. RAG context - text snippets
        3. Fallback - generic description
        """
        elements_text = []

        # PRIORITY 1: Use structured knowledge if available
        if self.structured_knowledge and category:
            category_knowledge = [
                k for k in self.structured_knowledge.values()
                if k.get('category') == category and k.get('country') == country
            ]

            if category_knowledge:
                # Build comprehensive description from structured data
                for entry in category_knowledge[:3]:  # Top 3 most relevant
                    if 'visual_features' in entry:
                        elements_text.append(f"Visual Features: {entry['visual_features'][:200]}")
                    if 'colors_patterns' in entry:
                        elements_text.append(f"Colors/Patterns: {entry['colors_patterns'][:150]}")
                    if 'materials_textures' in entry:
                        elements_text.append(f"Materials: {entry['materials_textures'][:150]}")

                if elements_text:
                    return '\n'.join(elements_text)

        # PRIORITY 2: Use RAG context
        if context and len(context.strip()) > 50:
            elements = []
            lines = context.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('[Doc'):
                    continue
                if len(line) > 20:
                    elements.append(f"- {line}")

            if elements:
                return '\n'.join(elements[:10])

        # PRIORITY 3: Fallback
        return f"Traditional {category} elements specific to {country} culture"

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
        iteration_number: int = 0,
        previous_cultural_score: Optional[int] = None,
        previous_prompt_score: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Get cultural quality scores.

        Args:
            image_path: Path to image
            prompt: Generation prompt
            country: Target country
            editing_prompt: Optional editing instruction
            iteration_number: Current iteration number for context
            previous_cultural_score: Previous cultural score for comparison
            previous_prompt_score: Previous prompt score for comparison

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
            iteration_number=iteration_number,
            previous_cultural_score=previous_cultural_score,
            previous_prompt_score=previous_prompt_score,
        )

    def _deduplicate_vlm_analysis(self, text: str) -> str:
        """Remove duplicate/repetitive lines from VLM hallucination."""
        lines = text.split('\n')
        seen = set()
        unique_lines = []

        for line in lines:
            # Normalize line for comparison (remove numbers, extra spaces)
            normalized = line.strip()
            # Remove leading numbers like "1. ", "2. ", etc.
            import re
            normalized = re.sub(r'^\d+\.\s*', '', normalized)

            # Skip if we've seen this exact content before
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)

                # Limit to first 10 unique issues
                if len(unique_lines) >= 10:
                    break

        return '\n'.join(unique_lines)


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
