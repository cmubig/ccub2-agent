"""Enhanced cultural metric pipeline with improved question generation and resumption support."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import pickle
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Any

import faiss
import torch
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .taxonomy import (
    CulturalDimension,
    CulturalProfile,
    DimensionScore,
    FailurePenalty,
    DIMENSION_QUESTION_TEMPLATES,
    format_dimension_questions,
    get_dimension_weights,
)
from .cultscore import CultScoreComputer, CultScoreConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enhanced data containers
# ---------------------------------------------------------------------------

@dataclass
class EnhancedCulturalEvalSample:
    uid: str
    group_id: str
    step: str
    prompt: str
    country: str
    image_path: Path
    editing_prompt: Optional[str] = None
    # New metadata fields
    model: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    variant: Optional[str] = None


@dataclass
class RetrievedDoc:
    text: str
    score: float
    metadata: Dict[str, str]
    raw_similarity: float = 0.0  # raw FAISS similarity before authority weighting
    source_authority: float = 1.0  # authority weight from source type


@dataclass
class CulturalQuestion:
    question: str
    expected_answer: str
    rationale: str


@dataclass
class EvaluationResult:
    sample: EnhancedCulturalEvalSample
    questions: List[CulturalQuestion]
    answers: List[str]
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_questions: int
    processing_time: float
    question_source: str  # "model", "heuristic", "fallback"
    cultural_representation_score: Optional[int] = None
    prompt_alignment_score: Optional[int] = None
    # CultScore fields (Phase 2)
    cultural_profile: Optional[CulturalProfile] = None
    cultscore: Optional[float] = None
    cultscore_confidence: Optional[float] = None
    cultscore_penalised: Optional[float] = None
    dimension_scores: Optional[Dict[str, float]] = None


@dataclass
class CheckpointData:
    completed_samples: List[str]  # UIDs of completed samples
    results: List[EvaluationResult]
    timestamp: str
    total_samples: int
    current_index: int


# ---------------------------------------------------------------------------
# Enhanced knowledge base with cultural context
# ---------------------------------------------------------------------------

class EnhancedCulturalKnowledgeBase:
    # Source authority weights (C3)
    SOURCE_AUTHORITY: Dict[str, float] = {
        "unesco_ich": 1.0,
        "wikipedia": 0.7,
        "wikivoyage": 0.5,
    }
    DEFAULT_AUTHORITY: float = 0.5

    def __init__(self, index_dir: Path):
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        meta_path = index_dir / "metadata.jsonl"
        self.metadata = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()]
        config = json.loads((index_dir / "index_config.json").read_text(encoding="utf-8"))
        # Support both old and new key names for backward compatibility
        model_name = config.get("model_name") or config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)

        # Section priorities for different categories
        self.section_priorities = {
            "architecture": ["Culture", "History", "Geography", "Art", "Tourism"],
            "art": ["Culture", "Art", "History", "Religion", "Traditional"],
            "event": ["Culture", "Religion", "History", "Society", "Traditional"],
            "fashion": ["Culture", "Traditional", "Society", "Art", "Modern"],
            "food": ["Culture", "Traditional", "Geography", "Agriculture", "Cuisine"],
            "landscape": ["Geography", "Tourism", "Nature", "Climate", "Environment"],
            "people": ["Demographics", "Society", "Culture", "Economy", "Politics"],
            "wildlife": ["Geography", "Environment", "Nature", "Conservation", "Biodiversity"]
        }

    def retrieve_contextual(
        self,
        query: str,
        sample: EnhancedCulturalEvalSample,
        top_k: int = 8,
    ) -> List[RetrievedDoc]:
        """Enhanced retrieval using sample metadata for better context."""
        # Build enhanced query with metadata
        enhanced_query_parts = [query]
        
        if sample.category:
            enhanced_query_parts.append(sample.category)
        if sample.sub_category:
            enhanced_query_parts.append(sample.sub_category)
        if sample.variant and sample.variant != "general":
            enhanced_query_parts.append(sample.variant)
            
        enhanced_query = " ".join(enhanced_query_parts)
        
        # Get section bias based on category
        section_bias = self.section_priorities.get(sample.category, ["Culture"])
        
        return self.retrieve(enhanced_query, sample.country, top_k, section_bias)

    def retrieve(
        self,
        query: str,
        country: Optional[str],
        top_k: int = 8,
        section_bias: Optional[Sequence[str]] = None,
    ) -> List[RetrievedDoc]:
        vector = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(vector, top_k * 6)

        docs: List[RetrievedDoc] = []
        preferred = set(section_bias or [])

        # First pass: country-specific docs
        for score, idx in zip(scores[0], indices[0]):
            meta = self.metadata[idx]
            if country and meta.get("country") != country:
                continue
            raw_sim = float(score)
            # Prefer pre-computed source_authority from metadata (Phase 3),
            # fall back to runtime lookup by source_type.
            if "source_authority" in meta:
                authority = float(meta["source_authority"])
            else:
                source_type = meta.get("source_type", "wikipedia")
                authority = self.SOURCE_AUTHORITY.get(source_type, self.DEFAULT_AUTHORITY)
            weighted_score = raw_sim * authority
            docs.append(RetrievedDoc(
                text=meta["text"], score=weighted_score, metadata=meta,
                raw_similarity=raw_sim, source_authority=authority,
            ))
            if len(docs) >= top_k:
                break

        # Second pass: other countries if needed
        if len(docs) < top_k:
            for score, idx in zip(scores[0], indices[0]):
                meta = self.metadata[idx]
                if country and meta.get("country") == country:
                    continue
                raw_sim = float(score)
                if "source_authority" in meta:
                    authority = float(meta["source_authority"])
                else:
                    source_type = meta.get("source_type", "wikipedia")
                    authority = self.SOURCE_AUTHORITY.get(source_type, self.DEFAULT_AUTHORITY)
                weighted_score = raw_sim * authority
                docs.append(RetrievedDoc(
                    text=meta["text"], score=weighted_score, metadata=meta,
                    raw_similarity=raw_sim, source_authority=authority,
                ))
                if len(docs) >= top_k:
                    break

        # Sort by section preference, then by score
        if preferred:
            docs.sort(
                key=lambda d: (
                    0 if d.metadata.get("section") in preferred else 1,
                    -d.score,
                )
            )
        else:
            docs.sort(key=lambda d: -d.score)
        return docs[:top_k]


# ---------------------------------------------------------------------------
# Enhanced question generator with better templates
# ---------------------------------------------------------------------------

class EnhancedQuestionGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        min_questions: int = 4,
        min_negative: int = 0,
        allow_fallback: bool = True,
        debug: bool = False,
    ) -> None:
        tokenizer_kwargs = {"trust_remote_code": True}
        model_kwargs = {"trust_remote_code": True}
        
        if load_in_8bit or load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quant_config
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **model_kwargs)
        self.model.eval()
        self.device = device
        self.use_chat_template = hasattr(self.tokenizer, "apply_chat_template")
        self.min_questions = min_questions
        self.min_negative = min_negative
        self.allow_fallback = allow_fallback
        self.debug = debug
        
        # Enhanced keyword templates organized by category and variant
        self.enhanced_templates = self._build_enhanced_templates()

    def _build_enhanced_templates(self) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        """Build context-aware question templates."""
        return {
            "architecture": {
                "traditional": [
                    ("Does the architecture show traditional {country} building styles and materials?", "yes"),
                    ("Are there modern Western architectural elements that contradict traditional {country} design?", "no"),
                    ("Does the structure incorporate traditional {country} cultural motifs or decorative elements?", "yes"),
                    ("Are there elements that clash with historical {country} architectural principles?", "no"),
                ],
                "modern": [
                    ("Does the architecture show contemporary design while maintaining {country} cultural identity?", "yes"),
                    ("Are there outdated or purely traditional elements inappropriate for modern {country} architecture?", "no"),
                    ("Does the building reflect modern {country} urban development and lifestyle?", "yes"),
                    ("Does the design ignore modern {country} architectural innovations and trends?", "no"),
                ],
                "general": [
                    ("Does the architecture represent recognizable {country} building characteristics?", "yes"),
                    ("Are there foreign architectural elements unrelated to {country}?", "no"),
                    ("Does the structure fit within the typical {country} built environment?", "yes"),
                    ("Are there elements that would be culturally inappropriate in {country}?", "no"),
                ]
            },
            "art": {
                "traditional": [
                    ("Does the artwork show traditional {country} artistic techniques and styles?", "yes"),
                    ("Are there modern or foreign artistic elements that contradict traditional {country} art?", "no"),
                    ("Does the piece incorporate traditional {country} cultural symbols or themes?", "yes"),
                    ("Are there elements that go against traditional {country} artistic principles?", "no"),
                ],
                "modern": [
                    ("Does the artwork represent contemporary {country} artistic expression?", "yes"),
                    ("Are there purely traditional elements inappropriate for modern {country} art?", "no"),
                    ("Does the piece reflect current {country} cultural and social themes?", "yes"),
                    ("Does the artwork ignore modern {country} artistic movements and innovations?", "no"),
                ],
                "general": [
                    ("Does the artwork represent {country} cultural aesthetic values?", "yes"),
                    ("Are there foreign artistic influences unrelated to {country} culture?", "no"),
                    ("Would this artwork be recognizable as reflecting {country} artistic heritage?", "yes"),
                    ("Are there elements that would be culturally inappropriate in {country} art?", "no"),
                ]
            },
            "event": {
                "traditional": [
                    ("Does the event show traditional {country} customs and practices?", "yes"),
                    ("Are there modern or foreign elements that contradict traditional {country} ceremonies?", "no"),
                    ("Do participants wear traditional {country} attire appropriate for the event?", "yes"),
                    ("Are there elements that violate traditional {country} cultural protocols?", "no"),
                ],
                "modern": [
                    ("Does the event represent contemporary {country} social gatherings and practices?", "yes"),
                    ("Are there outdated traditional elements inappropriate for modern {country} events?", "no"),
                    ("Does the event reflect current {country} lifestyle and values?", "yes"),
                    ("Does the event ignore modern {country} social norms and practices?", "no"),
                ],
                "general": [
                    ("Does the event represent typical {country} social and cultural activities?", "yes"),
                    ("Are there foreign cultural elements unrelated to {country} traditions?", "no"),
                    ("Would this event be recognizable as part of {country} cultural life?", "yes"),
                    ("Are there elements that would be culturally inappropriate for {country} events?", "no"),
                ]
            },
            "fashion": {
                "traditional": [
                    ("Does the clothing show traditional {country} garments and textiles?", "yes"),
                    ("Are there modern Western fashion elements that contradict traditional {country} dress?", "no"),
                    ("Do the garments use traditional {country} colors, patterns, and styling?", "yes"),
                    ("Are there elements that violate traditional {country} dress codes or customs?", "no"),
                ],
                "modern": [
                    ("Does the fashion represent contemporary {country} style and trends?", "yes"),
                    ("Are there purely traditional elements inappropriate for modern {country} fashion?", "no"),
                    ("Does the clothing reflect current {country} fashion preferences and lifestyle?", "yes"),
                    ("Does the style ignore modern {country} fashion developments and influences?", "no"),
                ],
                "general": [
                    ("Does the clothing represent typical {country} fashion sensibilities?", "yes"),
                    ("Are there foreign fashion elements unrelated to {country} culture?", "no"),
                    ("Would this style be recognizable as reflecting {country} fashion heritage?", "yes"),
                    ("Are there elements that would be culturally inappropriate in {country} fashion?", "no"),
                ]
            },
            "food": {
                "traditional": [
                    ("Does the food show traditional {country} ingredients and cooking methods?", "yes"),
                    ("Are there foreign cuisines or ingredients unrelated to traditional {country} food?", "no"),
                    ("Does the dish incorporate traditional {country} flavors and presentation?", "yes"),
                    ("Are there elements that contradict traditional {country} culinary practices?", "no"),
                ],
                "modern": [
                    ("Does the food represent contemporary {country} cuisine and dining trends?", "yes"),
                    ("Are there outdated traditional elements inappropriate for modern {country} food?", "no"),
                    ("Does the dish reflect current {country} culinary innovations and preferences?", "yes"),
                    ("Does the food ignore modern {country} dietary trends and cooking techniques?", "no"),
                ],
                "general": [
                    ("Does the food represent typical {country} culinary traditions?", "yes"),
                    ("Are there foreign culinary elements unrelated to {country} cuisine?", "no"),
                    ("Would this dish be recognizable as part of {country} food culture?", "yes"),
                    ("Are there elements that would be culturally inappropriate in {country} cuisine?", "no"),
                ]
            }
        }

    @torch.inference_mode()
    def generate(
        self,
        sample: EnhancedCulturalEvalSample,
        docs: Sequence[RetrievedDoc],
        max_questions: int,
    ) -> Tuple[List[CulturalQuestion], str]:
        """Enhanced question generation with metadata context."""
        context_text = "\n".join(
            f"[Doc {idx} | {doc.metadata.get('section', 'unknown')}] {doc.text}"
            for idx, doc in enumerate(docs, start=1)
        )

        # Try model-based generation with enhanced context
        for attempt in range(2):
            prompt = self._build_enhanced_instruction(sample, context_text, max_questions, reinforce=attempt == 1)
            raw_items = self._invoke_llm(prompt)
            questions = self._convert_items(raw_items, max_questions)
            processed = self._post_process(questions, max_questions)
            if processed:
                return processed, "model"

        # Enhanced heuristic fallback
        if self.allow_fallback:
            heuristic_questions = self._enhanced_heuristic_questions(sample)
            if heuristic_questions:
                return heuristic_questions[:max_questions], "enhanced_heuristic"
            return [self._fallback_question(sample)], "fallback"
        return [], "none"

    def _build_enhanced_instruction(
        self, sample: EnhancedCulturalEvalSample, context: str, max_questions: int, reinforce: bool
    ) -> str:
        """Build instruction with enhanced metadata context."""
        metadata_context = []
        if sample.category:
            metadata_context.append(f"Category: {sample.category}")
        if sample.sub_category:
            metadata_context.append(f"Sub-category: {sample.sub_category}")
        if sample.variant:
            metadata_context.append(f"Variant: {sample.variant}")
        
        metadata_str = " | ".join(metadata_context) if metadata_context else "General"
        
        reinforcement = ""
        if reinforce and self.min_negative > 0:
            reinforcement = (
                f"Focus on {sample.category} {sample.variant} specific cultural elements. "
                "Include negative checks for inappropriate elements."
            )
        
        neg_line = (
            f"- At least {self.min_negative} questions must have expected_answer 'no'.\n"
            if self.min_negative > 0
            else ""
        )
        editing_line = sample.editing_prompt or "<none>"
        
        return (
            f"Design yes/no questions evaluating whether an image reflects {sample.country} cultural elements "
            f"for {metadata_str}.\n"
            "Use the provided factual snippets. Keep each question <=25 words and unique.\n"
            "Return a JSON array:\n"
            "[{\"question\": str, \"expected_answer\": \"yes\"/\"no\", \"rationale\": str}, ...]\n"
            "Rules:\n"
            f"{neg_line}"
            "- 'yes' means culturally appropriate elements SHOULD appear.\n"
            "- 'no' means elements MUST NOT appear.\n"
            "- Cite supporting doc IDs in rationale when possible.\n"
            f"- Focus on {sample.category}-specific cultural aspects.\n"
            f"{reinforcement}\n"
            f"Prompt: {sample.prompt}\n"
            f"Editing prompt: {editing_line}\n"
            f"Cultural Context ({metadata_str}):\n{context}"
        )

    def _enhanced_heuristic_questions(self, sample: EnhancedCulturalEvalSample) -> List[CulturalQuestion]:
        """Generate questions using enhanced templates based on metadata."""
        category = sample.category or "general"
        variant = sample.variant or "general"
        
        # Get category-specific templates
        category_templates = self.enhanced_templates.get(category, {})
        variant_templates = category_templates.get(variant, category_templates.get("general", []))
        
        # If no specific templates, use generic ones
        if not variant_templates:
            variant_templates = [
                ("Does the image accurately represent {country} cultural elements?", "yes"),
                ("Are there foreign cultural elements unrelated to {country}?", "no"),
                ("Does the image match the prompt: {prompt}?", "yes"),
                ("Do you observe elements that would clash with cultural expectations for {country}?", "no"),
            ]
        
        questions = []
        for template, expected in variant_templates[:self.min_questions]:
            question_text = template.format(country=sample.country, prompt=sample.prompt)
            rationale = f"Template-based question for {category} {variant} in {sample.country}"
            questions.append(CulturalQuestion(question_text, expected, rationale))
        
        return questions

    def _invoke_llm(self, instruction: str) -> List[Dict[str, str]]:
        """Invoke LLM with enhanced error handling."""
        try:
            if self.use_chat_template:
                messages = [
                    {"role": "system", "content": "You are a cultural domain expert."},
                    {"role": "user", "content": instruction},
                ]
                encoded = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)
            else:
                encoded = self.tokenizer.encode(
                    f"system You are a cultural domain expert. user {instruction}",
                    return_tensors="pt"
                ).to(self.device)

            if self.debug:
                print(f"[DEBUG] question LLM raw output: {self.tokenizer.decode(encoded[0])[:200]}")

            with torch.inference_mode():
                outputs = self.model.generate(
                    encoded,
                    attention_mask=torch.ones_like(encoded),
                    max_new_tokens=512,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
            response = self.tokenizer.decode(outputs[0][encoded.shape[1]:], skip_special_tokens=True)
            if self.debug:
                print(f"[DEBUG] LLM response: {response[:200]}")
            
            return self._parse_json_response(response)
            
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] LLM generation failed: {e}")
            return []

    def _parse_json_response(self, response: str) -> List[Dict[str, str]]:
        """Parse JSON response with enhanced error handling."""
        # Try to extract JSON array
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []
        
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_str = json_match.group()
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
            json_str = re.sub(r"'", '"', json_str)  # Replace single quotes
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return []

    def _convert_items(self, raw_items: List[Dict[str, str]], max_questions: int) -> List[CulturalQuestion]:
        """Convert raw items to CulturalQuestion objects."""
        questions = []
        for item in raw_items[:max_questions]:
            if isinstance(item, dict) and all(k in item for k in ["question", "expected_answer", "rationale"]):
                expected = item["expected_answer"].lower().strip()
                if expected in ["yes", "no"]:
                    questions.append(CulturalQuestion(
                        question=item["question"].strip(),
                        expected_answer=expected,
                        rationale=item["rationale"].strip()
                    ))
        return questions

    def _post_process(self, questions: List[CulturalQuestion], max_questions: int) -> List[CulturalQuestion]:
        """Post-process questions to ensure quality and diversity."""
        if len(questions) < self.min_questions:
            return []
        
        # Check for negative question requirement
        negative_count = sum(1 for q in questions if q.expected_answer == "no")
        if negative_count < self.min_negative:
            return []
        
        # Remove duplicates and limit
        seen = set()
        unique_questions = []
        for q in questions:
            q_key = q.question.lower().strip()
            if q_key not in seen:
                seen.add(q_key)
                unique_questions.append(q)
        
        return unique_questions[:max_questions]

    def _fallback_question(self, sample: EnhancedCulturalEvalSample) -> CulturalQuestion:
        """Generate a fallback question."""
        return CulturalQuestion(
            question=f"Does this image appropriately represent {sample.country} cultural elements?",
            expected_answer="yes",
            rationale="Generic fallback question for cultural appropriateness."
        )

    def generate_dimensional(
        self,
        sample: EnhancedCulturalEvalSample,
    ) -> Dict[CulturalDimension, List[CulturalQuestion]]:
        """Generate dimension-tagged questions using templates (Phase 2).

        Returns a dict mapping each CulturalDimension to a list of
        CulturalQuestion objects derived from DIMENSION_QUESTION_TEMPLATES.
        """
        category = sample.category or "cultural"
        variant = sample.variant or "general"
        result: Dict[CulturalDimension, List[CulturalQuestion]] = {}

        for dim in CulturalDimension:
            questions_text = format_dimension_questions(
                dim, sample.country, category, variant,
            )
            dim_questions = []
            for q_text in questions_text:
                # Negative-phrased probes (containing "incorrect", "foreign", etc.)
                # expect "no" for a culturally authentic image
                negative_keywords = [
                    "incorrect", "misattributed", "inappropriate", "foreign",
                    "artificial", "inaccurate", "misused", "anachronistic",
                    "incompatible", "excessively",
                ]
                is_negative = any(kw in q_text.lower() for kw in negative_keywords)
                expected = "no" if is_negative else "yes"
                dim_questions.append(CulturalQuestion(
                    question=q_text,
                    expected_answer=expected,
                    rationale=f"Dimensional template ({dim.value}) for {category} in {sample.country}",
                ))
            result[dim] = dim_questions

        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def demonym(country: str) -> str:
    """Convert country name to demonym (nationality adjective)."""
    mapping = {
        "china": "Chinese", 
        "korea": "Korean", 
        "south korea": "Korean",
        "india": "Indian", 
        "kenya": "Kenyan", 
        "nigeria": "Nigerian",
        "united states": "American", 
        "usa": "American"
    }
    return mapping.get(country.strip().lower(), country.title())


# ---------------------------------------------------------------------------
# Enhanced VLM client with group evaluation
# ---------------------------------------------------------------------------

class EnhancedVLMClient:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        debug: bool = False,
    ) -> None:
        model_kwargs = {"trust_remote_code": True}

        # Use fp16 directly — RTX 4090 24GB has enough VRAM for 8B model (~17GB).
        # BitsAndBytes 4-bit quantization crashes on this environment, so use fp16 instead.
        model_kwargs["dtype"] = torch.float16

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        # Pin VLM to GPU 0 so GPU 1 stays free for the Edit model
        _device_map = {"": 0} if torch.cuda.device_count() >= 2 else "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map=_device_map, **model_kwargs)
        self.model.eval()
        self.device = device
        self.debug = debug

    def evaluate_group_best_worst(
        self,
        samples: List[EnhancedCulturalEvalSample],
        context: str
    ) -> Tuple[str, str, str]:  # (best_uid, worst_uid, reasoning)
        """Evaluate a group of images to select best and worst."""
        if len(samples) <= 1:
            return samples[0].uid if samples else "", "", "Only one or no images in group"
        
        # Prepare images and descriptions
        images = []
        descriptions = []
        for i, sample in enumerate(samples):
            try:
                img = Image.open(sample.image_path).convert("RGB")
                images.append(img)
                descriptions.append(f"Image {i+1} ({sample.step}): {sample.prompt}")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Failed to load image {sample.image_path}: {e}")
                continue
        
        if len(images) < 2:
            return samples[0].uid if samples else "", "", "Insufficient valid images"
        
        
        try:
            # Build evaluation prompt
            image_list = "\n".join(descriptions)
            prompt = f"""
You are evaluating images for cultural representation and prompt alignment.

Cultural Context for {samples[0].country}:
{context}

Images to evaluate:
{image_list}

Select the BEST and WORST images based on:
1. Cultural representation accuracy for {samples[0].country}
2. Alignment with the given prompts
3. Cultural appropriateness and authenticity

Return ONLY a JSON object:
{{
  "best_image": <number 1-{len(images)}>,
  "worst_image": <number 1-{len(images)}>,
  "reasoning": "<brief explanation>"
}}
"""
            
            # Use all images in messages content
            content = [{"type":"text","text":prompt}] + [{"type":"image","image":img} for img in images]
            messages = [
                {"role": "system", "content": "You are a cultural evaluation expert."},
                {"role": "user", "content": content}
            ]
            
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=images, return_tensors="pt", padding=True).to(self.device)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
                response = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Parse response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                best_idx = result.get("best_image", 1) - 1
                worst_idx = result.get("worst_image", 1) - 1
                reasoning = result.get("reasoning", "No reasoning provided")
                
                best_uid = samples[best_idx].uid if 0 <= best_idx < len(samples) else samples[0].uid
                worst_uid = samples[worst_idx].uid if 0 <= worst_idx < len(samples) else samples[-1].uid
                
                return best_uid, worst_uid, reasoning
            else:
                # Fallback: first as best, last as worst
                return samples[0].uid, samples[-1].uid, "Failed to parse VLM response"
                
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Group evaluation failed: {e}")
            return samples[0].uid, samples[-1].uid, f"Evaluation error: {str(e)}"

    def evaluate_cultural_scores(
        self,
        image_path: Path,
        prompt: str,
        editing_prompt: str,
        context: str,
        country: str,
        iteration_number: int = 0,
        previous_cultural_score: Optional[int] = None,
        previous_prompt_score: Optional[int] = None
    ) -> Tuple[int, int]:  # (cultural_representative, prompt_alignment)
        """Evaluate Cultural Representative (1-10) and Prompt Alignment (1-10) scores."""
        try:
            # Use demonym function instead of hardcoded "Chinese"
            dem = demonym(country)

            # Build iteration context for more accurate scoring
            iteration_context = ""
            if iteration_number > 0 and previous_cultural_score is not None:
                iteration_context = f"""
ITERATION CONTEXT:
- This is iteration {iteration_number} (edited version)
- Previous scores: Cultural={previous_cultural_score}/10, Prompt={previous_prompt_score}/10
- Edits made: {editing_prompt[:200]}
- You should score HIGHER if improvements were made, LOWER if issues remain or got worse
"""

            eval_question = f"""You are a STRICT cultural expert evaluator. Rate this image on a 1-10 scale with HIGH STANDARDS.
{iteration_context}
SCORING GUIDELINES:
- 9-10: Nearly perfect, authentic {dem} cultural representation with NO significant errors
- 7-8: Good representation with ONLY minor issues
- 5-6: Noticeable problems or inaccuracies that need fixing
- 3-4: Multiple serious cultural errors or misrepresentations
- 1-2: Completely incorrect or inappropriate

Examine critically:
1. Traditional {dem} cultural elements - are they AUTHENTIC and ACCURATE?
2. Colors, patterns, shapes - do they match REAL {dem} cultural artifacts?
3. Any elements from other cultures that DON'T belong?
4. How well it matches the prompt: "{prompt}"

BE CRITICAL. If you find ANY significant errors, the score MUST be 6 or lower.
If you find multiple severe issues, the score MUST be 4 or lower.

After your analysis, end with: Cultural:X,Prompt:Y
Example: Cultural:5,Prompt:7"""

            # Use the exact same format as answer() function that works
            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "system", "content": "You are a VERY STRICT cultural expert. Be critical and demanding. High scores (8+) should be RARE and only for truly excellent work. Any significant errors MUST result in scores of 6 or below. Respond with format Cultural:X,Prompt:Y where X,Y are 1-10."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": eval_question},
                        {"type": "image", "image": image},
                    ],
                },
            ]
            
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(self.device)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Increased from 30 to allow proper reasoning
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Extract only the new tokens (response)
            response_ids = generate_ids[0][len(inputs['input_ids'][0]):]
            output = self.processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            if self.debug:
                print(f"[DEBUG] Cultural scoring raw response: '{output}'")
                print(f"[DEBUG] Response length: {len(output)}")
                numbers_found = re.findall(r'\d+', output)
                print(f"[DEBUG] Numbers found: {numbers_found}")
            
            # Parse scores - try multiple formats
            # FIXED: Better defaults and fallback logic
            cultural_score = None  # Will be determined by parsing or text analysis
            prompt_score = None

            try:
                # Method 1: Look for Cultural:X,Prompt:Y format (strict)
                cultural_match = re.search(r'Cultural[:\s]+(\d+)', output, re.IGNORECASE)
                prompt_match = re.search(r'Prompt[:\s]+(\d+)', output, re.IGNORECASE)
                
                if cultural_match and prompt_match:
                    cultural_score = int(cultural_match.group(1))
                    prompt_score = int(prompt_match.group(1))
                    if self.debug:
                        print(f"[DEBUG] Method 1 (regex) - Cultural:{cultural_score}, Prompt:{prompt_score}")
                # Method 2: Look for any two numbers
                elif len(numbers_found) >= 2:
                    cultural_score = int(numbers_found[0])
                    prompt_score = int(numbers_found[1])
                    if self.debug:
                        print(f"[DEBUG] Method 2 (numbers) - Cultural:{cultural_score}, Prompt:{prompt_score}")
                # Method 3: Look for comma-separated numbers
                elif ',' in output:
                    parts = output.split(',')
                    if len(parts) >= 2:
                        cultural_nums = re.findall(r'\d+', parts[0])
                        prompt_nums = re.findall(r'\d+', parts[1])
                        if cultural_nums and prompt_nums:
                            cultural_score = int(cultural_nums[0])
                            prompt_score = int(prompt_nums[0])
                            if self.debug:
                                print(f"[DEBUG] Method 3 (comma) - Cultural:{cultural_score}, Prompt:{prompt_score}")

                # FIXED: Fallback - infer from text analysis if parsing failed
                if cultural_score is None or prompt_score is None:
                    output_lower = output.lower()
                    # Positive indicators
                    if any(word in output_lower for word in ['accurate', 'correct', 'authentic', 'good', 'well', 'proper']):
                        cultural_score = cultural_score or 7
                        prompt_score = prompt_score or 7
                        if self.debug:
                            print(f"[DEBUG] Positive text detected, using default 7/10")
                    # Negative indicators
                    elif any(word in output_lower for word in ['incorrect', 'inaccurate', 'wrong', 'poor', 'lacks', 'missing']):
                        cultural_score = cultural_score or 4
                        prompt_score = prompt_score or 4
                        if self.debug:
                            print(f"[DEBUG] Negative text detected, using default 4/10")
                    else:
                        # Neutral fallback
                        cultural_score = cultural_score or 5
                        prompt_score = prompt_score or 5
                        if self.debug:
                            print(f"[DEBUG] No clear sentiment, using default 5/10")
                
                # Clamp to valid range (1-10 scale)
                if self.debug:
                    print(f"[DEBUG] Before clamping: cultural={cultural_score}, prompt={prompt_score}")
                cultural_score = max(1, min(10, cultural_score))
                prompt_score = max(1, min(10, prompt_score))
                if self.debug:
                    print(f"[DEBUG] After clamping: cultural={cultural_score}, prompt={prompt_score}")

            except (ValueError, IndexError) as e:
                if self.debug:
                    print(f"[DEBUG] Score parsing failed: {e}, output was: '{output}', using defaults")

            if self.debug:
                print(f"[DEBUG] Final scores: cultural={cultural_score}, prompt={prompt_score}")
            
            return cultural_score, prompt_score
            
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Cultural scoring failed: {e}")
                import traceback
                traceback.print_exc()
            return 3, 3  # default neutral scores

    def query_with_context(
        self,
        image: Image.Image,
        query: str,
        context: str,
        max_tokens: int = 300
    ) -> str:
        """
        Query VLM with detailed context for open-ended answers.

        Args:
            image: PIL Image
            query: Question/instruction for VLM
            context: Cultural context from knowledge base
            max_tokens: Max response length

        Returns:
            VLM's detailed response
        """
        try:
            messages = [
                {"role": "system", "content": "You are a cultural expert analyzing images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Cultural Knowledge:\n{context}\n\n{query}"},
                        {"type": "image", "image": image},
                    ],
                },
            ]

            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(self.device)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                response = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] VLM query_with_context failed: {e}")
            return ""

    def answer(self, image_path: Path, question: str, context: str) -> str:
        """Answer a yes/no question about an image."""
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "system", "content": "You are a cultural compliance checker. Reply 'yes' or 'no'."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Context:\n{context}\nQuestion: {question}\nAnswer strictly with yes or no."},
                        {"type": "image", "image": image},
                    ],
                },
            ]

            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(self.device)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
                response = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            return self._normalize_answer(response)

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] VLM answer failed for {image_path}: {e}")
            return "ambiguous"

    def _normalize_answer(self, text: str) -> str:
        """Normalize VLM response to yes/no/ambiguous."""
        text = text.lower().strip()
        first_word = text.split()[0] if text.split() else ""

        yes_patterns = [r"\byes\b", r"\btrue\b", r"\by\b"]
        no_patterns = [r"\bno\b", r"\bfalse\b", r"\bn\b"]

        for pattern in yes_patterns:
            if re.search(pattern, first_word):
                return "yes"
        for pattern in no_patterns:
            if re.search(pattern, first_word):
                return "no"
        return "ambiguous"

    # -- CultScore dimensional evaluation (Phase 2) --

    def evaluate_dimension_scores(
        self,
        image_path: Path,
        country: str,
        category: str,
        context: str,
        variant: str = "general",
        num_passes: int = 3,
        temperature: float = 0.7,
    ) -> Dict[CulturalDimension, List[float]]:
        """Evaluate each cultural dimension with multi-pass stochastic scoring.

        Returns a dict mapping each dimension to a list of 0-1 scores (one per pass).
        """
        dimension_raw_scores: Dict[CulturalDimension, List[float]] = {}

        for dim in CulturalDimension:
            questions = format_dimension_questions(dim, country, category, variant)
            if not questions:
                continue

            pass_scores: List[float] = []
            for _pass_idx in range(num_passes):
                # Ask each question for this dimension; average yes-rate → score
                yes_count = 0
                total = 0
                for q in questions:
                    ans = self._stochastic_answer(image_path, q, context, temperature)
                    if ans == "yes":
                        yes_count += 1
                    total += 1
                pass_score = yes_count / total if total > 0 else 0.0
                pass_scores.append(pass_score)

            dimension_raw_scores[dim] = pass_scores

        return dimension_raw_scores

    def _stochastic_answer(
        self, image_path: Path, question: str, context: str, temperature: float
    ) -> str:
        """Answer a yes/no question with stochastic sampling (do_sample=True)."""
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "system", "content": "You are a cultural compliance checker. Reply 'yes' or 'no'."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Context:\n{context}\nQuestion: {question}\nAnswer strictly with yes or no."},
                        {"type": "image", "image": image},
                    ],
                },
            ]
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(self.device)
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
                response = self.processor.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )

            return self._normalize_answer(response)
        except Exception as e:
            if self.debug:
                logger.debug(f"Stochastic answer failed for {image_path}: {e}")
            return "ambiguous"


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, data: CheckpointData, model_name: str) -> None:
        """Save checkpoint data."""
        checkpoint_file = self.checkpoint_dir / f"{model_name}_checkpoint.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(data, f)
        if hasattr(data, 'current_index'):
            print(f"[CHECKPOINT] Saved at sample {data.current_index}/{data.total_samples}")
    
    def load_checkpoint(self, model_name: str) -> Optional[CheckpointData]:
        """Load checkpoint data if exists."""
        checkpoint_file = self.checkpoint_dir / f"{model_name}_checkpoint.pkl"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
        return None
    
    def clear_checkpoint(self, model_name: str) -> None:
        """Clear checkpoint after successful completion."""
        checkpoint_file = self.checkpoint_dir / f"{model_name}_checkpoint.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()


# ---------------------------------------------------------------------------
# Enhanced main pipeline
# ---------------------------------------------------------------------------

def load_enhanced_samples_from_csv(csv_path: Path, image_root: Path) -> List[EnhancedCulturalEvalSample]:
    """Load samples with enhanced metadata support."""
    samples = []
    print(f"[DEBUG] Loading CSV from: {csv_path}")
    print(f"[DEBUG] CSV exists: {csv_path.exists()}")
    
    df = pd.read_csv(csv_path)
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")
    print(f"[DEBUG] First row data: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
    
    # Detect format types
    has_extended_metadata = all(col in df.columns for col in ["model", "country", "category", "sub_category", "variant"])
    has_basic_metadata = "T2I prompt" in df.columns and "I2I prompt" in df.columns
    
    print(f"[DEBUG] has_extended_metadata: {has_extended_metadata}")
    print(f"[DEBUG] has_basic_metadata: {has_basic_metadata}")
    
    for _, row in df.iterrows():
        # Extract basic info based on format
        if has_extended_metadata:
            prompt = row["T2I prompt"] if "T2I prompt" in df.columns else row.get("prompt", "")
            editing_prompt = row["I2I prompt"] if "I2I prompt" in df.columns else row.get("editing_prompt", "")
            country = row["country"]
            model = row["model"]
            category = row["category"]
            sub_category = row["sub_category"]  
            variant = row["variant"] if pd.notna(row["variant"]) else None
        elif has_basic_metadata:
            # Basic hidream/flux format
            prompt = row["T2I prompt"]
            editing_prompt = row["I2I prompt"]
            country = extract_country_from_prompt(prompt)
            model = None
            category = extract_category_from_prompt(prompt)
            sub_category = extract_sub_category_from_prompt(prompt)
            variant = extract_variant_from_prompt(prompt)
        else:
            # Legacy format support
            prompt = row["prompt"]
            editing_prompt = row.get("editing_prompt", "")
            country = extract_country_from_prompt(prompt)
            model = None
            category = None
            sub_category = None
            variant = None
        
        # Generate group_id from metadata
        if has_extended_metadata:
            group_id = f"{model}_{country}_{category}_{sub_category}_{variant or 'general'}"
        elif has_basic_metadata:
            # Extract model from image path or use default
            sample_step_col = [col for col in df.columns if col.startswith(("base", "step0"))][0]
            sample_path = str(row[sample_step_col])
            if "hidream" in sample_path:
                model = "hidream"
            elif "flux" in sample_path:
                model = "flux"
            else:
                model = "unknown"
            group_id = f"{model}_{country.lower().replace(' ', '_')}_{category}_{sub_category}_{variant or 'general'}"
        else:
            group_id = f"legacy_{hash(prompt) % 10000}"
        
        # Process step columns
        step_columns = [col for col in df.columns if col.startswith(("step", "base", "edit_"))]
        for step_col in step_columns:
            if pd.notna(row[step_col]) and row[step_col]:
                image_path = image_root / row[step_col]
                if image_path.exists():
                    # Extract step name
                    if step_col.startswith("base") or step_col == "step0_path":
                        step = "step0"
                    elif step_col.startswith("edit_") or step_col.startswith("step"):
                        step = step_col.split("_")[-1] if "_" in step_col else step_col.replace("step", "step").replace("_path", "")
                    else:
                        step = step_col
                    
                    uid = f"{group_id}::{step}"
                    
                    samples.append(EnhancedCulturalEvalSample(
                        uid=uid,
                        group_id=group_id,
                        step=step,
                        prompt=prompt,
                        country=country,
                        image_path=image_path,
                        editing_prompt=editing_prompt,
                        model=model,
                        category=category,
                        sub_category=sub_category,
                        variant=variant,
                    ))
    
    return samples


def extract_country_from_prompt(prompt: str) -> str:
    """Extract country from prompt for legacy support."""
    countries = ["china", "india", "kenya", "nigeria", "korea", "united states"]
    prompt_lower = prompt.lower()
    for country in countries:
        if country in prompt_lower:
            return country.title()
    return "Unknown"


def extract_category_from_prompt(prompt: str) -> str:
    """Extract category from prompt."""
    categories = ["architecture", "art", "event", "fashion", "food", "landscape", "people", "wildlife"]
    prompt_lower = prompt.lower()
    for category in categories:
        if category in prompt_lower:
            return category
    
    # Additional mappings
    if any(word in prompt_lower for word in ["house", "building", "landmark"]):
        return "architecture"
    elif any(word in prompt_lower for word in ["dance", "painting", "sculpture"]):
        return "art"
    elif any(word in prompt_lower for word in ["festival", "wedding", "funeral", "sport", "game", "ritual"]):
        return "event"
    elif any(word in prompt_lower for word in ["clothing", "accessories", "makeup"]):
        return "fashion"
    elif any(word in prompt_lower for word in ["food", "dish", "beverage", "dessert", "snack"]):
        return "food"
    elif any(word in prompt_lower for word in ["landscape", "city", "countryside", "nature"]):
        return "landscape"
    elif any(word in prompt_lower for word in ["people", "person", "athlete", "chef", "doctor", "farmer", "teacher", "student"]):
        return "people"
    elif any(word in prompt_lower for word in ["animal", "plant"]):
        return "wildlife"
    
    return "general"


def extract_sub_category_from_prompt(prompt: str) -> str:
    """Extract sub-category from prompt."""
    prompt_lower = prompt.lower()
    
    # Architecture
    if any(word in prompt_lower for word in ["house", "building"]):
        return "house"
    elif "landmark" in prompt_lower:
        return "landmark"
    
    # Art
    elif "dance" in prompt_lower:
        return "dance"
    elif "painting" in prompt_lower:
        return "painting"
    elif "sculpture" in prompt_lower:
        return "sculpture"
    
    # Event
    elif "festival" in prompt_lower:
        return "festival"
    elif "wedding" in prompt_lower:
        return "wedding"
    elif "funeral" in prompt_lower:
        return "funeral"
    elif "sport" in prompt_lower:
        return "sport"
    elif "game" in prompt_lower:
        return "game"
    elif "ritual" in prompt_lower:
        return "religious_ritual"
    
    # Fashion
    elif "clothing" in prompt_lower:
        return "clothing"
    elif "accessories" in prompt_lower:
        return "accessories"
    elif "makeup" in prompt_lower:
        return "makeup"
    
    # Food
    elif any(word in prompt_lower for word in ["main dish", "dish"]):
        return "main_dish"
    elif "dessert" in prompt_lower:
        return "dessert"
    elif "snack" in prompt_lower:
        return "snack"
    elif "beverage" in prompt_lower:
        return "beverage"
    elif any(word in prompt_lower for word in ["staple food", "staple"]):
        return "staple_food"
    
    # Landscape
    elif "city" in prompt_lower:
        return "city"
    elif "countryside" in prompt_lower:
        return "countryside"
    elif "nature" in prompt_lower:
        return "nature"
    
    # People
    elif "athlete" in prompt_lower:
        return "athlete"
    elif any(word in prompt_lower for word in ["bride", "groom"]):
        return "bride_and_groom"
    elif "celebrity" in prompt_lower:
        return "celebrity"
    elif "chef" in prompt_lower:
        return "chef"
    elif "daily life" in prompt_lower:
        return "daily_life"
    elif "doctor" in prompt_lower:
        return "doctor"
    elif "farmer" in prompt_lower:
        return "farmer"
    elif "model" in prompt_lower:
        return "model"
    elif "president" in prompt_lower:
        return "president"
    elif "soldier" in prompt_lower:
        return "soldier"
    elif "student" in prompt_lower:
        return "student"
    elif "teacher" in prompt_lower:
        return "teacher"
    
    # Wildlife
    elif "animal" in prompt_lower:
        return "animal"
    elif "plant" in prompt_lower:
        return "plant"
    
    return "general"


def extract_variant_from_prompt(prompt: str) -> str:
    """Extract variant from prompt."""
    prompt_lower = prompt.lower()
    if "traditional" in prompt_lower:
        return "traditional"
    elif "modern" in prompt_lower:
        return "modern"
    else:
        return "general"


def calculate_metrics(questions: List[CulturalQuestion], answers: List[str]) -> Tuple[float, float, float, float]:
    """Calculate accuracy, precision, recall, F1."""
    if not questions or len(questions) != len(answers):
        return 0.0, 0.0, 0.0, 0.0
    
    tp = sum(1 for q, a in zip(questions, answers) if q.expected_answer == "yes" and a == "yes")
    fp = sum(1 for q, a in zip(questions, answers) if q.expected_answer == "no" and a == "yes")
    fn = sum(1 for q, a in zip(questions, answers) if q.expected_answer == "yes" and a == "no")
    tn = sum(1 for q, a in zip(questions, answers) if q.expected_answer == "no" and a == "no")
    
    total = len(questions)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1


def enhanced_main() -> None:
    """Enhanced main function with resumption support and performance optimizations."""
    parser = argparse.ArgumentParser(description="Enhanced cultural metric evaluation")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True) 
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--detail-csv", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--question-model", default="openai/gpt-oss-20b")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--min-questions", type=int, default=4)
    parser.add_argument("--min-negative", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch processing size")
    parser.add_argument("--save-frequency", type=int, default=10, help="Save checkpoint every N samples")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Extract model name for checkpoint
    model_name = args.input_csv.stem
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    
    # Load samples
    print(f"[LOADING] Reading samples from {args.input_csv}")
    samples = load_enhanced_samples_from_csv(args.input_csv, args.image_root)
    
    # Apply max_samples limit if specified
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]
        print(f"[LOADING] Limited to {len(samples)} samples (max_samples={args.max_samples})")
    else:
        print(f"[LOADING] Loaded {len(samples)} samples")
    
    # Initialize components
    print(f"[INIT] Loading knowledge base from {args.index_dir}")
    kb = EnhancedCulturalKnowledgeBase(args.index_dir)
    
    print(f"[INIT] Loading question generator: {args.question_model}")
    question_gen = EnhancedQuestionGenerator(
        args.question_model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        min_questions=args.min_questions,
        min_negative=args.min_negative,
        debug=args.debug,
    )
    
    print(f"[INIT] Loading VLM: {args.vlm_model}")
    vlm = EnhancedVLMClient(
        args.vlm_model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        debug=args.debug,
    )

    # Initialize CultScore computation engine (Phase 2)
    cultscore_config = CultScoreConfig()
    cultscore_computer = CultScoreComputer(cultscore_config)
    print(f"[INIT] CultScore engine initialized (passes={cultscore_config.num_passes}, temp={cultscore_config.temperature})")

    # Initialize failure detector (Phase 4)
    from .components.failure_detector import FailureModeDetector
    failure_detector = FailureModeDetector(vlm_model=args.vlm_model)
    print(f"[INIT] Failure detector initialized")

    # Check for resume
    start_index = 0
    completed_uids = set()
    results = []
    
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(model_name)
        if checkpoint:
            print(f"[RESUME] Found checkpoint with {len(checkpoint.completed_samples)} completed samples")
            completed_uids = set(checkpoint.completed_samples)
            results = checkpoint.results
            start_index = checkpoint.current_index
    
    # Filter out completed samples
    remaining_samples = [s for s in samples if s.uid not in completed_uids]
    print(f"[PROCESSING] {len(remaining_samples)} samples remaining")
    
    # Process samples
    for i, sample in enumerate(tqdm(remaining_samples, desc="Cultural metric", initial=start_index)):
        if sample.uid in completed_uids:
            continue
            
        start_time = time.time()
        
        try:
            # Retrieve cultural context
            docs = kb.retrieve_contextual(sample.prompt, sample, args.top_k)
            context_text = "\n".join(f"[Doc {idx}] {doc.text}" for idx, doc in enumerate(docs, 1))
            
            # Generate questions
            questions, source = question_gen.generate(sample, docs, args.max_questions)
            
            if args.debug:
                print(f"[DEBUG] Processing {sample.uid} ({sample.country})")
                print(f"[DEBUG] {sample.uid}: evaluating {len(questions)} questions (source={source})")
            
            # Get VLM answers
            answers = []
            for q in questions:
                answer = vlm.answer(sample.image_path, q.question, context_text)
                answers.append(answer)
                if args.debug:
                    print(f"[DEBUG]    Q: {q.question} | expected={q.expected_answer} | raw={answer}")
            
            # Calculate metrics
            accuracy, precision, recall, f1 = calculate_metrics(questions, answers)
            processing_time = time.time() - start_time
            
            # Evaluate cultural scores
            cultural_score, prompt_score = vlm.evaluate_cultural_scores(
                sample.image_path, sample.prompt, sample.editing_prompt, context_text, sample.country
            )
            
            if args.debug:
                print(f"[DEBUG]    cultural_representative={cultural_score}, prompt_alignment={prompt_score}")

            # CultScore dimensional evaluation (Phase 2)
            cultural_profile = None
            cs_value = None
            cs_conf = None
            cs_pen = None
            dim_scores_flat = None

            if cultscore_computer is not None:
                try:
                    dim_raw = vlm.evaluate_dimension_scores(
                        image_path=sample.image_path,
                        country=sample.country,
                        category=sample.category or "general",
                        context=context_text,
                        variant=sample.variant or "general",
                        num_passes=cultscore_config.num_passes,
                        temperature=cultscore_config.temperature,
                    )

                    # Compute mean source authority from retrieved docs
                    mean_authority = (
                        sum(d.source_authority for d in docs) / len(docs)
                        if docs else 1.0
                    )

                    dim_scores: Dict[CulturalDimension, DimensionScore] = {}
                    for dim, raw_list in dim_raw.items():
                        dim_scores[dim] = cultscore_computer.build_dimension_score(
                            dim, raw_list, source_authority=mean_authority,
                        )

                    # Phase 4: detect failures and compute penalties
                    failure_penalties = []
                    try:
                        failure_detections = failure_detector.detect_enhanced(
                            image_path=sample.image_path,
                            country=sample.country,
                            category=sample.category or "general",
                            context=context_text,
                        )
                        failure_penalties = failure_detector.compute_penalties(failure_detections)
                        if args.debug and failure_detections:
                            print(f"[DEBUG]    Failures detected: {[f.mode for f in failure_detections]}")
                    except Exception as fe:
                        if args.debug:
                            print(f"[DEBUG]    Failure detection skipped: {fe}")

                    cultural_profile = cultscore_computer.build_cultural_profile(
                        dimension_scores=dim_scores,
                        category=sample.category or "general",
                        country=sample.country,
                        failure_penalties=failure_penalties,
                    )
                    cs_value = cultural_profile.cultscore
                    cs_conf = cultural_profile.cultscore_confidence
                    cs_pen = cultural_profile.cultscore_penalised
                    dim_scores_flat = {
                        dim.value: ds.raw_score
                        for dim, ds in dim_scores.items()
                    }

                    if args.debug:
                        print(f"[DEBUG]    CultScore={cs_value:.4f} conf={cs_conf:.4f} pen={cs_pen:.4f}")
                except Exception as e:
                    if args.debug:
                        print(f"[DEBUG]    CultScore computation failed: {e}")

            # Create result
            result = EvaluationResult(
                sample=sample,
                questions=questions,
                answers=answers,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                num_questions=len(questions),
                processing_time=processing_time,
                question_source=source,
                cultural_representation_score=cultural_score,
                prompt_alignment_score=prompt_score,
                cultural_profile=cultural_profile,
                cultscore=cs_value,
                cultscore_confidence=cs_conf,
                cultscore_penalised=cs_pen,
                dimension_scores=dim_scores_flat,
            )
            
            results.append(result)
            completed_uids.add(sample.uid)
            
            if args.debug:
                print(f"[DEBUG]    metrics -> acc={accuracy:.3f}, prec={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {sample.uid}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
        
        # Save checkpoint periodically
        if (i + 1) % args.save_frequency == 0:
            checkpoint_data = CheckpointData(
                completed_samples=list(completed_uids),
                results=results,
                timestamp=time.strftime("%Y%m%d_%H%M%S"),
                total_samples=len(samples),
                current_index=start_index + i + 1,
            )
            checkpoint_manager.save_checkpoint(checkpoint_data, model_name)
    
    # Process group best/worst selection
    print(f"[GROUP EVAL] Processing best/worst selection for {len(set(s.group_id for s in samples))} groups")
    group_results = {}
    for sample in samples:
        if sample.group_id not in group_results:
            group_results[sample.group_id] = []
        group_results[sample.group_id].append(sample)
    
    # Add group evaluation results
    group_evaluations = {}
    for group_id, group_samples in group_results.items():
        if len(group_samples) > 1:
            # Get context for first sample (assuming same country/category for group)
            first_sample = group_samples[0]
            docs = kb.retrieve_contextual(first_sample.prompt, first_sample, args.top_k)
            context_text = "\n".join(f"[Doc {idx}] {doc.text}" for idx, doc in enumerate(docs, 1))
            
            best_uid, worst_uid, reasoning = vlm.evaluate_group_best_worst(group_samples, context_text)
            print(f"[GROUP EVAL] {group_id}: Best={best_uid}, Worst={worst_uid}")
            if args.debug:
                print(f"[GROUP EVAL] Reasoning: {reasoning}")
            group_evaluations[group_id] = {
                "best_uid": best_uid,
                "worst_uid": worst_uid,
                "reasoning": reasoning,
            }
    
    # Write results
    print(f"[OUTPUT] Writing {len(results)} results to {args.summary_csv}")
    write_enhanced_results(results, group_evaluations, args.summary_csv, args.detail_csv)
    
    # Clear checkpoint on successful completion
    checkpoint_manager.clear_checkpoint(model_name)
    print("[COMPLETE] Enhanced cultural metric evaluation finished successfully")


def write_enhanced_results(
    results: List[EvaluationResult],
    group_evaluations: Dict[str, Dict[str, str]],
    summary_csv: Path,
    detail_csv: Path,
) -> None:
    """Write enhanced results with group evaluation data."""
    # Write summary
    # Dimension column names
    dim_names = [d.value for d in CulturalDimension]

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uid", "group_id", "step", "country", "category", "sub_category", "variant",
            "accuracy", "precision", "recall", "f1", "num_questions",
            "processing_time", "question_source", "cultural_representative", "prompt_alignment",
            "cultscore", "cultscore_confidence", "cultscore_penalised",
        ] + [f"dim_{d}" for d in dim_names] + [
            "is_best", "is_worst"
        ])

        for result in results:
            sample = result.sample
            group_eval = group_evaluations.get(sample.group_id, {})
            is_best = sample.uid == group_eval.get("best_uid", "")
            is_worst = sample.uid == group_eval.get("worst_uid", "")

            dim_values = []
            for d in dim_names:
                if result.dimension_scores and d in result.dimension_scores:
                    dim_values.append(f"{result.dimension_scores[d]:.4f}")
                else:
                    dim_values.append("")

            writer.writerow([
                sample.uid, sample.group_id, sample.step, sample.country,
                sample.category or "", sample.sub_category or "", sample.variant or "",
                result.accuracy, result.precision, result.recall, result.f1,
                result.num_questions, result.processing_time, result.question_source,
                result.cultural_representation_score or 0, result.prompt_alignment_score or 0,
                f"{result.cultscore:.4f}" if result.cultscore is not None else "",
                f"{result.cultscore_confidence:.4f}" if result.cultscore_confidence is not None else "",
                f"{result.cultscore_penalised:.4f}" if result.cultscore_penalised is not None else "",
            ] + dim_values + [
                is_best, is_worst
            ])
    
    # Write detailed results
    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "uid", "group_id", "step", "country", "category", "sub_category", "variant",
            "question", "expected_answer", "actual_answer", "question_rationale",
            "cultscore", "cultscore_confidence",
        ])

        for result in results:
            sample = result.sample
            for q, a in zip(result.questions, result.answers):
                writer.writerow([
                    sample.uid, sample.group_id, sample.step, sample.country,
                    sample.category or "", sample.sub_category or "", sample.variant or "",
                    q.question, q.expected_answer, a, q.rationale,
                    f"{result.cultscore:.4f}" if result.cultscore is not None else "",
                    f"{result.cultscore_confidence:.4f}" if result.cultscore_confidence is not None else "",
                ])


if __name__ == "__main__":
    enhanced_main()