"""Simplified cultural metric pipeline for single image evaluation."""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


@dataclass
class CulturalQuestion:
    question: str
    expected_answer: str
    rationale: str


@dataclass
class RetrievedDoc:
    text: str
    score: float
    metadata: Dict[str, str]


@dataclass
class CulturalScore:
    accuracy: float
    precision: float
    recall: float
    f1: float
    cultural_representative: int
    prompt_alignment: int
    num_questions: int
    processing_time: float


class SimpleCulturalKnowledgeBase:
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
        country: str,
        category: Optional[str] = None,
        top_k: int = 8,
    ) -> List[RetrievedDoc]:
        """Enhanced retrieval using metadata for better context."""
        # Build enhanced query with metadata
        enhanced_query_parts = [query]
        
        if category:
            enhanced_query_parts.append(category)
            
        enhanced_query = " ".join(enhanced_query_parts)
        
        # Get section bias based on category
        section_bias = self.section_priorities.get(category, ["Culture"])
        
        return self.retrieve(enhanced_query, country, top_k, section_bias)

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
            docs.append(RetrievedDoc(text=meta["text"], score=float(score), metadata=meta))
            if len(docs) >= top_k:
                break

        # Second pass: other countries if needed
        if len(docs) < top_k:
            for score, idx in zip(scores[0], indices[0]):
                meta = self.metadata[idx]
                if country and meta.get("country") == country:
                    continue
                docs.append(RetrievedDoc(text=meta["text"], score=float(score), metadata=meta))
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


class SimpleQuestionGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        min_questions: int = 4,
        min_negative: int = 0,
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
        self.debug = debug
        
        # Enhanced keyword templates organized by category
        self.enhanced_templates = self._build_enhanced_templates()

    def _build_enhanced_templates(self) -> Dict[str, List[Tuple[str, str]]]:
        """Build context-aware question templates."""
        return {
            "architecture": [
                ("Does the architecture show traditional {country} building styles and materials?", "yes"),
                ("Are there modern Western architectural elements that contradict traditional {country} design?", "no"),
                ("Does the structure incorporate traditional {country} cultural motifs or decorative elements?", "yes"),
                ("Are there elements that clash with historical {country} architectural principles?", "no"),
            ],
            "art": [
                ("Does the artwork show traditional {country} artistic techniques and styles?", "yes"),
                ("Are there modern or foreign artistic elements that contradict traditional {country} art?", "no"),
                ("Does the piece incorporate traditional {country} cultural symbols or themes?", "yes"),
                ("Are there elements that go against traditional {country} artistic principles?", "no"),
            ],
            "event": [
                ("Does the event show traditional {country} customs and practices?", "yes"),
                ("Are there modern or foreign elements that contradict traditional {country} ceremonies?", "no"),
                ("Do participants wear traditional {country} attire appropriate for the event?", "yes"),
                ("Are there elements that violate traditional {country} cultural protocols?", "no"),
            ],
            "fashion": [
                ("Does the clothing show traditional {country} garments and textiles?", "yes"),
                ("Are there modern Western fashion elements that contradict traditional {country} dress?", "no"),
                ("Do the garments use traditional {country} colors, patterns, and styling?", "yes"),
                ("Are there elements that violate traditional {country} dress codes or customs?", "no"),
            ],
            "food": [
                ("Does the food show traditional {country} ingredients and cooking methods?", "yes"),
                ("Are there foreign cuisines or ingredients unrelated to traditional {country} food?", "no"),
                ("Does the dish incorporate traditional {country} flavors and presentation?", "yes"),
                ("Are there elements that contradict traditional {country} culinary practices?", "no"),
            ],
            "landscape": [
                ("Does the landscape show typical {country} geographical features?", "yes"),
                ("Are there foreign landscape elements unrelated to {country}?", "no"),
                ("Does the scene represent authentic {country} natural environment?", "yes"),
                ("Are there elements that would be geographically inappropriate for {country}?", "no"),
            ],
            "people": [
                ("Do the people represent typical {country} demographics and characteristics?", "yes"),
                ("Are there foreign cultural elements unrelated to {country} people?", "no"),
                ("Does the scene show authentic {country} social interactions?", "yes"),
                ("Are there elements that would be culturally inappropriate for {country} people?", "no"),
            ],
            "wildlife": [
                ("Does the wildlife represent typical {country} animal species?", "yes"),
                ("Are there foreign animal species unrelated to {country}?", "no"),
                ("Does the scene show authentic {country} natural habitat?", "yes"),
                ("Are there elements that would be geographically inappropriate for {country} wildlife?", "no"),
            ]
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        country: str,
        category: Optional[str],
        docs: Sequence[RetrievedDoc],
        max_questions: int,
    ) -> Tuple[List[CulturalQuestion], str]:
        """Generate questions with metadata context."""
        context_text = "\n".join(
            f"[Doc {idx} | {doc.metadata.get('section', 'unknown')}] {doc.text}"
            for idx, doc in enumerate(docs, start=1)
        )

        # Try model-based generation
        for attempt in range(2):
            prompt_text = self._build_instruction(prompt, country, category, context_text, max_questions, reinforce=attempt == 1)
            raw_items = self._invoke_llm(prompt_text)
            questions = self._convert_items(raw_items, max_questions)
            processed = self._post_process(questions, max_questions)
            if processed:
                return processed, "model"

        # Enhanced heuristic fallback
        heuristic_questions = self._enhanced_heuristic_questions(prompt, country, category)
        if heuristic_questions:
            return heuristic_questions[:max_questions], "enhanced_heuristic"
        return [self._fallback_question(country)], "fallback"

    def _build_instruction(
        self, prompt: str, country: str, category: Optional[str], context: str, max_questions: int, reinforce: bool
    ) -> str:
        """Build instruction with enhanced metadata context."""
        metadata_context = []
        if category:
            metadata_context.append(f"Category: {category}")
        
        metadata_str = " | ".join(metadata_context) if metadata_context else "General"
        
        reinforcement = ""
        if reinforce and self.min_negative > 0:
            reinforcement = (
                f"Focus on {category} specific cultural elements. "
                "Include negative checks for inappropriate elements."
            )
        
        neg_line = (
            f"- At least {self.min_negative} questions must have expected_answer 'no'.\n"
            if self.min_negative > 0
            else ""
        )
        
        return (
            f"Design yes/no questions evaluating whether an image reflects {country} cultural elements "
            f"for {metadata_str}.\n"
            "Use the provided factual snippets. Keep each question <=25 words and unique.\n"
            "Return a JSON array:\n"
            "[{\"question\": str, \"expected_answer\": \"yes\"/\"no\", \"rationale\": str}, ...]\n"
            "Rules:\n"
            f"{neg_line}"
            "- 'yes' means culturally appropriate elements SHOULD appear.\n"
            "- 'no' means elements MUST NOT appear.\n"
            "- Cite supporting doc IDs in rationale when possible.\n"
            f"- Focus on {category}-specific cultural aspects.\n"
            f"{reinforcement}\n"
            f"Prompt: {prompt}\n"
            f"Cultural Context ({metadata_str}):\n{context}"
        )

    def _enhanced_heuristic_questions(self, prompt: str, country: str, category: Optional[str]) -> List[CulturalQuestion]:
        """Generate questions using enhanced templates based on metadata."""
        category = category or "general"
        
        # Get category-specific templates
        category_templates = self.enhanced_templates.get(category, [])
        
        # If no specific templates, use generic ones
        if not category_templates:
            category_templates = [
                ("Does the image accurately represent {country} cultural elements?", "yes"),
                ("Are there foreign cultural elements unrelated to {country}?", "no"),
                ("Does the image match the prompt: {prompt}?", "yes"),
                ("Do you observe elements that would clash with cultural expectations for {country}?", "no"),
            ]
        
        questions = []
        for template, expected in category_templates[:self.min_questions]:
            question_text = template.format(country=country, prompt=prompt)
            rationale = f"Template-based question for {category} in {country}"
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

    def _fallback_question(self, country: str) -> CulturalQuestion:
        """Generate a fallback question."""
        return CulturalQuestion(
            question=f"Does this image appropriately represent {country} cultural elements?",
            expected_answer="yes",
            rationale="Generic fallback question for cultural appropriateness."
        )


class SimpleVLMClient:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        debug: bool = False,
    ) -> None:
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
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", **model_kwargs)
        self.model.eval()
        self.device = device
        self.debug = debug

    def evaluate_cultural_scores(
        self, 
        image_path: Path, 
        prompt: str, 
        context: str,
        country: str
    ) -> Tuple[int, int]:  # (cultural_representative, prompt_alignment)
        """Evaluate Cultural Representative (1-5) and Prompt Alignment (1-5) scores."""
        try:
            # Use demonym function
            dem = demonym(country)
            eval_question = f"""Rate this image on two aspects (1-5 scale):

Cultural Representative: How well does this image represent {dem} cultural elements? (1=Very Poor, 5=Very Good)
Prompt Alignment: How well does this image match this prompt? (1=Very Poor, 5=Very Good)

Original Prompt: "{prompt}"

Respond with exactly: Cultural:X,Prompt:Y (where X,Y are numbers 1-5)
Example: Cultural:4,Prompt:3"""

            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "system", "content": "You are an expert evaluator. Respond with format Cultural:X,Prompt:Y where X,Y are 1-5."},
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
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Extract only the new tokens (response)
            response_ids = generate_ids[0][len(inputs['input_ids'][0]):]
            output = self.processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            if self.debug:
                print(f"[DEBUG] Cultural scoring raw response: '{output}'")
                numbers_found = re.findall(r'\d+', output)
                print(f"[DEBUG] Numbers found: {numbers_found}")
            
            # Parse scores - try multiple formats
            cultural_score = 3  # default
            prompt_score = 3    # default
            
            try:
                # Method 1: Look for Cultural:X,Prompt:Y format
                cultural_match = re.search(r'Cultural:(\d+)', output, re.IGNORECASE)
                prompt_match = re.search(r'Prompt:(\d+)', output, re.IGNORECASE)
                
                if cultural_match and prompt_match:
                    cultural_score = int(cultural_match.group(1))
                    prompt_score = int(prompt_match.group(1))
                # Method 2: Look for any two numbers
                elif len(numbers_found) >= 2:
                    cultural_score = int(numbers_found[0])
                    prompt_score = int(numbers_found[1])
                # Method 3: Look for comma-separated numbers
                elif ',' in output:
                    parts = output.split(',')
                    if len(parts) >= 2:
                        cultural_nums = re.findall(r'\d+', parts[0])
                        prompt_nums = re.findall(r'\d+', parts[1])
                        if cultural_nums and prompt_nums:
                            cultural_score = int(cultural_nums[0])
                            prompt_score = int(prompt_nums[0])
                
                # Clamp to valid range
                cultural_score = max(1, min(5, cultural_score))
                prompt_score = max(1, min(5, prompt_score))
                
            except (ValueError, IndexError) as e:
                if self.debug:
                    print(f"[DEBUG] Score parsing failed: {e}, output was: '{output}', using defaults")
            
            if self.debug:
                print(f"[DEBUG] Final scores: cultural={cultural_score}, prompt={prompt_score}")
            
            return cultural_score, prompt_score
            
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Cultural scoring failed: {e}")
            return 3, 3  # default neutral scores

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


class SimpleCulturalMetric:
    """Simplified cultural metric evaluator."""
    
    def __init__(
        self,
        index_dir: Path,
        question_model: str = "openai/gpt-oss-20b",
        vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        debug: bool = False,
    ):
        self.kb = SimpleCulturalKnowledgeBase(index_dir)
        self.question_gen = SimpleQuestionGenerator(
            question_model,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            debug=debug,
        )
        self.vlm = SimpleVLMClient(
            vlm_model,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            debug=debug,
        )
        self.debug = debug

    def evaluate(
        self,
        image_path: Path,
        caption: str,
        country: str,
        category: Optional[str] = None,
        max_questions: int = 8,
        top_k: int = 8,
    ) -> CulturalScore:
        """
        Evaluate a single image for cultural appropriateness.
        
        Args:
            image_path: Path to the image file
            caption: Text description of the image
            country: Country to evaluate against
            category: Optional category (architecture, art, food, etc.)
            max_questions: Maximum number of questions to generate
            top_k: Number of context documents to retrieve
            
        Returns:
            CulturalScore with accuracy, precision, recall, f1, and cultural scores
        """
        start_time = time.time()
        
        if self.debug:
            print(f"[DEBUG] Evaluating image: {image_path}")
            print(f"[DEBUG] Caption: {caption}")
            print(f"[DEBUG] Country: {country}, Category: {category}")
        
        # Auto-detect category if not provided
        if category is None:
            category = extract_category_from_prompt(caption)
            if self.debug:
                print(f"[DEBUG] Auto-detected category: {category}")
        
        # Retrieve cultural context
        docs = self.kb.retrieve_contextual(caption, country, category, top_k)
        context_text = "\n".join(f"[Doc {idx}] {doc.text}" for idx, doc in enumerate(docs, 1))
        
        if self.debug:
            print(f"[DEBUG] Retrieved {len(docs)} context documents")
        
        # Generate questions
        questions, source = self.question_gen.generate(caption, country, category, docs, max_questions)
        
        if self.debug:
            print(f"[DEBUG] Generated {len(questions)} questions (source={source})")
            for i, q in enumerate(questions):
                print(f"[DEBUG]   Q{i+1}: {q.question} (expected: {q.expected_answer})")
        
        # Get VLM answers
        answers = []
        for q in questions:
            answer = self.vlm.answer(image_path, q.question, context_text)
            answers.append(answer)
            if self.debug:
                print(f"[DEBUG]   A{i+1}: {answer}")
        
        # Calculate metrics
        accuracy, precision, recall, f1 = calculate_metrics(questions, answers)
        
        # Evaluate cultural scores
        cultural_score, prompt_score = self.vlm.evaluate_cultural_scores(
            image_path, caption, context_text, country
        )
        
        processing_time = time.time() - start_time
        
        if self.debug:
            print(f"[DEBUG] Final scores: acc={accuracy:.3f}, prec={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
            print(f"[DEBUG] Cultural: {cultural_score}, Prompt: {prompt_score}")
            print(f"[DEBUG] Processing time: {processing_time:.2f}s")
        
        return CulturalScore(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cultural_representative=cultural_score,
            prompt_alignment=prompt_score,
            num_questions=len(questions),
            processing_time=processing_time,
        )


def main():
    """Simple command line interface."""
    parser = argparse.ArgumentParser(description="Simple cultural metric evaluation")
    parser.add_argument("--image", type=Path, required=True, help="Path to image file")
    parser.add_argument("--caption", type=str, required=True, help="Image caption/description")
    parser.add_argument("--country", type=str, required=True, help="Country to evaluate against")
    parser.add_argument("--category", type=str, default=None, help="Category (architecture, art, food, etc.)")
    parser.add_argument("--index-dir", type=Path, default=Path("./vector_store"), help="Path to vector store")
    parser.add_argument("--question-model", default="openai/gpt-oss-20b", help="Question generation model")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct", help="Vision-language model")
    parser.add_argument("--max-questions", type=int, default=8, help="Maximum number of questions")
    parser.add_argument("--top-k", type=int, default=8, help="Number of context documents")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load models in 8-bit")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load models in 4-bit")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SimpleCulturalMetric(
        index_dir=args.index_dir,
        question_model=args.question_model,
        vlm_model=args.vlm_model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        debug=args.debug,
    )
    
    # Evaluate
    result = evaluator.evaluate(
        image_path=args.image,
        caption=args.caption,
        country=args.country,
        category=args.category,
        max_questions=args.max_questions,
        top_k=args.top_k,
    )
    
    # Print results
    print("\n" + "="*50)
    print("CULTURAL METRIC EVALUATION RESULTS")
    print("="*50)
    print(f"Image: {args.image}")
    print(f"Caption: {args.caption}")
    print(f"Country: {args.country}")
    print(f"Category: {args.category or 'auto-detected'}")
    print()
    print("QUESTION-ANSWERING METRICS:")
    print(f"  Accuracy:  {result.accuracy:.3f}")
    print(f"  Precision: {result.precision:.3f}")
    print(f"  Recall:    {result.recall:.3f}")
    print(f"  F1 Score:  {result.f1:.3f}")
    print(f"  Questions: {result.num_questions}")
    print()
    print("CULTURAL SCORES (1-5 scale):")
    print(f"  Cultural Representative: {result.cultural_representative}/5")
    print(f"  Prompt Alignment:        {result.prompt_alignment}/5")
    print()
    print(f"Processing Time: {result.processing_time:.2f}s")
    print("="*50)


if __name__ == "__main__":
    main()
