"""
Extract cultural knowledge from high-quality images using Qwen3-VL-8B.

This script analyzes real cultural images and extracts structured knowledge
that will be used to enhance the RAG system for better cultural evaluation.

Unlike hardcoded CULTURAL_KNOWLEDGE dictionaries, this extracts knowledge
automatically from actual verified images.
"""

import argparse
import json
import logging
import os
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CulturalKnowledge:
    """Structured cultural knowledge extracted from an image."""

    item_id: str
    category: str
    country: str

    # Visual analysis
    visual_features: str  # What you see: structure, shapes, arrangement
    materials_textures: str  # Materials, fabrics, textures visible
    colors_patterns: str  # Color schemes and patterns

    # Cultural analysis
    cultural_elements: str  # Authentic cultural elements present
    correct_aspects: List[str]  # What this image does right culturally
    incorrect_aspects: List[str]  # Any cultural inaccuracies (if present)

    # Guidance for AI generation
    key_characteristics: str  # Essential characteristics to include
    common_mistakes: str  # Common mistakes to avoid

    # Metadata
    quality_score: int
    confidence: str  # high/medium/low
    extraction_notes: str


class CulturalKnowledgeExtractor:
    """Extract cultural knowledge from images using VLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit: bool = True,
        device: str = "cuda",
    ):
        """Initialize the VLM knowledge extractor."""
        logger.info(f"Loading VLM: {model_name}")

        model_kwargs = {"trust_remote_code": True}

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quant_config

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto",
            **model_kwargs
        )
        self.model.eval()
        self.device = device

        logger.info("VLM loaded successfully")

    def _preprocess_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Preprocess image to reduce memory usage.
        Resize to max_size x max_size while maintaining aspect ratio.
        Pad with white to make it square.

        Args:
            image: Input PIL Image
            max_size: Maximum dimension (default 1024)

        Returns:
            Preprocessed square PIL Image
        """
        # Get original size
        orig_width, orig_height = image.size

        # Calculate scaling factor to fit within max_size
        scale = min(max_size / orig_width, max_size / orig_height)

        # Only resize if image is larger than max_size
        if scale < 1.0:
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        else:
            new_width, new_height = orig_width, orig_height

        # Create square canvas with white background
        canvas = Image.new('RGB', (max_size, max_size), (255, 255, 255))

        # Paste resized image in center
        paste_x = (max_size - new_width) // 2
        paste_y = (max_size - new_height) // 2
        canvas.paste(image, (paste_x, paste_y))

        return canvas

    def extract_knowledge(
        self,
        image_path: Path,
        item_data: Dict,
        country: str,
        max_retries: int = 3,
    ) -> Optional[CulturalKnowledge]:
        """
        Extract cultural knowledge from a single image with retry logic.

        Args:
            image_path: Path to the image
            item_data: Item metadata from approved_dataset.json
            country: Country name
            max_retries: Maximum retry attempts

        Returns:
            CulturalKnowledge object or None if all attempts fail
        """
        category = item_data.get('category', 'general')
        item_id = item_data.get('id', 'unknown')
        quality_score = item_data.get('quality_score', 0)

        # Load and preprocess image once
        try:
            image = Image.open(image_path).convert("RGB")
            orig_size = image.size
            image = self._preprocess_image(image, max_size=1024)
            logger.debug(f"Preprocessed {item_id}: {orig_size} -> {image.size}")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

        # Retry strategies
        strategies = [
            {'temperature': 0.0, 'max_tokens': 1024, 'desc': 'deterministic'},
            {'temperature': 0.3, 'max_tokens': 1024, 'desc': 'low temperature'},
            {'temperature': 0.7, 'max_tokens': 1536, 'desc': 'higher temperature + more tokens'},
        ]

        last_error = None
        best_knowledge = None

        for attempt in range(max_retries):
            try:
                strategy = strategies[min(attempt, len(strategies) - 1)]
                logger.debug(f"Attempt {attempt + 1}/{max_retries} for {item_id} ({strategy['desc']})")

                # Build extraction prompt
                prompt = self._build_extraction_prompt(category, country, item_data)

                # VLM inference with strategy
                response = self._run_vlm(
                    image,
                    prompt,
                    temperature=strategy['temperature'],
                    max_tokens=strategy['max_tokens']
                )

                # Parse response
                knowledge = self._parse_response(
                    response,
                    item_id,
                    category,
                    country,
                    quality_score
                )

                if knowledge:
                    # Success - check quality
                    if not knowledge.extraction_notes:
                        # Perfect extraction
                        logger.debug(f"✓ {item_id} extracted successfully on attempt {attempt + 1}")
                        return knowledge
                    else:
                        # Partial extraction - keep trying but save as backup
                        logger.debug(f"Partial extraction for {item_id} on attempt {attempt + 1}")
                        if best_knowledge is None:
                            best_knowledge = knowledge

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM for {image_path} on attempt {attempt + 1}: {e}")
                last_error = e
                # Try to recover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import time
                time.sleep(2)  # Wait before retry

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {image_path}: {e}")
                last_error = e

        # All attempts exhausted
        if best_knowledge:
            logger.warning(f"Returning partial extraction for {item_id}")
            return best_knowledge
        else:
            logger.error(f"All {max_retries} attempts failed for {item_id}: {last_error}")

            # Last resort: create minimal knowledge from description
            if item_data.get('description'):
                logger.info(f"Creating minimal knowledge from description for {item_id}")
                return CulturalKnowledge(
                    item_id=item_id,
                    category=category,
                    country=country,
                    visual_features=item_data.get('description', ''),
                    materials_textures='',
                    colors_patterns='',
                    cultural_elements=f"Authentic {country} {category}",
                    correct_aspects=[],
                    incorrect_aspects=[],
                    key_characteristics='',
                    common_mistakes='',
                    quality_score=quality_score,
                    confidence='low',
                    extraction_notes=f'Failed VLM extraction - using description only. Error: {str(last_error)[:100]}'
                )

            return None

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_extraction_prompt(
        self,
        category: str,
        country: str,
        item_data: Dict,
    ) -> str:
        """Build VLM prompt for cultural knowledge extraction."""

        # Get description if available
        description = item_data.get('description', '')

        prompt = f"""You are a cultural expert analyzing an authentic {country} image in the category: {category}.

**Your task**: Extract detailed cultural knowledge from this image that will help AI systems understand what is culturally authentic.

Original description: "{description}"

Please analyze and provide:

1. **VISUAL FEATURES** (What you see):
   - Describe the structure, shapes, arrangement, composition
   - Be specific about physical characteristics
   - Example: "Two-piece garment: short jacket (jeogori) with curved sleeve seams ending at hip level, paired with a long skirt (chima) starting from chest level"

2. **MATERIALS & TEXTURES**:
   - What materials/fabrics are visible?
   - Texture characteristics (smooth, rough, flowing, etc.)
   - Example: "Silk fabric with visible sheen, flowing and layered rather than fitted"

3. **COLORS & PATTERNS**:
   - Color scheme and harmony
   - Patterns and decorative elements
   - Example: "Soft gray with white floral patterns, harmonious traditional color palette (not bright neon)"

4. **CULTURAL ELEMENTS** (Why it's authentic):
   - What makes this authentically {country}?
   - Cultural significance of elements
   - Example: "Authentic Korean hanbok structure with traditional silhouette and proportions"

5. **CORRECT ASPECTS** (List 3-5 things this image does RIGHT):
   - Specific authentic details
   - Example: ["Proper jeogori length at hip level", "High-waisted chima placement", "Traditional curved seam lines"]

6. **KEY CHARACTERISTICS** (Must-have for AI generation):
   - Essential features that MUST be present
   - Example: "Must have: two-piece structure, jeogori + chima, high waistline, flowing fabric, modest coverage"

7. **COMMON MISTAKES TO AVOID**:
   - What should AI models NOT do?
   - Example: "Avoid: Chinese mandarin collar, Japanese obi belt, tight Western silhouette, mixing elements from different cultures"

Respond in JSON format:
{{
  "visual_features": "...",
  "materials_textures": "...",
  "colors_patterns": "...",
  "cultural_elements": "...",
  "correct_aspects": ["...", "...", "..."],
  "key_characteristics": "...",
  "common_mistakes": "...",
  "confidence": "high/medium/low"
}}

Be specific and detailed. Focus on visual, verifiable characteristics."""

        return prompt

    def _run_vlm(self, image: Image.Image, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Run VLM inference with configurable parameters."""
        messages = [
            {
                "role": "system",
                "content": "You are a cultural analysis expert. Respond in JSON format."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # Configure generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }

        if temperature > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9,
            })
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        response_ids = outputs[0][len(inputs['input_ids'][0]):]
        response = self.processor.tokenizer.decode(
            response_ids,
            skip_special_tokens=True
        ).strip()

        return response

    def _parse_response(
        self,
        response: str,
        item_id: str,
        category: str,
        country: str,
        quality_score: int,
    ) -> Optional[CulturalKnowledge]:
        """Parse VLM response into CulturalKnowledge object."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in response for {item_id}")
                return None

            data = json.loads(json_match.group())

            # Check if essential fields are present
            essential_fields = ['visual_features', 'cultural_elements', 'key_characteristics']
            missing_fields = [f for f in essential_fields if not data.get(f)]

            if missing_fields:
                logger.warning(f"Missing essential fields for {item_id}: {missing_fields}")
                # Don't return None - create partial knowledge
                extraction_notes = f"Partial extraction - missing: {', '.join(missing_fields)}"
            else:
                extraction_notes = ''

            knowledge = CulturalKnowledge(
                item_id=item_id,
                category=category,
                country=country,
                visual_features=data.get('visual_features', ''),
                materials_textures=data.get('materials_textures', ''),
                colors_patterns=data.get('colors_patterns', ''),
                cultural_elements=data.get('cultural_elements', ''),
                correct_aspects=data.get('correct_aspects', []),
                incorrect_aspects=data.get('incorrect_aspects', []),
                key_characteristics=data.get('key_characteristics', ''),
                common_mistakes=data.get('common_mistakes', ''),
                quality_score=quality_score,
                confidence=data.get('confidence', 'medium'),
                extraction_notes=extraction_notes
            )

            return knowledge

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {item_id}: {e}")
            logger.debug(f"Response was: {response[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Parse error for {item_id}: {e}")
            return None


def process_worker(args_tuple):
    """
    Worker function for parallel processing.

    Args:
        args_tuple: (worker_id, items_chunk, data_dir, output_file, model_name, load_in_4bit, country, gpu_id)
    """
    worker_id, items_chunk, data_dir, output_file, model_name, load_in_4bit, country, gpu_id = args_tuple

    # Set GPU for this worker
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Worker {worker_id}: Using GPU {gpu_id}")

    # Initialize extractor for this worker
    extractor = CulturalKnowledgeExtractor(
        model_name=model_name,
        load_in_4bit=load_in_4bit
    )

    extracted_knowledge = []

    for item in tqdm(items_chunk, desc=f"Worker {worker_id}", position=worker_id):
        item_id = item['id']

        # Get image path
        image_path = data_dir / item['image_path']
        if not image_path.exists():
            logger.warning(f"Worker {worker_id}: Image not found: {image_path}")
            continue

        # Extract knowledge
        knowledge = extractor.extract_knowledge(
            image_path=image_path,
            item_data=item,
            country=country
        )

        if knowledge:
            extracted_knowledge.append(asdict(knowledge))

    # Save worker output
    output_data = {
        'worker_id': worker_id,
        'country': country,
        'extracted_count': len(extracted_knowledge),
        'knowledge': extracted_knowledge
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Worker {worker_id}: Completed {len(extracted_knowledge)} items -> {output_file}")
    return len(extracted_knowledge)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract cultural knowledge from images using VLM"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to country pack directory (e.g., PROJECT_ROOT/data/country_packs/korea)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for extracted knowledge (e.g., PROJECT_ROOT/data/cultural_knowledge/korea_knowledge.json)"
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="VLM model name"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for chunk processing (for parallel execution)"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for chunk processing (for parallel execution)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, recommend: 2-4)"
    )

    args = parser.parse_args()

    # Load dataset - prefer enhanced version with VLM captions
    dataset_file_enhanced = args.data_dir / "approved_dataset_enhanced.json"
    dataset_file = args.data_dir / "approved_dataset.json"

    if dataset_file_enhanced.exists():
        logger.info(f"Using enhanced dataset: {dataset_file_enhanced}")
        with open(dataset_file_enhanced) as f:
            dataset = json.load(f)
    elif dataset_file.exists():
        logger.info(f"Using original dataset: {dataset_file}")
        with open(dataset_file) as f:
            dataset = json.load(f)
    else:
        logger.error(f"Dataset not found: {dataset_file} or {dataset_file_enhanced}")
        return

    items = dataset['items']
    country = dataset_file.parent.name  # 'korea' from path

    # Apply chunk slicing first (for parallel processing)
    if args.start_index > 0 or args.end_index is not None:
        end_idx = args.end_index if args.end_index else len(items)
        items = items[args.start_index:end_idx]
        logger.info(f"Processing chunk: items [{args.start_index}:{end_idx}] ({len(items)} images)")

    if args.max_images:
        items = items[:args.max_images]
        logger.info(f"Limited to {args.max_images} images for testing")

    # Parallel processing with workers
    if args.num_workers > 1:
        logger.info(f"Using {args.num_workers} parallel workers")

        # Split items into chunks
        chunk_size = len(items) // args.num_workers
        chunks = []
        for i in range(args.num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < args.num_workers - 1 else len(items)
            chunks.append(items[start_idx:end_idx])

        # Prepare worker arguments
        worker_args = []
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPUs")

        for i, chunk in enumerate(chunks):
            gpu_id = i % num_gpus if num_gpus > 0 else None
            output_file = args.output.parent / f"{args.output.stem}_worker{i}.json"
            worker_args.append((
                i,  # worker_id
                chunk,  # items_chunk
                args.data_dir,  # data_dir
                output_file,  # output_file
                args.model_name,  # model_name
                args.load_in_4bit,  # load_in_4bit
                country,  # country
                gpu_id  # gpu_id
            ))

        # Run workers in parallel
        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.map(process_worker, worker_args)

        # Merge results
        logger.info("Merging worker outputs...")
        extracted_knowledge = []
        for i in range(args.num_workers):
            worker_file = args.output.parent / f"{args.output.stem}_worker{i}.json"
            if worker_file.exists():
                with open(worker_file) as f:
                    worker_data = json.load(f)
                    extracted_knowledge.extend(worker_data.get('knowledge', []))
                # Optionally remove worker files
                # worker_file.unlink()

        # Save merged output
        output_data = {
            'country': country,
            'total_items': len(items),
            'extracted_count': len(extracted_knowledge),
            'knowledge': extracted_knowledge
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Parallel processing complete: {len(extracted_knowledge)} items extracted")
        logger.info(f"Output saved to: {args.output}")
        return

    # Sequential processing (original logic)
    # Check for resume
    processed_ids = set()
    extracted_knowledge = []

    if args.resume and args.output.exists():
        logger.info(f"Resuming from {args.output}")
        with open(args.output) as f:
            existing_data = json.load(f)
            extracted_knowledge = existing_data.get('knowledge', [])
            processed_ids = {k['item_id'] for k in extracted_knowledge}
        logger.info(f"Found {len(processed_ids)} already processed items")

    # Initialize extractor
    extractor = CulturalKnowledgeExtractor(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit
    )

    # Process images
    logger.info(f"Processing {len(items)} images...")

    for item in tqdm(items, desc="Extracting knowledge"):
        item_id = item['id']

        # Skip if already processed
        if item_id in processed_ids:
            continue

        # Get image path
        image_path = args.data_dir / item['image_path']
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        # Extract knowledge
        knowledge = extractor.extract_knowledge(
            image_path=image_path,
            item_data=item,
            country=country
        )

        if knowledge:
            extracted_knowledge.append(asdict(knowledge))
            processed_ids.add(item_id)

            # Save checkpoint every 10 items
            if len(extracted_knowledge) % 10 == 0:
                output_data = {
                    'country': country,
                    'total_items': len(items),
                    'extracted_count': len(extracted_knowledge),
                    'knowledge': extracted_knowledge
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Checkpoint saved: {len(extracted_knowledge)} items")

    # Final save
    output_data = {
        'country': country,
        'total_items': len(items),
        'extracted_count': len(extracted_knowledge),
        'knowledge': extracted_knowledge
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Extraction complete!")
    logger.info(f"   Processed: {len(extracted_knowledge)}/{len(items)} images")
    logger.info(f"   Output: {args.output}")

    # Print sample
    if extracted_knowledge:
        logger.info("\n📄 Sample extracted knowledge:")
        sample = extracted_knowledge[0]
        logger.info(f"   ID: {sample['item_id']}")
        logger.info(f"   Category: {sample['category']}")
        logger.info(f"   Visual: {sample['visual_features'][:100]}...")
        logger.info(f"   Cultural: {sample['cultural_elements'][:100]}...")


if __name__ == "__main__":
    main()
