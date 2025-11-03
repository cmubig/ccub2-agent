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

    def extract_knowledge(
        self,
        image_path: Path,
        item_data: Dict,
        country: str,
    ) -> Optional[CulturalKnowledge]:
        """
        Extract cultural knowledge from a single image.

        Args:
            image_path: Path to the image
            item_data: Item metadata from approved_dataset.json
            country: Country name

        Returns:
            CulturalKnowledge object or None if extraction fails
        """
        try:
            category = item_data.get('category', 'general')
            item_id = item_data.get('id', 'unknown')
            quality_score = item_data.get('quality_score', 0)

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Build extraction prompt
            prompt = self._build_extraction_prompt(category, country, item_data)

            # VLM inference
            response = self._run_vlm(image, prompt)

            # Parse response
            knowledge = self._parse_response(
                response,
                item_id,
                category,
                country,
                quality_score
            )

            return knowledge

        except Exception as e:
            logger.error(f"Failed to extract knowledge from {image_path}: {e}")
            return None

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

    def _run_vlm(self, image: Image.Image, prompt: str) -> str:
        """Run VLM inference."""
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

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

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
                extraction_notes=''
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
        help="Path to country pack directory (e.g., ~/ccub2-agent-data/country_packs/korea)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for extracted knowledge (e.g., ~/ccub2-agent-data/cultural_knowledge/korea_knowledge.json)"
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
