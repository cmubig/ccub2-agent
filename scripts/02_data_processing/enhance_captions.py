#!/usr/bin/env python3
"""
Enhance SNS-style captions to cultural descriptions using Qwen3-VL.

Converts:
"Beautiful hanbok at Changdeokgung Palace in the evening!"
→ "Traditional Korean hanbok worn at Changdeokgung Palace,
   featuring authentic design, colors, and structure"

Uses LOCAL Qwen3-VL-8B model (no API needed).
"""

import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import logging

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add metric module to path
METRIC_PATH = PROJECT_ROOT / "metric" / "cultural_metric"
if str(METRIC_PATH) not in sys.path:
    sys.path.insert(0, str(METRIC_PATH))

from enhanced_cultural_metric_pipeline import EnhancedVLMClient
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def enhance_caption_with_vlm(
    vlm: EnhancedVLMClient,
    image_path: Path,
    original_caption: str,
    category: str,
    country: str,
    lang: str = "ko",
) -> str:
    """
    Use Qwen3-VL to generate cultural description from image.

    Args:
        vlm: EnhancedVLMClient instance
        image_path: Path to image
        original_caption: Original SNS-style caption
        category: Image category
        country: Country name
        lang: Original language

    Returns:
        Enhanced cultural description in English
    """
    # Construct prompt for VLM
    prompt_text = f"""Look at this {country} {category} image. The original caption is: "{original_caption}" ({lang})

Describe what you see in the image in English (2-3 sentences):
1. Describe the visual details: what objects, people, colors, setting, and actions you see
2. Use Korean cultural terms with English translations when relevant (e.g., "hanbok (traditional Korean clothing)", "bulgogi (grilled beef)", "gunbam (roasted chestnuts)")
3. Be specific and descriptive about what is visible in the image
4. Focus on describing what you observe, not analyzing cultural significance

Output only the image description, no explanation or metadata."""

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Build messages
    messages = [
        {"role": "system", "content": "You are a cultural expert. Describe cultural elements in images accurately and informatively."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image},
            ],
        },
    ]

    # Generate response
    text_prompt = vlm.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm.processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(vlm.device)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    with torch.no_grad():
        generate_ids = vlm.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=vlm.processor.tokenizer.pad_token_id
        )

    # Extract only the new tokens (response)
    response_ids = generate_ids[0][len(inputs['input_ids'][0]):]
    response = vlm.processor.decode(response_ids, skip_special_tokens=True)

    return response.strip()


def enhance_dataset_captions(
    dataset_path: Path,
    images_dir: Path,
    output_path: Path,
    country: str,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_items: int = None,
):
    """
    Enhance all captions in the dataset using Qwen3-VL.

    Args:
        dataset_path: Path to approved_dataset.json
        images_dir: Directory containing images
        output_path: Output path for enhanced dataset
        country: Country name for context
        model_name: VLM model name
        max_items: Max items to process (None = all)
    """
    logger.info("="*70)
    logger.info("CAPTION ENHANCEMENT WITH QWEN3-VL")
    logger.info("="*70)
    logger.info(f"Input: {dataset_path}")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: {model_name}")
    logger.info("")

    # Load VLM
    logger.info("Loading Qwen3-VL model...")
    vlm = EnhancedVLMClient(
        model_name=model_name,
        load_in_4bit=True,
        debug=False,
    )
    logger.info("✓ Model loaded")
    logger.info("")

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = data['items']
    if max_items:
        items = items[:max_items]

    logger.info(f"Processing {len(items)} items...")
    logger.info("")

    # Enhance captions
    enhanced = 0
    failed = 0
    skipped = 0

    for item in tqdm(items, desc="Enhancing"):
        original_desc = item.get('description', '')
        category = item.get('category', 'general')
        lang = item.get('description_lang', 'ko')
        image_rel_path = item.get('image_path', '')

        # Skip if already enhanced (incremental update)
        if item.get('description_enhanced'):
            skipped += 1
            continue

        # Skip if no image path
        if not image_rel_path:
            skipped += 1
            continue

        # Construct absolute image path
        # image_rel_path is like "images/traditional_clothing/01vCHXPdeqCNMg5Sr5al.jpg"
        # Remove "images/" prefix if present
        if image_rel_path.startswith("images/"):
            image_rel_path = image_rel_path[len("images/"):]
        image_path = images_dir / image_rel_path
        if not image_path.exists():
            logger.warning(f"  Image not found: {image_path}")
            skipped += 1
            continue

        try:
            # Generate cultural description
            enhanced_desc = enhance_caption_with_vlm(
                vlm=vlm,
                image_path=image_path,
                original_caption=original_desc or f"{country.capitalize()} cultural image",
                category=category,
                country=country,
                lang=lang,
            )

            item['description_enhanced'] = enhanced_desc
            if original_desc:
                item['description_original'] = original_desc
            enhanced += 1

            # Log before/after
            logger.info(f"\n{'='*80}")
            logger.info(f"[{enhanced}] {category.upper()} - {image_path.name}")
            logger.info(f"\n❌ BEFORE: \"{original_desc}\"")
            logger.info(f"\n✅ AFTER: \"{enhanced_desc}\"")
            logger.info(f"{'='*80}\n")

        except Exception as e:
            logger.error(f"  Failed for {image_path.name}: {e}")
            failed += 1

    # Save enhanced dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("="*70)
    logger.info("ENHANCEMENT COMPLETE")
    logger.info("="*70)
    logger.info(f"Enhanced: {enhanced}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Output: {output_path}")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Enhance SNS captions with Qwen3-VL")
    parser.add_argument(
        '--dataset',
        type=Path,
        help='Dataset path (default: PROJECT_ROOT/data/country_packs/{country}/approved_dataset.json)'
    )
    parser.add_argument(
        '--images-dir',
        type=Path,
        help='Images directory (default: PROJECT_ROOT/data/country_packs/{country}/images)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path (default: PROJECT_ROOT/data/country_packs/{country}/approved_dataset_enhanced.json)'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., korea, japan, china)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-8B-Instruct',
        help='VLM model name'
    )
    parser.add_argument(
        '--max-items',
        type=int,
        default=None,
        help='Max items to process (for testing)'
    )

    args = parser.parse_args()

    # Set default paths if not provided
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    if args.dataset is None:
        args.dataset = PROJECT_ROOT / f"data/country_packs/{args.country}/approved_dataset.json"
    if args.images_dir is None:
        args.images_dir = PROJECT_ROOT / f"data/country_packs/{args.country}/images"
    if args.output is None:
        args.output = PROJECT_ROOT / f"data/country_packs/{args.country}/approved_dataset_enhanced.json"

    enhance_dataset_captions(
        dataset_path=args.dataset,
        images_dir=args.images_dir,
        country=args.country,
        output_path=args.output,
        model_name=args.model,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
