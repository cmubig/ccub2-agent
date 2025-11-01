#!/usr/bin/env python3
"""
Test VLM detector with Korea cultural index.

Usage:
    python scripts/test_vlm_detector.py --image-path <path> --prompt "korean hanbok"
"""

import argparse
from pathlib import Path
import sys
import logging

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.vlm_detector import create_vlm_detector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_detector(
    image_path: Path,
    prompt: str,
    country: str = "korea",
    category: str = None,
    use_index: bool = True,
):
    """
    Test VLM detector on an image.

    Args:
        image_path: Path to test image
        prompt: Generation prompt
        country: Target country
        category: Optional category
        use_index: Whether to use cultural index
    """
    logger.info("="*60)
    logger.info("VLM Cultural Detector Test")
    logger.info("="*60)
    logger.info(f"Image: {image_path}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Country: {country}")
    logger.info(f"Category: {category or 'None'}")
    logger.info("")

    # Determine index paths
    index_dir = None
    clip_index_dir = None

    if use_index:
        # Text-based cultural index
        index_dir = PROJECT_ROOT / "data" / "cultural_index" / country
        if not index_dir.exists():
            logger.warning(f"Text index not found at {index_dir}")
            index_dir = None
        else:
            logger.info(f"Using text cultural index: {index_dir}")

        # CLIP image index
        clip_index_dir = PROJECT_ROOT / "data" / "clip_index" / country
        if not clip_index_dir.exists():
            logger.warning(f"CLIP index not found at {clip_index_dir}")
            clip_index_dir = None
        else:
            logger.info(f"Using CLIP image index: {clip_index_dir}")

    # Create detector
    logger.info("Initializing VLM detector...")
    detector = create_vlm_detector(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        index_dir=index_dir,
        clip_index_dir=clip_index_dir,
        load_in_4bit=True,
        debug=True,
    )

    # Detect issues
    logger.info("\nDetecting cultural issues...")
    issues = detector.detect(
        image_path=image_path,
        prompt=prompt,
        country=country,
        editing_prompt=None,
        category=category,
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info("DETECTION RESULTS")
    logger.info("="*60)

    if not issues:
        logger.info("✓ No issues detected! Image appears culturally accurate.")
    else:
        logger.info(f"⚠ Found {len(issues)} issues:\n")
        for i, issue in enumerate(issues, 1):
            logger.info(f"{i}. [{issue['type'].upper()}] {issue['category']}")
            logger.info(f"   {issue['description']}")
            logger.info(f"   Severity: {issue['severity']}/10\n")

    # Get cultural quality scores
    logger.info("="*60)
    logger.info("QUALITY SCORES")
    logger.info("="*60)

    cultural, prompt_align = detector.score_cultural_quality(
        image_path=image_path,
        prompt=prompt,
        country=country,
    )

    logger.info(f"Cultural Representation: {cultural}/5")
    logger.info(f"Prompt Alignment: {prompt_align}/5")

    # Overall assessment
    logger.info("\n" + "="*60)
    logger.info("OVERALL ASSESSMENT")
    logger.info("="*60)

    if cultural >= 4 and prompt_align >= 4 and len(issues) == 0:
        logger.info("✓ EXCELLENT - High cultural accuracy and prompt alignment")
    elif cultural >= 3 and prompt_align >= 3 and len(issues) <= 1:
        logger.info("○ GOOD - Generally accurate with minor issues")
    elif cultural >= 2 and prompt_align >= 2:
        logger.info("△ FAIR - Some cultural inaccuracies detected")
    else:
        logger.info("✗ POOR - Significant cultural problems detected")

    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Test VLM cultural detector")
    parser.add_argument(
        '--image-path',
        type=Path,
        required=True,
        help='Path to test image'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Generation prompt'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Target country (e.g., korea, japan, china)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Image category (food, traditional_clothing, etc.)'
    )
    parser.add_argument(
        '--no-index',
        action='store_true',
        help='Run without cultural index'
    )

    args = parser.parse_args()

    if not args.image_path.exists():
        logger.error(f"Image not found: {args.image_path}")
        sys.exit(1)

    test_detector(
        image_path=args.image_path,
        prompt=args.prompt,
        country=args.country,
        category=args.category,
        use_index=not args.no_index,
    )


if __name__ == "__main__":
    main()
