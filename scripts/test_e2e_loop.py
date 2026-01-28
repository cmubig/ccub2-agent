#!/usr/bin/env python3
"""
End-to-End Test for WorldCCUB Multi-Agent Loop

This script validates that the full OrchestratorAgent pipeline works:
Judge → Scout (with CLIP RAG) → Verification → Edit → Judge (re-eval)

Usage:
    python scripts/test_e2e_loop.py --image path/to/test.jpg --country korea
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.agents.core import OrchestratorAgent
from ccub2_agent.agents.base_agent import AgentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_e2e_test(
    image_path: str,
    country: str = "korea",
    category: str = "traditional_clothing",
    prompt: str = None,
    max_iterations: int = 3,
    score_threshold: float = 8.0,
    output_dir: str = None,
) -> dict:
    """
    Run end-to-end multi-agent loop test.

    Args:
        image_path: Path to input test image
        country: Target country (e.g., korea, china, japan)
        category: Cultural category
        prompt: Optional prompt describing the image
        max_iterations: Maximum loop iterations
        score_threshold: Target cultural score
        output_dir: Directory to save results

    Returns:
        dict with test results
    """
    logger.info("=" * 60)
    logger.info("WorldCCUB E2E Multi-Agent Loop Test")
    logger.info("=" * 60)

    # Validate input
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = PROJECT_ROOT / "outputs" / "e2e_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default prompt if not provided
    if prompt is None:
        prompt = f"A cultural image from {country} in the {category} category"

    logger.info(f"Input image: {image_path}")
    logger.info(f"Country: {country}")
    logger.info(f"Category: {category}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Score threshold: {score_threshold}")

    # Initialize OrchestratorAgent
    logger.info("\n[1/2] Initializing OrchestratorAgent...")
    config = AgentConfig(country=country, category=category)
    orchestrator = OrchestratorAgent(config)

    # Prepare input
    input_data = {
        "image_path": str(image_path),
        "prompt": prompt,
        "country": country,
        "category": category,
        "max_iterations": max_iterations,
        "score_threshold": score_threshold,
    }

    # Execute loop
    logger.info("\n[2/2] Executing Multi-Agent Loop...")
    logger.info("-" * 40)

    result = orchestrator.execute(input_data)

    logger.info("-" * 40)

    # Process results
    test_result = {
        "success": result.success,
        "message": result.message,
        "input": {
            "image_path": str(image_path),
            "country": country,
            "category": category,
            "prompt": prompt,
        },
        "output": result.data,
        "timestamp": datetime.now().isoformat(),
    }

    # Log results
    if result.success:
        logger.info("\n" + "=" * 60)
        logger.info("TEST PASSED")
        logger.info("=" * 60)

        score_history = result.data.get("score_history", [])
        improvement = result.data.get("improvement", 0)
        iterations = result.data.get("iterations", 0)
        final_score = result.data.get("final_score", 0)

        logger.info(f"Iterations: {iterations}")
        logger.info(f"Score history: {score_history}")
        logger.info(f"Final score: {final_score:.2f}")
        logger.info(f"Improvement: {improvement:+.2f}")

        if improvement > 0:
            logger.info(f"Cultural score IMPROVED by {improvement:.2f} points!")
        elif improvement == 0:
            logger.info("Score unchanged (may have reached threshold on first try)")
        else:
            logger.warning(f"Score decreased by {abs(improvement):.2f} points")

    else:
        logger.error("\n" + "=" * 60)
        logger.error("TEST FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {result.message}")

    # Save results
    result_file = output_dir / "test_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to: {result_file}")

    return test_result


def main():
    parser = argparse.ArgumentParser(
        description="Run WorldCCUB E2E Multi-Agent Loop Test"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input test image"
    )
    parser.add_argument(
        "--country", "-c",
        default="korea",
        help="Target country (default: korea)"
    )
    parser.add_argument(
        "--category",
        default="traditional_clothing",
        help="Cultural category (default: traditional_clothing)"
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Optional prompt describing the image"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum loop iterations (default: 3)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=8.0,
        help="Target cultural score (default: 8.0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()

    try:
        result = run_e2e_test(
            image_path=args.image,
            country=args.country,
            category=args.category,
            prompt=args.prompt,
            max_iterations=args.max_iterations,
            score_threshold=args.score_threshold,
            output_dir=args.output_dir,
        )

        sys.exit(0 if result["success"] else 1)

    except Exception as e:
        logger.exception(f"Test failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
