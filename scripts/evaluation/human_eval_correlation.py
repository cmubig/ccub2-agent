#!/usr/bin/env python3
"""
Human Evaluation Correlation Study

Computes Spearman correlation between CultScore metric predictions
and human judgments. Target: >= 0.7.

Usage:
    # Generate metric scores for sample images (to be paired with human scores later)
    python scripts/evaluation/human_eval_correlation.py \
        --country korea --sample-size 100 --generate-scores

    # Compute correlation after collecting human scores
    python scripts/evaluation/human_eval_correlation.py \
        --country korea --human-scores experiments/human_scores_korea.json

    # Full pipeline: generate template, collect scores, then compute
    python scripts/evaluation/human_eval_correlation.py \
        --country korea --sample-size 50 --generate-template
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("human_eval")

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def load_dataset(country: str) -> List[Dict]:
    path = DATA_DIR / "country_packs" / country / "approved_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_path(country: str, item_id: str) -> Optional[Path]:
    images_dir = DATA_DIR / "country_packs" / country / "images"
    for ext in [".jpg", ".jpeg", ".png"]:
        p = images_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    return None


def generate_metric_scores(country: str, dataset: List[Dict]) -> List[Dict]:
    """Run VLM judge on sample images to get metric predictions."""
    from ccub2_agent.agents.base_agent import AgentConfig
    from ccub2_agent.agents.core.judge_agent import JudgeAgent
    from ccub2_agent.detection.vlm_detector import VLMCulturalDetector

    clip_index_dir = DATA_DIR / "clip_index" / country
    cultural_index_dir = DATA_DIR / "cultural_index" / country

    vlm = VLMCulturalDetector(
        load_in_4bit=True,
        index_dir=cultural_index_dir if cultural_index_dir.exists() else None,
        clip_index_dir=clip_index_dir if clip_index_dir.exists() else None,
    )

    config = AgentConfig(country=country)
    judge = JudgeAgent(config, shared_vlm_detector=vlm)

    scores = []
    for idx, item in enumerate(dataset):
        item_id = item["__id__"]
        category = item.get("category_normalized", "uncategorized")
        img_path = get_image_path(country, item_id)

        if img_path is None:
            continue

        logger.info(f"[{idx+1}/{len(dataset)}] Scoring {item_id}")

        result = judge.execute({
            "image_path": str(img_path),
            "prompt": item.get("description", f"Cultural image from {country}"),
            "country": country,
            "category": category,
        })

        if result.success:
            scores.append({
                "image_id": item_id,
                "category": category,
                "metric_cultural_score": result.data.get("cultural_score", 0),
                "metric_prompt_score": result.data.get("prompt_score", 0),
                "metric_failure_modes": result.data.get("failure_modes", []),
            })

    return scores


def generate_human_template(
    country: str,
    dataset: List[Dict],
    output_path: Path,
):
    """Generate a JSON template for human annotators to fill in."""
    template = {
        "study_info": {
            "country": country,
            "n_images": len(dataset),
            "instructions": (
                "For each image, provide: "
                "cultural_score (1-10, how culturally authentic), "
                "prompt_score (1-10, how well it matches the description), "
                "failure_modes (list of issues like 'wrong_pattern', 'anachronistic', etc.), "
                "comments (optional free text)."
            ),
            "categories": [
                "material_error", "symbolic_error", "contextual_error",
                "temporal_error", "performative_error", "compositional_error",
            ],
        },
        "images": [],
    }

    for item in dataset:
        item_id = item["__id__"]
        img_path = get_image_path(country, item_id)
        if img_path is None:
            continue
        template["images"].append({
            "image_id": item_id,
            "category": item.get("category_normalized", "uncategorized"),
            "description": item.get("description", ""),
            "image_path": str(img_path),
            # Human annotator fills these:
            "human_cultural_score": None,
            "human_prompt_score": None,
            "human_failure_modes": [],
            "human_comments": "",
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    logger.info(f"Human eval template saved: {output_path} ({len(template['images'])} images)")
    return output_path


def compute_correlation(
    metric_scores: List[Dict],
    human_scores_path: Path,
    output_dir: Path,
) -> Dict:
    """Compute Spearman/Pearson correlation between metric and human scores."""
    from ccub2_agent.evaluation.metrics.cultural_metric.calibration.human_validation import (
        HumanValidationProtocol,
    )

    # Load human scores
    with open(human_scores_path, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    protocol = HumanValidationProtocol(output_dir)

    # Add human judgments
    images = human_data.get("images", human_data.get("judgments", []))
    for img in images:
        image_id = img.get("image_id", "")
        h_score = img.get("human_cultural_score")
        p_score = img.get("human_prompt_score", 0)
        if h_score is None:
            continue
        protocol.add_human_judgment(
            image_id=image_id,
            judge_id=img.get("judge_id", "human_1"),
            cultural_score=float(h_score),
            prompt_score=float(p_score),
            failure_modes=img.get("human_failure_modes", []),
            comments=img.get("human_comments", ""),
        )

    # Add metric predictions
    for ms in metric_scores:
        protocol.add_metric_prediction(
            image_id=ms["image_id"],
            cultural_score=ms["metric_cultural_score"],
            prompt_score=ms["metric_prompt_score"],
            failure_modes=ms.get("metric_failure_modes", []),
        )

    # Compute correlation
    correlation = protocol.compute_correlation()
    protocol.save_results()

    logger.info("=" * 60)
    logger.info("HUMAN EVAL CORRELATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Matched pairs: {correlation['num_pairs']}")
    if correlation.get("cultural_spearman") is not None:
        logger.info(f"Cultural Spearman rho: {correlation['cultural_spearman']:.4f} (p={correlation['cultural_spearman_p']:.4f})")
        logger.info(f"Cultural Pearson r:    {correlation['cultural_pearson']:.4f} (p={correlation['cultural_pearson_p']:.4f})")
        target_met = correlation['cultural_spearman'] >= 0.7
        logger.info(f"Target (Spearman >= 0.7): {'MET' if target_met else 'NOT MET'}")
    else:
        logger.warning("Not enough matched pairs for correlation")

    return correlation


def main():
    parser = argparse.ArgumentParser(description="Human Eval Correlation Study")
    parser.add_argument("--country", type=str, default="korea")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-scores", action="store_true", help="Generate metric scores only")
    parser.add_argument("--generate-template", action="store_true", help="Generate human annotation template")
    parser.add_argument("--human-scores", type=str, default=None, help="Path to human scores JSON")
    args = parser.parse_args()

    output_dir = EXPERIMENTS_DIR / f"human_eval_{args.country}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and sample dataset
    dataset = load_dataset(args.country)
    if args.sample_size < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.sample_size)

    logger.info(f"Human eval study: country={args.country}, n={len(dataset)}")

    if args.generate_template:
        template_path = output_dir / f"human_eval_template_{args.country}.json"
        generate_human_template(args.country, dataset, template_path)
        return

    # Generate metric scores
    logger.info("Generating metric scores...")
    metric_scores = generate_metric_scores(args.country, dataset)

    # Save metric scores
    metric_path = output_dir / "metric_scores.json"
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metric_scores, f, indent=2, ensure_ascii=False)
    logger.info(f"Metric scores saved: {metric_path} ({len(metric_scores)} images)")

    if args.generate_scores:
        # Also generate template
        template_path = output_dir / f"human_eval_template_{args.country}.json"
        generate_human_template(args.country, dataset, template_path)
        logger.info("Metric scores generated. Collect human scores, then re-run with --human-scores.")
        return

    # Compute correlation
    if args.human_scores:
        human_path = Path(args.human_scores)
        if not human_path.exists():
            logger.error(f"Human scores file not found: {human_path}")
            sys.exit(1)
        correlation = compute_correlation(metric_scores, human_path, output_dir)

        # Save final results
        final = {
            "experiment": "human_eval_correlation",
            "country": args.country,
            "n_images": len(metric_scores),
            "correlation": correlation,
            "target_spearman": 0.7,
            "target_met": (correlation.get("cultural_spearman") or 0) >= 0.7,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        final_path = output_dir / "human_eval_results.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        logger.info(f"Final results: {final_path}")
    else:
        logger.info("No human scores provided. Generate template with --generate-template or --generate-scores, "
                     "collect human annotations, then re-run with --human-scores <path>.")


if __name__ == "__main__":
    main()
