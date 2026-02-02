#!/usr/bin/env python3
"""
Ablation Study Runner for NeurIPS Paper

Runs 4 ablation variants on Korea images:
    1. NO_CORRECTION:    Baseline — no cultural correction
    2. RETRIEVAL_ONLY:   Detection + Retrieval, no editing
    3. SINGLE_AGENT:     Single iteration (detect → retrieve → edit, no loop)
    4. MULTI_AGENT_LOOP: Full multi-agent iterative loop

Usage:
    # Full ablation on 20 sample images
    python scripts/experiments/run_ablation_study.py --country korea --sample-size 20

    # Specific variants only
    python scripts/experiments/run_ablation_study.py --country korea --sample-size 10 \
        --variants no_correction,multi_agent_loop

    # All variants on specific category
    python scripts/experiments/run_ablation_study.py --country korea --category food_drink --sample-size 15
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ablation")

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def load_dataset(country: str, category: Optional[str] = None) -> List[Dict]:
    path = DATA_DIR / "country_packs" / country / "approved_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if category:
        data = [d for d in data if d.get("category_normalized", "") == category]
    return data


def get_image_path(country: str, item_id: str) -> Optional[Path]:
    images_dir = DATA_DIR / "country_packs" / country / "images"
    for ext in [".jpg", ".jpeg", ".png"]:
        p = images_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    return None


def run_no_correction(country: str, dataset: List[Dict]) -> List[Dict]:
    """Baseline: Just score the original image, no correction."""
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

    results = []
    for idx, item in enumerate(dataset):
        item_id = item["__id__"]
        category = item.get("category_normalized", "uncategorized")
        img_path = get_image_path(country, item_id)
        if img_path is None:
            continue

        logger.info(f"[NO_CORRECTION][{idx+1}/{len(dataset)}] {item_id}")
        result = judge.execute({
            "image_path": str(img_path),
            "prompt": item.get("description", ""),
            "country": country,
            "category": category,
        })

        if result.success:
            score = result.data.get("cultural_score", 0)
            results.append({
                "item_id": item_id,
                "category": category,
                "initial_score": score,
                "final_score": score,
                "improvement": 0,
                "iterations": 0,
            })

    return results


def run_single_agent(country: str, dataset: List[Dict]) -> List[Dict]:
    """Single iteration: detect → retrieve → edit once."""
    from ccub2_agent.agents.base_agent import AgentConfig
    from ccub2_agent.agents.core.orchestrator_agent import OrchestratorAgent

    output_dir = EXPERIMENTS_DIR / "ablation_single_agent"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AgentConfig(country=country, output_dir=output_dir)
    orchestrator = OrchestratorAgent(config)

    results = []
    for idx, item in enumerate(dataset):
        item_id = item["__id__"]
        category = item.get("category_normalized", "uncategorized")
        img_path = get_image_path(country, item_id)
        if img_path is None:
            continue

        logger.info(f"[SINGLE_AGENT][{idx+1}/{len(dataset)}] {item_id}")
        result = orchestrator.execute({
            "image_path": str(img_path),
            "prompt": item.get("description", ""),
            "country": country,
            "category": category,
            "max_iterations": 1,
            "score_threshold": 10.0,
        })

        if result.success:
            sh = result.data.get("score_history", [])
            results.append({
                "item_id": item_id,
                "category": category,
                "initial_score": sh[0] if sh else 0,
                "final_score": result.data.get("final_score", 0),
                "improvement": result.data.get("improvement", 0),
                "iterations": 1,
                "score_history": sh,
            })

    return results


def run_multi_agent_loop(country: str, dataset: List[Dict], max_iters: int = 3) -> List[Dict]:
    """Full multi-agent iterative loop."""
    from ccub2_agent.agents.base_agent import AgentConfig
    from ccub2_agent.agents.core.orchestrator_agent import OrchestratorAgent

    output_dir = EXPERIMENTS_DIR / "ablation_multi_agent"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AgentConfig(country=country, output_dir=output_dir)
    orchestrator = OrchestratorAgent(config)

    results = []
    for idx, item in enumerate(dataset):
        item_id = item["__id__"]
        category = item.get("category_normalized", "uncategorized")
        img_path = get_image_path(country, item_id)
        if img_path is None:
            continue

        logger.info(f"[MULTI_AGENT_LOOP][{idx+1}/{len(dataset)}] {item_id}")
        result = orchestrator.execute({
            "image_path": str(img_path),
            "prompt": item.get("description", ""),
            "country": country,
            "category": category,
            "max_iterations": max_iters,
            "score_threshold": 7.0,
        })

        if result.success:
            sh = result.data.get("score_history", [])
            results.append({
                "item_id": item_id,
                "category": category,
                "initial_score": sh[0] if sh else 0,
                "final_score": result.data.get("final_score", 0),
                "improvement": result.data.get("improvement", 0),
                "iterations": result.data.get("iterations", 0),
                "score_history": sh,
            })

    return results


def summarize_variant(results: List[Dict]) -> Dict:
    """Compute summary stats for a variant."""
    if not results:
        return {"n": 0}

    improvements = [r["improvement"] for r in results]
    initial_scores = [r["initial_score"] for r in results]
    final_scores = [r["final_score"] for r in results]
    positive = [i for i in improvements if i > 0]

    return {
        "n": len(results),
        "avg_improvement": round(sum(improvements) / len(improvements), 3),
        "avg_initial_score": round(sum(initial_scores) / len(initial_scores), 3),
        "avg_final_score": round(sum(final_scores) / len(final_scores), 3),
        "positive_rate": f"{len(positive)}/{len(results)}",
        "positive_pct": round(len(positive) / len(results) * 100, 1),
        "max_improvement": max(improvements),
        "min_improvement": min(improvements),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument("--country", type=str, default="korea")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated: no_correction,single_agent,multi_agent_loop")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = EXPERIMENTS_DIR / f"ablation_{args.country}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse variants
    all_variants = ["no_correction", "single_agent", "multi_agent_loop"]
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]
    else:
        variants = all_variants

    # Load dataset
    dataset = load_dataset(args.country, args.category)
    if args.sample_size < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.sample_size)

    logger.info("=" * 60)
    logger.info("ABLATION STUDY")
    logger.info("=" * 60)
    logger.info(f"Country: {args.country}")
    logger.info(f"Variants: {variants}")
    logger.info(f"Sample size: {len(dataset)}")
    logger.info(f"Category: {args.category or 'ALL'}")

    all_results = {}
    summaries = {}

    for variant in variants:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running variant: {variant}")
        logger.info(f"{'='*40}")

        start_t = time.time()

        if variant == "no_correction":
            results = run_no_correction(args.country, dataset)
        elif variant == "single_agent":
            results = run_single_agent(args.country, dataset)
        elif variant == "multi_agent_loop":
            results = run_multi_agent_loop(args.country, dataset, args.max_iterations)
        else:
            logger.warning(f"Unknown variant: {variant}")
            continue

        elapsed = time.time() - start_t
        summary = summarize_variant(results)
        summary["elapsed_sec"] = round(elapsed, 1)

        all_results[variant] = results
        summaries[variant] = summary

        logger.info(f"Variant {variant}: avg_imp={summary.get('avg_improvement', 0):+.3f}, "
                     f"avg_final={summary.get('avg_final_score', 0):.3f}, "
                     f"n={summary['n']}")

    # Save results
    ablation_output = {
        "experiment": "ablation_study",
        "country": args.country,
        "category": args.category,
        "sample_size": len(dataset),
        "max_iterations": args.max_iterations,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summaries": summaries,
        "per_variant_results": all_results,
    }

    output_file = output_dir / "ablation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ablation_output, f, indent=2, ensure_ascii=False)

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION COMPARISON TABLE")
    logger.info("=" * 60)
    logger.info(f"{'Variant':25s} | {'N':>4s} | {'Avg Imp':>8s} | {'Avg Final':>10s} | {'Pos Rate':>10s} | {'Time':>8s}")
    logger.info("-" * 80)
    for variant in variants:
        s = summaries.get(variant, {})
        if s.get("n", 0) == 0:
            continue
        logger.info(
            f"{variant:25s} | {s['n']:4d} | {s['avg_improvement']:+8.3f} | "
            f"{s['avg_final_score']:10.3f} | {s['positive_rate']:>10s} | "
            f"{s['elapsed_sec']:7.0f}s"
        )

    logger.info(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    main()
