#!/usr/bin/env python3
"""
WorldCCUB Bulk Pipeline — Korea 575 Images

Runs the full agent loop on all (or sampled) images from a country,
collects per-image and per-category improvement statistics, and writes
structured results to experiments/.

Usage:
    # Full 575 images (skip download/indexing if already done)
    python scripts/run_bulk_pipeline.py --country korea --skip-download --skip-indexing

    # Category-specific run
    python scripts/run_bulk_pipeline.py --country korea --category food_drink --sample-size 20

    # Quick sanity check
    python scripts/run_bulk_pipeline.py --country korea --sample-size 10 --skip-download --skip-indexing
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"bulk_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("bulk_pipeline")

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def load_dataset(country: str) -> List[Dict]:
    """Load approved dataset for a country."""
    dataset_path = DATA_DIR / "country_packs" / country / "approved_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_by_category(dataset: List[Dict], category: Optional[str]) -> List[Dict]:
    """Filter dataset by normalized category."""
    if not category:
        return dataset
    return [d for d in dataset if d.get("category_normalized", "") == category]


def sample_dataset(dataset: List[Dict], sample_size: Optional[int], seed: int = 42) -> List[Dict]:
    """Sample from dataset if sample_size specified."""
    if sample_size and sample_size < len(dataset):
        random.seed(seed)
        return random.sample(dataset, sample_size)
    return dataset


def run_bulk_agent_loop(
    country: str,
    dataset: List[Dict],
    max_iterations: int = 3,
    score_threshold: float = 7.0,
) -> Dict:
    """Run agent loop on all images in dataset."""
    from ccub2_agent.agents.base_agent import AgentConfig
    from ccub2_agent.agents.core.orchestrator_agent import OrchestratorAgent

    images_dir = DATA_DIR / "country_packs" / country / "images"
    output_dir = EXPERIMENTS_DIR / f"bulk_{country}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing OrchestratorAgent for {country}...")
    config = AgentConfig(country=country, output_dir=output_dir)
    orchestrator = OrchestratorAgent(config)

    results = []
    category_stats = defaultdict(lambda: {
        "count": 0, "improvements": [], "initial_scores": [],
        "final_scores": [], "successes": 0, "failures": 0,
    })

    total = len(dataset)
    for idx, item in enumerate(dataset):
        item_id = item.get("__id__", "unknown")
        category = item.get("category_normalized", "uncategorized")
        prompt = item.get("description", f"Cultural image from {country}")

        # Find image file
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / f"{item_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            logger.warning(f"[{idx+1}/{total}] Image not found for {item_id}, skipping")
            category_stats[category]["failures"] += 1
            results.append({
                "item_id": item_id, "category": category,
                "success": False, "error": "image_not_found",
            })
            continue

        logger.info(f"[{idx+1}/{total}] Processing {item_id} ({category})")
        start_t = time.time()

        try:
            result = orchestrator.execute({
                "image_path": str(img_path),
                "prompt": prompt,
                "country": country,
                "category": category,
                "max_iterations": max_iterations,
                "score_threshold": score_threshold,
            })

            elapsed = time.time() - start_t
            score_history = result.data.get("score_history", [])
            improvement = result.data.get("improvement", 0)
            final_score = result.data.get("final_score", 0)
            initial_score = score_history[0] if score_history else 0

            entry = {
                "item_id": item_id,
                "category": category,
                "success": result.success,
                "score_history": score_history,
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": improvement,
                "iterations": result.data.get("iterations", 0),
                "elapsed_sec": round(elapsed, 1),
                "message": result.message,
            }
            results.append(entry)

            stats = category_stats[category]
            stats["count"] += 1
            stats["improvements"].append(improvement)
            stats["initial_scores"].append(initial_score)
            stats["final_scores"].append(final_score)
            if result.success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1

            logger.info(
                f"  Score: {initial_score} -> {final_score} "
                f"(improvement: {improvement:+.1f}, {elapsed:.0f}s)"
            )

        except Exception as e:
            elapsed = time.time() - start_t
            logger.error(f"  Error: {e}")
            results.append({
                "item_id": item_id, "category": category,
                "success": False, "error": str(e),
                "elapsed_sec": round(elapsed, 1),
            })
            category_stats[category]["failures"] += 1

    return {
        "results": results,
        "category_stats": category_stats,
        "output_dir": str(output_dir),
    }


def compute_summary(results: List[Dict], category_stats: Dict) -> Dict:
    """Compute overall and per-category summary statistics."""
    successful = [r for r in results if r.get("success") and "improvement" in r]

    if not successful:
        return {"error": "no successful results"}

    improvements = [r["improvement"] for r in successful]
    initial_scores = [r["initial_score"] for r in successful]
    final_scores = [r["final_score"] for r in successful]
    positive_improvements = [i for i in improvements if i > 0]

    overall = {
        "total_images": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_improvement": round(sum(improvements) / len(improvements), 3) if improvements else 0,
        "avg_initial_score": round(sum(initial_scores) / len(initial_scores), 3) if initial_scores else 0,
        "avg_final_score": round(sum(final_scores) / len(final_scores), 3) if final_scores else 0,
        "positive_improvement_rate": f"{len(positive_improvements)}/{len(successful)}",
        "positive_improvement_pct": round(len(positive_improvements) / len(successful) * 100, 1) if successful else 0,
        "max_improvement": max(improvements) if improvements else 0,
        "min_improvement": min(improvements) if improvements else 0,
    }

    per_category = {}
    for cat, stats in category_stats.items():
        if stats["count"] == 0:
            continue
        imps = stats["improvements"]
        pos = [i for i in imps if i > 0]
        per_category[cat] = {
            "count": stats["count"],
            "successes": stats["successes"],
            "failures": stats["failures"],
            "avg_improvement": round(sum(imps) / len(imps), 3) if imps else 0,
            "avg_initial_score": round(sum(stats["initial_scores"]) / len(stats["initial_scores"]), 3) if stats["initial_scores"] else 0,
            "avg_final_score": round(sum(stats["final_scores"]) / len(stats["final_scores"]), 3) if stats["final_scores"] else 0,
            "positive_rate": f"{len(pos)}/{len(imps)}",
        }

    return {"overall": overall, "per_category": per_category}


def main():
    parser = argparse.ArgumentParser(description="WorldCCUB Bulk Pipeline")
    parser.add_argument("--country", type=str, default="korea")
    parser.add_argument("--category", type=str, default=None, help="Filter to specific category")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of images to sample")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=7.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-indexing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("WorldCCUB Bulk Pipeline")
    logger.info("=" * 60)
    logger.info(f"Country: {args.country}")
    logger.info(f"Category filter: {args.category or 'ALL'}")
    logger.info(f"Sample size: {args.sample_size or 'ALL'}")

    # Load and filter dataset
    dataset = load_dataset(args.country)
    logger.info(f"Loaded {len(dataset)} items")

    dataset = filter_by_category(dataset, args.category)
    logger.info(f"After category filter: {len(dataset)} items")

    dataset = sample_dataset(dataset, args.sample_size, args.seed)
    logger.info(f"After sampling: {len(dataset)} items")

    # Run agent loop
    start = time.time()
    bulk_results = run_bulk_agent_loop(
        country=args.country,
        dataset=dataset,
        max_iterations=args.max_iterations,
        score_threshold=args.score_threshold,
    )
    total_elapsed = time.time() - start

    # Compute summary
    summary = compute_summary(bulk_results["results"], bulk_results["category_stats"])

    # Save results
    output_file = Path(bulk_results["output_dir"]) / "bulk_results.json"
    full_results = {
        "experiment_id": f"bulk_{args.country}_{time.strftime('%Y%m%d_%H%M%S')}",
        "country": args.country,
        "category_filter": args.category,
        "sample_size": args.sample_size,
        "max_iterations": args.max_iterations,
        "score_threshold": args.score_threshold,
        "total_elapsed_sec": round(total_elapsed, 1),
        "summary": summary,
        "per_image_results": bulk_results["results"],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # Also save to experiments/
    exp_file = EXPERIMENTS_DIR / f"bulk_{args.country}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(exp_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 60)
    logger.info("BULK PIPELINE RESULTS")
    logger.info("=" * 60)
    if "error" not in summary:
        o = summary["overall"]
        logger.info(f"Total: {o['total_images']} images ({o['successful']} OK, {o['failed']} failed)")
        logger.info(f"Avg improvement: {o['avg_improvement']:+.3f}")
        logger.info(f"Avg initial score: {o['avg_initial_score']:.3f}")
        logger.info(f"Avg final score: {o['avg_final_score']:.3f}")
        logger.info(f"Positive improvement rate: {o['positive_improvement_rate']} ({o['positive_improvement_pct']}%)")
        logger.info(f"Total time: {total_elapsed:.0f}s")
        logger.info("")
        logger.info("Per-category results:")
        for cat, cs in sorted(summary["per_category"].items()):
            logger.info(
                f"  {cat:25s}: n={cs['count']:3d}, "
                f"avg_imp={cs['avg_improvement']:+.2f}, "
                f"avg_final={cs['avg_final_score']:.2f}, "
                f"pos_rate={cs['positive_rate']}"
            )
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Experiment file: {exp_file}")


if __name__ == "__main__":
    main()
