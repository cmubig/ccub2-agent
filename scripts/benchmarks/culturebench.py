#!/usr/bin/env python3
"""
CultureBench — Benchmark Tasks for WorldCCUB Paper

Implements four benchmark tasks:
    Task A (Fidelity):     CultScore distribution across all images
    Task B (Degradation):  Score change across iterations (loop dynamics)
    Task C (Cross-Country): Cross-country comparison (requires 2+ countries)
    Task D (Contrastive):  Model comparison (FLUX vs Qwen vs SD3.5)

Usage:
    # All tasks for Korea pilot
    python scripts/benchmarks/culturebench.py --country korea --tasks A,B,D

    # Task A only on a sample
    python scripts/benchmarks/culturebench.py --country korea --tasks A --sample-size 50

    # Task D with specific models
    python scripts/benchmarks/culturebench.py --country korea --tasks D --models qwen,sd35
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
logger = logging.getLogger("culturebench")

DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)


def load_dataset(country: str) -> List[Dict]:
    """Load approved dataset."""
    path = DATA_DIR / "country_packs" / country / "approved_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_path(country: str, item_id: str) -> Optional[Path]:
    """Find image file for an item."""
    images_dir = DATA_DIR / "country_packs" / country / "images"
    for ext in [".jpg", ".jpeg", ".png"]:
        p = images_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    return None


# ======================================================================
# Task A: Fidelity — CultScore distribution
# ======================================================================

def task_a_fidelity(
    country: str,
    dataset: List[Dict],
    output_dir: Path,
) -> Dict:
    """
    Evaluate CultScore distribution across all images.
    Shows the overall cultural fidelity of the dataset.
    """
    logger.info("=" * 60)
    logger.info("Task A: Cultural Fidelity Score Distribution")
    logger.info("=" * 60)

    from ccub2_agent.agents.base_agent import AgentConfig
    from ccub2_agent.agents.core.judge_agent import JudgeAgent
    from ccub2_agent.detection.vlm_detector import VLMCulturalDetector

    # Create shared VLM
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
    category_scores = defaultdict(list)

    for idx, item in enumerate(dataset):
        item_id = item["__id__"]
        category = item.get("category_normalized", "uncategorized")
        img_path = get_image_path(country, item_id)

        if img_path is None:
            continue

        logger.info(f"[{idx+1}/{len(dataset)}] Scoring {item_id} ({category})")

        judge_result = judge.execute({
            "image_path": str(img_path),
            "prompt": item.get("description", f"Cultural image from {country}"),
            "country": country,
            "category": category,
        })

        if judge_result.success:
            score = judge_result.data.get("cultural_score", 0)
            results.append({
                "item_id": item_id,
                "category": category,
                "cultural_score": score,
                "prompt_score": judge_result.data.get("prompt_score", 0),
            })
            category_scores[category].append(score)

    # Compute statistics
    all_scores = [r["cultural_score"] for r in results]
    summary = {
        "task": "A_fidelity",
        "country": country,
        "n_images": len(results),
        "mean_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
        "median_score": round(sorted(all_scores)[len(all_scores) // 2], 3) if all_scores else 0,
        "min_score": min(all_scores) if all_scores else 0,
        "max_score": max(all_scores) if all_scores else 0,
        "std_score": round(_std(all_scores), 3) if all_scores else 0,
        "score_distribution": _histogram(all_scores, bins=10),
        "per_category": {
            cat: {
                "n": len(scores),
                "mean": round(sum(scores) / len(scores), 3),
                "std": round(_std(scores), 3),
            }
            for cat, scores in sorted(category_scores.items())
        },
        "per_image": results,
    }

    _save(summary, output_dir / "task_a_fidelity.json")
    logger.info(f"Task A complete: mean={summary['mean_score']}, n={summary['n_images']}")
    return summary


# ======================================================================
# Task B: Degradation — Score change across iterations
# ======================================================================

def task_b_degradation(
    country: str,
    dataset: List[Dict],
    output_dir: Path,
    max_iterations: int = 3,
) -> Dict:
    """
    Track score changes across agent loop iterations.
    Measures if the loop improves, maintains, or degrades quality.
    """
    logger.info("=" * 60)
    logger.info("Task B: Iteration Degradation Analysis")
    logger.info("=" * 60)

    from ccub2_agent.agents.base_agent import AgentConfig
    from ccub2_agent.agents.core.orchestrator_agent import OrchestratorAgent

    config = AgentConfig(country=country, output_dir=output_dir / "task_b_outputs")
    orchestrator = OrchestratorAgent(config)

    results = []
    for idx, item in enumerate(dataset):
        item_id = item["__id__"]
        category = item.get("category_normalized", "uncategorized")
        img_path = get_image_path(country, item_id)

        if img_path is None:
            continue

        logger.info(f"[{idx+1}/{len(dataset)}] Processing {item_id} ({category})")

        result = orchestrator.execute({
            "image_path": str(img_path),
            "prompt": item.get("description", f"Cultural image from {country}"),
            "country": country,
            "category": category,
            "max_iterations": max_iterations,
            "score_threshold": 10.0,  # Never stop early — run all iterations
        })

        if result.success:
            score_history = result.data.get("score_history", [])
            results.append({
                "item_id": item_id,
                "category": category,
                "score_history": score_history,
                "improvement": result.data.get("improvement", 0),
                "iterations": result.data.get("iterations", 0),
            })

    # Analyze patterns
    improvements = [r["improvement"] for r in results]
    patterns = {"improved": 0, "unchanged": 0, "degraded": 0}
    for imp in improvements:
        if imp > 0:
            patterns["improved"] += 1
        elif imp < 0:
            patterns["degraded"] += 1
        else:
            patterns["unchanged"] += 1

    # Per-iteration analysis
    max_iter = max((len(r["score_history"]) for r in results), default=0)
    per_iteration_means = []
    for i in range(max_iter):
        scores_at_i = [
            r["score_history"][i]
            for r in results
            if i < len(r["score_history"])
        ]
        per_iteration_means.append({
            "iteration": i,
            "n": len(scores_at_i),
            "mean": round(sum(scores_at_i) / len(scores_at_i), 3) if scores_at_i else 0,
        })

    summary = {
        "task": "B_degradation",
        "country": country,
        "n_images": len(results),
        "avg_improvement": round(sum(improvements) / len(improvements), 3) if improvements else 0,
        "patterns": patterns,
        "per_iteration_means": per_iteration_means,
        "per_image": results,
    }

    _save(summary, output_dir / "task_b_degradation.json")
    logger.info(f"Task B complete: avg_imp={summary['avg_improvement']}, patterns={patterns}")
    return summary


# ======================================================================
# Task C: Cross-Country (placeholder)
# ======================================================================

def task_c_cross_country(
    countries: List[str],
    output_dir: Path,
) -> Dict:
    """Cross-country comparison. Requires 2+ countries with data."""
    logger.info("=" * 60)
    logger.info("Task C: Cross-Country Comparison")
    logger.info("=" * 60)

    available = []
    for c in countries:
        ds_path = DATA_DIR / "country_packs" / c / "approved_dataset.json"
        if ds_path.exists():
            available.append(c)

    if len(available) < 2:
        logger.warning(f"Need 2+ countries, only {len(available)} available. Skipping.")
        summary = {
            "task": "C_cross_country",
            "status": "skipped",
            "reason": f"Only {len(available)} countries available (need 2+)",
            "available_countries": available,
        }
        _save(summary, output_dir / "task_c_cross_country.json")
        return summary

    # Run Task A for each country and compare
    per_country = {}
    for c in available:
        dataset = load_dataset(c)
        country_dir = output_dir / f"task_c_{c}"
        country_dir.mkdir(parents=True, exist_ok=True)
        result = task_a_fidelity(c, dataset[:50], country_dir)  # sample 50 per country
        per_country[c] = {
            "mean_score": result["mean_score"],
            "n_images": result["n_images"],
        }

    summary = {
        "task": "C_cross_country",
        "status": "completed",
        "countries": per_country,
    }
    _save(summary, output_dir / "task_c_cross_country.json")
    return summary


# ======================================================================
# Task D: Contrastive — Model comparison
# ======================================================================

def task_d_contrastive(
    country: str,
    dataset: List[Dict],
    output_dir: Path,
    models: List[str] = None,
    max_iterations: int = 3,
) -> Dict:
    """
    Compare different T2I/I2I models on the same images.
    Default models: qwen, sd35 (FLUX requires separate setup).
    """
    logger.info("=" * 60)
    logger.info("Task D: Contrastive Model Comparison")
    logger.info("=" * 60)

    if models is None:
        models = ["qwen", "sd35"]

    from ccub2_agent.agents.base_agent import AgentConfig
    from ccub2_agent.agents.core.orchestrator_agent import OrchestratorAgent

    model_results = {}

    for model_type in models:
        logger.info(f"\n--- Model: {model_type} ---")

        config = AgentConfig(country=country, output_dir=output_dir / f"task_d_{model_type}")

        # Override the edit model by monkey-patching after init
        orchestrator = OrchestratorAgent(config)
        orchestrator.edit_agent.i2i_model = model_type

        per_image = []
        for idx, item in enumerate(dataset):
            item_id = item["__id__"]
            category = item.get("category_normalized", "uncategorized")
            img_path = get_image_path(country, item_id)

            if img_path is None:
                continue

            logger.info(f"[{model_type}][{idx+1}/{len(dataset)}] {item_id}")

            result = orchestrator.execute({
                "image_path": str(img_path),
                "prompt": item.get("description", f"Cultural image from {country}"),
                "country": country,
                "category": category,
                "max_iterations": max_iterations,
                "score_threshold": 7.0,
            })

            if result.success:
                per_image.append({
                    "item_id": item_id,
                    "category": category,
                    "score_history": result.data.get("score_history", []),
                    "improvement": result.data.get("improvement", 0),
                    "final_score": result.data.get("final_score", 0),
                })

        improvements = [r["improvement"] for r in per_image]
        final_scores = [r["final_score"] for r in per_image]

        model_results[model_type] = {
            "n_images": len(per_image),
            "avg_improvement": round(sum(improvements) / len(improvements), 3) if improvements else 0,
            "avg_final_score": round(sum(final_scores) / len(final_scores), 3) if final_scores else 0,
            "per_image": per_image,
        }

    summary = {
        "task": "D_contrastive",
        "country": country,
        "models_compared": models,
        "model_results": model_results,
    }

    _save(summary, output_dir / "task_d_contrastive.json")
    logger.info(f"Task D complete. Models: {models}")
    for m, r in model_results.items():
        logger.info(f"  {m}: avg_imp={r['avg_improvement']:+.2f}, avg_final={r['avg_final_score']:.2f}")
    return summary


# ======================================================================
# Utilities
# ======================================================================

def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _histogram(values: List[float], bins: int = 10) -> Dict[str, int]:
    """Simple histogram for score distribution."""
    if not values:
        return {}
    hist = {}
    for v in values:
        bucket = int(v)
        bucket = max(0, min(bucket, 10))
        key = str(bucket)
        hist[key] = hist.get(key, 0) + 1
    return hist


def _save(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="CultureBench Evaluation Tasks")
    parser.add_argument("--country", type=str, default="korea")
    parser.add_argument("--tasks", type=str, default="A,B,D", help="Comma-separated tasks: A,B,C,D")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated models for Task D")
    parser.add_argument("--countries", type=str, default=None, help="Comma-separated countries for Task C")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = [t.strip().upper() for t in args.tasks.split(",")]
    output_dir = EXPERIMENTS_DIR / f"culturebench_{args.country}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.country)
    if args.sample_size and args.sample_size < len(dataset):
        random.seed(args.seed)
        dataset = random.sample(dataset, args.sample_size)

    logger.info(f"CultureBench: country={args.country}, tasks={tasks}, n={len(dataset)}")

    all_results = {}

    if "A" in tasks:
        all_results["A"] = task_a_fidelity(args.country, dataset, output_dir)

    if "B" in tasks:
        all_results["B"] = task_b_degradation(
            args.country, dataset, output_dir, args.max_iterations,
        )

    if "C" in tasks:
        countries = args.countries.split(",") if args.countries else [args.country]
        all_results["C"] = task_c_cross_country(countries, output_dir)

    if "D" in tasks:
        models = args.models.split(",") if args.models else None
        all_results["D"] = task_d_contrastive(
            args.country, dataset, output_dir, models, args.max_iterations,
        )

    # Save combined results
    combined = {
        "benchmark": "CultureBench",
        "country": args.country,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks_run": tasks,
        "n_images": len(dataset),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in all_results.items()},
    }
    _save(combined, output_dir / "culturebench_summary.json")
    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
