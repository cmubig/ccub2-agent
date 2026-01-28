#!/usr/bin/env python3
"""
WorldCCUB Full Pipeline

End-to-end pipeline: load data -> download images -> build country pack ->
extract knowledge -> build indices -> run agent loop -> gap analysis.

Usage:
    python scripts/run_full_pipeline.py --country korea
    python scripts/run_full_pipeline.py --country korea --skip-download --skip-indexing
    python scripts/run_full_pipeline.py --all-countries
    python scripts/run_full_pipeline.py --country korea --skip-agents --sample-size 5
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

# Project root = ccub2-agent/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.data.firebase_client import FirebaseClient
from ccub2_agent.data.country_pack import CountryDataPack
from ccub2_agent.data.data_gap_detector import DataGapDetector
from ccub2_agent.data.job_creator import AgentJobCreator

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pipeline")

# Directories
DATA_DIR = PROJECT_ROOT / "data"
COUNTRY_PACKS_DIR = DATA_DIR / "country_packs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ======================================================================
# Pipeline Steps
# ======================================================================

def step_load_data(country: str) -> FirebaseClient:
    """Step 1: Initialize FirebaseClient and display stats."""
    logger.info(f"=== Step 1: Loading data for '{country}' ===")
    client = FirebaseClient()

    contribs = client.get_contributions(country=country)
    all_contribs = client.get_contributions()
    jobs = client.get_jobs(country=country)

    logger.info(f"Total contributions in CSV: {len(all_contribs)}")
    logger.info(f"Contributions for {country}: {len(contribs)}")
    logger.info(f"Jobs for {country}: {len(jobs)}")

    # Status breakdown
    statuses = {}
    for c in contribs:
        s = c.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    logger.info(f"Status breakdown: {statuses}")

    return client


def step_download_images(client: FirebaseClient, country: str, workers: int = 4) -> Path:
    """Step 2: Download images from Firebase Storage URLs."""
    logger.info(f"=== Step 2: Downloading images for '{country}' ===")
    images_dir = COUNTRY_PACKS_DIR / country / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    contribs = client.get_contributions(country=country)
    urls = []
    for c in contribs:
        url = c.get("imageURL", "")
        cid = c.get("__id__", "unknown")
        if url:
            ext = ".jpg"  # default
            if ".png" in url.lower():
                ext = ".png"
            elif ".jpeg" in url.lower():
                ext = ".jpeg"
            dest = images_dir / f"{cid}{ext}"
            urls.append((url, dest))

    if not urls:
        logger.warning(f"No images to download for {country}")
        return images_dir

    # Check how many already exist
    existing = sum(1 for _, dest in urls if dest.exists())
    to_download = len(urls) - existing
    logger.info(f"Images: {len(urls)} total, {existing} cached, {to_download} to download")

    if to_download == 0:
        logger.info("All images already cached, skipping download")
        return images_dir

    # Parallel download
    downloaded = 0
    failed = 0

    def _download_one(item):
        url, dest = item
        try:
            client.download_image(url, dest)
            return True
        except Exception as e:
            logger.warning(f"Failed: {dest.name} - {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, item): item for item in urls if not item[1].exists()}
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                failed += 1

    logger.info(f"Download complete: {downloaded} OK, {failed} failed")
    return images_dir


def step_build_country_pack(country: str) -> CountryDataPack:
    """Step 3: Build CountryDataPack and save approved dataset."""
    logger.info(f"=== Step 3: Building CountryDataPack for '{country}' ===")
    pack = CountryDataPack(country)
    stats = pack.get_stats()
    logger.info(f"Stats: {json.dumps(stats, indent=2, ensure_ascii=False)}")

    out_path = pack.save_as_approved_dataset()
    logger.info(f"Saved approved dataset: {out_path}")
    return pack


def step_extract_knowledge(country: str) -> bool:
    """Step 4: Run cultural knowledge extraction (subprocess)."""
    logger.info(f"=== Step 4: Extracting cultural knowledge for '{country}' ===")

    # Try the stable extraction script
    script = SCRIPTS_DIR / "data_processing" / "stable_extract_all_countries.py"
    if not script.exists():
        logger.warning(f"Knowledge extraction script not found: {script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        if result.returncode != 0:
            logger.error(f"Knowledge extraction failed:\n{result.stderr[:500]}")
            return False
        logger.info("Knowledge extraction completed")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Knowledge extraction timed out")
        return False
    except Exception as e:
        logger.error(f"Knowledge extraction error: {e}")
        return False


def step_build_indices(country: str) -> bool:
    """Step 5-6: Build CLIP and cultural indices (subprocess)."""
    logger.info(f"=== Step 5-6: Building indices for '{country}' ===")

    script = SCRIPTS_DIR / "indexing" / "build_all_country_indices.py"
    if not script.exists():
        logger.warning(f"Index building script not found: {script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script), "--countries", country],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            logger.error(f"Index building failed:\n{result.stderr[:500]}")
            return False
        logger.info("Index building completed")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Index building timed out")
        return False
    except Exception as e:
        logger.error(f"Index building error: {e}")
        return False


def step_run_agent_loop(country: str, sample_size: int = 3) -> dict:
    """Step 7: Run OrchestratorAgent on sample images."""
    logger.info(f"=== Step 7: Running agent loop for '{country}' (sample={sample_size}) ===")

    try:
        from ccub2_agent.agents.base_agent import AgentConfig
        from ccub2_agent.agents.core.orchestrator_agent import OrchestratorAgent

        config = AgentConfig(country=country)
        orchestrator = OrchestratorAgent(config)

        # Get sample images
        images_dir = COUNTRY_PACKS_DIR / country / "images"
        if not images_dir.exists():
            logger.warning(f"No images directory for {country}")
            return {"error": "no images"}

        image_files = list(images_dir.glob("*.*"))[:sample_size]
        if not image_files:
            logger.warning(f"No image files found in {images_dir}")
            return {"error": "no image files"}

        results = []
        for img_path in image_files:
            logger.info(f"Processing: {img_path.name}")
            result = orchestrator.execute({
                "image_path": str(img_path),
                "prompt": f"Cultural image from {country}",
                "country": country,
                "max_iterations": 3,
                "score_threshold": 7.0,
            })
            results.append({
                "image": img_path.name,
                "success": result.success,
                "message": result.message,
                "data": result.data,
            })
            logger.info(f"  Result: {result.message}")

        return {"results": results, "total": len(results)}

    except ImportError as e:
        logger.error(f"Agent import failed (missing ML dependencies?): {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Agent loop error: {e}", exc_info=True)
        return {"error": str(e)}


def step_gap_analysis(country: str) -> dict:
    """Step 8: Run gap analysis and create jobs."""
    logger.info(f"=== Step 8: Gap analysis for '{country}' ===")

    detector = DataGapDetector()
    gaps = detector.detect_gaps(country)
    summary = detector.get_coverage_summary(country)

    logger.info(f"Coverage summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")
    logger.info(f"Total gaps: {len(gaps)}")

    for g in gaps[:5]:
        logger.info(f"  Gap: {g['category']} (severity={g['severity']}, needed={g['needed']})")

    # Create jobs from gaps
    if gaps:
        creator = AgentJobCreator()
        job_ids = creator.create_jobs_from_gaps(gaps, country)
        logger.info(f"Created {len(job_ids)} collection jobs")
        return {"gaps": len(gaps), "jobs_created": len(job_ids), "summary": summary}

    return {"gaps": 0, "jobs_created": 0, "summary": summary}


# ======================================================================
# Main
# ======================================================================

def run_pipeline(
    country: str,
    skip_download: bool = False,
    skip_indexing: bool = False,
    skip_agents: bool = False,
    sample_size: int = 3,
):
    """Run the full pipeline for a single country."""
    logger.info(f"{'='*60}")
    logger.info(f"Starting pipeline for: {country}")
    logger.info(f"  skip_download={skip_download}, skip_indexing={skip_indexing}")
    logger.info(f"  skip_agents={skip_agents}, sample_size={sample_size}")
    logger.info(f"{'='*60}")

    start = time.time()

    # Step 1: Load data
    client = step_load_data(country)

    # Step 2: Download images
    if not skip_download:
        step_download_images(client, country)
    else:
        logger.info("Skipping image download (--skip-download)")

    # Step 3: Build country pack
    pack = step_build_country_pack(country)

    # Steps 4-6: Knowledge extraction & indexing
    if not skip_indexing:
        step_extract_knowledge(country)
        step_build_indices(country)
    else:
        logger.info("Skipping knowledge extraction & indexing (--skip-indexing)")

    # Step 7: Agent loop
    agent_results = {}
    if not skip_agents:
        agent_results = step_run_agent_loop(country, sample_size)
    else:
        logger.info("Skipping agent loop (--skip-agents)")

    # Step 8: Gap analysis
    gap_results = step_gap_analysis(country)

    elapsed = time.time() - start
    logger.info(f"{'='*60}")
    logger.info(f"Pipeline completed for {country} in {elapsed:.1f}s")
    logger.info(f"  Gaps found: {gap_results.get('gaps', 0)}")
    logger.info(f"  Jobs created: {gap_results.get('jobs_created', 0)}")
    if agent_results:
        logger.info(f"  Agent results: {agent_results.get('total', 0)} images processed")
    logger.info(f"{'='*60}")

    return {
        "country": country,
        "elapsed": elapsed,
        "gap_results": gap_results,
        "agent_results": agent_results,
    }


def main():
    parser = argparse.ArgumentParser(description="WorldCCUB Full Pipeline")
    parser.add_argument("--country", type=str, help="Country to process (e.g., korea)")
    parser.add_argument("--all-countries", action="store_true", help="Process all countries")
    parser.add_argument("--skip-download", action="store_true", help="Skip image download")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip knowledge extraction & indexing")
    parser.add_argument("--skip-agents", action="store_true", help="Skip agent loop")
    parser.add_argument("--sample-size", type=int, default=3, help="Number of sample images for agent loop")
    args = parser.parse_args()

    if not args.country and not args.all_countries:
        parser.error("Specify --country or --all-countries")

    if args.all_countries:
        client = FirebaseClient()
        countries = client.get_all_countries()
        logger.info(f"Processing all {len(countries)} countries: {countries}")
        for c in countries:
            try:
                run_pipeline(
                    country=c,
                    skip_download=args.skip_download,
                    skip_indexing=args.skip_indexing,
                    skip_agents=args.skip_agents,
                    sample_size=args.sample_size,
                )
            except Exception as e:
                logger.error(f"Pipeline failed for {c}: {e}", exc_info=True)
    else:
        run_pipeline(
            country=args.country,
            skip_download=args.skip_download,
            skip_indexing=args.skip_indexing,
            skip_agents=args.skip_agents,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
