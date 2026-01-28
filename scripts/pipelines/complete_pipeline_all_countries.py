#!/usr/bin/env python3
"""
Complete pipeline for all countries: Knowledge → CLIP → RAG
Runs everything sequentially with proper logging and error handling.
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / f"complete_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

COUNTRIES = [
    'korea', 'china', 'japan', 'usa', 'nigeria',
    'mexico', 'kenya', 'italy', 'france', 'germany'
]


def run_command(cmd, env=None, timeout=None):
    """Run a command and return success status."""
    try:
        proc = subprocess.run(
            cmd,
            env=env or dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout
        )

        # Log output
        for line in proc.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")

        return proc.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out")
        return False
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return False


def step1_extract_knowledge():
    """Step 1: Extract cultural knowledge for all countries."""
    logger.info("="*80)
    logger.info("STEP 1: EXTRACTING CULTURAL KNOWLEDGE")
    logger.info("="*80)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "stable_extract_all_countries.py")
    ]

    logger.info("Running stable knowledge extraction...")
    success = run_command(cmd, timeout=28800)  # 8 hour timeout

    if success:
        logger.info("✓ Knowledge extraction completed")
    else:
        logger.error("✗ Knowledge extraction failed")

    return success


def step2_build_clip_indices():
    """Step 2: Build CLIP indices for all countries."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: BUILDING CLIP INDICES")
    logger.info("="*80)

    results = {}

    for country in COUNTRIES:
        logger.info(f"\nBuilding CLIP index for {country.upper()}...")

        images_dir = PROJECT_ROOT / "data" / "country_packs" / country / "images"
        dataset_path = PROJECT_ROOT / "data" / "country_packs" / country / "approved_dataset.json"
        output_dir = PROJECT_ROOT / "data" / "clip_index" / country

        if not images_dir.exists():
            logger.warning(f"Skipping {country}: images directory not found")
            results[country] = False
            continue

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "indexing" / "build_clip_image_index.py"),
            "--country", country,
            "--images-dir", str(images_dir),
            "--output-dir", str(output_dir)
        ]

        if dataset_path.exists():
            cmd.extend(["--dataset", str(dataset_path)])

        success = run_command(cmd, timeout=3600)  # 1 hour timeout per country
        results[country] = success

        if success:
            logger.info(f"✓ {country.upper()} CLIP index built")
        else:
            logger.error(f"✗ {country.upper()} CLIP index failed")

        # Wait between countries
        time.sleep(5)

    successful = sum(results.values())
    logger.info(f"\nCLIP indices: {successful}/{len(COUNTRIES)} successful")

    return all(results.values())


def step3_build_rag_indices():
    """Step 3: Build RAG indices for all countries."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: BUILDING RAG INDICES")
    logger.info("="*80)

    results = {}

    for country in COUNTRIES:
        logger.info(f"\nBuilding RAG index for {country.upper()}...")

        knowledge_file = PROJECT_ROOT / "data" / "cultural_knowledge" / f"{country}_knowledge.json"
        index_dir = PROJECT_ROOT / "data" / "cultural_index" / country

        if not knowledge_file.exists():
            logger.warning(f"Skipping {country}: knowledge file not found")
            results[country] = False
            continue

        # Check if knowledge has content
        with open(knowledge_file) as f:
            data = json.load(f)
            if data.get('extracted_count', 0) == 0:
                logger.warning(f"Skipping {country}: no knowledge extracted")
                results[country] = False
                continue

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "indexing" / "integrate_knowledge_to_rag.py"),
            "--knowledge-file", str(knowledge_file),
            "--index-dir", str(index_dir),
            "--rebuild"
        ]

        success = run_command(cmd, timeout=1800)  # 30 min timeout per country
        results[country] = success

        if success:
            logger.info(f"✓ {country.upper()} RAG index built")
        else:
            logger.error(f"✗ {country.upper()} RAG index failed")

        # Wait between countries
        time.sleep(5)

    successful = sum(results.values())
    logger.info(f"\nRAG indices: {successful}/{len(COUNTRIES)} successful")

    return all(results.values())


def verify_pipeline():
    """Verify that all components are ready."""
    logger.info("\n" + "="*80)
    logger.info("PIPELINE VERIFICATION")
    logger.info("="*80)

    for country in COUNTRIES:
        # Check knowledge
        knowledge_path = PROJECT_ROOT / "data" / "cultural_knowledge" / f"{country}_knowledge.json"
        knowledge_ok = knowledge_path.exists()
        knowledge_count = 0
        if knowledge_ok:
            with open(knowledge_path) as f:
                data = json.load(f)
                knowledge_count = data.get('extracted_count', 0)

        # Check CLIP
        clip_path = PROJECT_ROOT / "data" / "clip_index" / country / "clip.index"
        clip_ok = clip_path.exists()

        # Check RAG
        rag_path = PROJECT_ROOT / "data" / "cultural_index" / country / "faiss.index"
        rag_ok = rag_path.exists()

        # Status
        status = "✓" if all([knowledge_ok, clip_ok, rag_ok]) else "✗"
        k_mark = "K" if knowledge_ok else " "
        c_mark = "C" if clip_ok else " "
        r_mark = "R" if rag_ok else " "

        logger.info(
            f"{status} {country:10s}: [{k_mark}{c_mark}{r_mark}] "
            f"Knowledge: {knowledge_count} items"
        )

    logger.info("="*80)


def main():
    """Run complete pipeline."""
    logger.info("="*80)
    logger.info("COMPLETE PIPELINE FOR ALL COUNTRIES")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*80)

    start_time = time.time()

    # Step 1: Extract knowledge
    step1_success = step1_extract_knowledge()

    # Step 2: Build CLIP indices
    step2_success = step2_build_clip_indices()

    # Step 3: Build RAG indices
    step3_success = step3_build_rag_indices()

    # Verify
    verify_pipeline()

    # Summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    logger.info(f"Step 1 (Knowledge):   {'✓' if step1_success else '✗'}")
    logger.info(f"Step 2 (CLIP):        {'✓' if step2_success else '✗'}")
    logger.info(f"Step 3 (RAG):         {'✓' if step3_success else '✗'}")
    logger.info(f"Total time: {hours}h {minutes}m")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("="*80)

    if all([step1_success, step2_success, step3_success]):
        logger.info("\n🎉 Complete pipeline finished successfully!")
        return 0
    else:
        logger.error("\n⚠️ Pipeline completed with errors. Check log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
