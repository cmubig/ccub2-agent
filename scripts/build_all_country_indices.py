#!/usr/bin/env python3
"""
Build Knowledge Base, CLIP Index, and RAG Index for All Countries

This master script automatically processes all countries with approved datasets:
1. Extract cultural knowledge from images (using Qwen3-VL-8B)
2. Build CLIP-based image index for similarity search
3. Integrate knowledge into RAG/FAISS index

Usage:
    # Process all countries
    python scripts/build_all_country_indices.py

    # Process specific countries
    python scripts/build_all_country_indices.py --countries korea china japan

    # Skip specific steps
    python scripts/build_all_country_indices.py --skip-knowledge  # Use existing knowledge
    python scripts/build_all_country_indices.py --skip-clip       # Skip CLIP indexing
    python scripts/build_all_country_indices.py --skip-rag        # Skip RAG integration

    # Parallel knowledge extraction (faster but uses more GPU memory)
    python scripts/build_all_country_indices.py --parallel-gpus 0,1

    # Force rebuild even if indices exist
    python scripts/build_all_country_indices.py --force

    # Dry run (show what would be done)
    python scripts/build_all_country_indices.py --dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CountryIndexBuilder:
    """Manages index building for all countries."""

    def __init__(
        self,
        data_root: Path,
        countries: Optional[List[str]] = None,
        skip_knowledge: bool = False,
        skip_clip: bool = False,
        skip_rag: bool = False,
        parallel_gpus: Optional[List[int]] = None,
        force: bool = False,
        dry_run: bool = False,
    ):
        self.data_root = data_root
        self.skip_knowledge = skip_knowledge
        self.skip_clip = skip_clip
        self.skip_rag = skip_rag
        self.parallel_gpus = parallel_gpus or []
        self.force = force
        self.dry_run = dry_run

        # Discover countries
        self.countries = self._discover_countries(countries)
        logger.info(f"Found {len(self.countries)} countries to process: {', '.join(self.countries)}")

    def _discover_countries(self, requested_countries: Optional[List[str]]) -> List[str]:
        """Find all countries with approved datasets."""
        country_packs_dir = self.data_root / "country_packs"
        if not country_packs_dir.exists():
            logger.error(f"Country packs directory not found: {country_packs_dir}")
            return []

        all_countries = []
        for country_dir in sorted(country_packs_dir.iterdir()):
            if not country_dir.is_dir():
                continue

            country = country_dir.name
            if country == "general":  # Skip general category
                continue

            # Check if approved_dataset.json exists
            dataset_file = country_dir / "approved_dataset.json"
            if not dataset_file.exists():
                logger.warning(f"Skipping {country}: no approved_dataset.json")
                continue

            # Check if images directory exists with images
            images_dir = country_dir / "images"
            if not images_dir.exists() or not any(images_dir.iterdir()):
                logger.warning(f"Skipping {country}: no images found")
                continue

            all_countries.append(country)

        # Filter by requested countries if specified
        if requested_countries:
            filtered = [c for c in all_countries if c in requested_countries]
            missing = set(requested_countries) - set(filtered)
            if missing:
                logger.warning(f"Requested countries not found or invalid: {', '.join(missing)}")
            return filtered

        return all_countries

    def _check_status(self, country: str) -> Dict[str, bool]:
        """Check what's already built for a country."""
        status = {
            'knowledge': False,
            'clip': False,
            'rag': False,
        }

        # Check knowledge file
        knowledge_file = self.data_root / "cultural_knowledge" / f"{country}_knowledge.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file) as f:
                    data = json.load(f)
                    if data.get('knowledge') and len(data['knowledge']) > 0:
                        status['knowledge'] = True
            except:
                pass

        # Check CLIP index
        clip_index = self.data_root / "clip_index" / country / "clip.index"
        clip_meta = self.data_root / "clip_index" / country / "clip_metadata.jsonl"
        if clip_index.exists() and clip_meta.exists():
            status['clip'] = True

        # Check RAG index
        rag_index = self.data_root / "cultural_index" / country / "faiss.index"
        rag_meta = self.data_root / "cultural_index" / country / "metadata.jsonl"
        if rag_index.exists() and rag_meta.exists():
            status['rag'] = True

        return status

    def _run_command(self, cmd: List[str], description: str) -> bool:
        """Run a subprocess command."""
        logger.info(f"  → {description}")
        logger.info(f"  Command: {' '.join(cmd)}")

        if self.dry_run:
            logger.info("  [DRY RUN] Would execute command")
            return True

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"  ✓ {description} completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ {description} failed")
            logger.error(f"  Error: {e.stderr}")
            return False

    def build_knowledge(self, country: str) -> bool:
        """Extract cultural knowledge for a country."""
        logger.info(f"[{country}] Extracting cultural knowledge...")

        knowledge_file = self.data_root / "cultural_knowledge" / f"{country}_knowledge.json"
        data_dir = self.data_root / "country_packs" / country
        dataset_file = data_dir / "approved_dataset.json"

        # Check dataset size
        with open(dataset_file) as f:
            dataset = json.load(f)
            total_items = len(dataset)

        logger.info(f"  Dataset size: {total_items} items")

        # Build command - NOTE: extract_cultural_knowledge.py uses --data-dir and --output
        cmd = [
            "conda", "run", "-n", "ccub2",
            "python", str(PROJECT_ROOT / "scripts/02_data_processing/extract_cultural_knowledge.py"),
            "--data-dir", str(data_dir),
            "--output", str(knowledge_file),
        ]

        # Add GPU selection if parallel processing
        if self.parallel_gpus and len(self.parallel_gpus) > 0:
            gpu_id = self.parallel_gpus[0]  # For now, use first GPU
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"  Using GPU {gpu_id}")

        return self._run_command(cmd, f"Knowledge extraction for {country}")

    def build_clip_index(self, country: str) -> bool:
        """Build CLIP index for a country."""
        logger.info(f"[{country}] Building CLIP image index...")

        clip_dir = self.data_root / "clip_index" / country
        clip_dir.mkdir(parents=True, exist_ok=True)

        # Prioritize enhanced dataset if available
        enhanced_dataset = self.data_root / "country_packs" / country / "approved_dataset_enhanced.json"
        dataset_file = self.data_root / "country_packs" / country / "approved_dataset.json"
        if enhanced_dataset.exists():
            dataset_file = enhanced_dataset
            logger.info(f"  Using enhanced dataset")

        images_dir = self.data_root / "country_packs" / country / "images"

        # NOTE: build_clip_image_index.py uses --dataset (not --dataset-file)
        cmd = [
            "conda", "run", "-n", "ccub2",
            "python", str(PROJECT_ROOT / "scripts/03_indexing/build_clip_image_index.py"),
            "--country", country,
            "--images-dir", str(images_dir),
            "--dataset", str(dataset_file),
            "--output-dir", str(clip_dir),
        ]

        return self._run_command(cmd, f"CLIP index building for {country}")

    def build_rag_index(self, country: str) -> bool:
        """Integrate knowledge into RAG index."""
        logger.info(f"[{country}] Building RAG/FAISS index...")

        rag_dir = self.data_root / "cultural_index" / country
        rag_dir.mkdir(parents=True, exist_ok=True)

        knowledge_file = self.data_root / "cultural_knowledge" / f"{country}_knowledge.json"

        # Check if knowledge file exists
        if not knowledge_file.exists():
            logger.warning(f"  Knowledge file not found: {knowledge_file}")
            logger.warning(f"  Skipping RAG index for {country}")
            return False

        cmd = [
            "conda", "run", "-n", "ccub2",
            "python", str(PROJECT_ROOT / "scripts/03_indexing/integrate_knowledge_to_rag.py"),
            "--knowledge-file", str(knowledge_file),
            "--index-dir", str(rag_dir),
            "--rebuild",  # Always rebuild for consistency
        ]

        return self._run_command(cmd, f"RAG index integration for {country}")

    def process_country(self, country: str) -> Dict[str, bool]:
        """Process all indices for a single country."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing: {country.upper()}")
        logger.info("=" * 80)

        results = {}

        # Check current status
        status = self._check_status(country)
        logger.info(f"Current status:")
        logger.info(f"  Knowledge: {'✓' if status['knowledge'] else '✗'}")
        logger.info(f"  CLIP:      {'✓' if status['clip'] else '✗'}")
        logger.info(f"  RAG:       {'✓' if status['rag'] else '✗'}")
        logger.info("")

        # Step 1: Extract Knowledge
        if not self.skip_knowledge:
            if status['knowledge'] and not self.force:
                logger.info(f"[{country}] Knowledge already exists, skipping (use --force to rebuild)")
                results['knowledge'] = True
            else:
                results['knowledge'] = self.build_knowledge(country)
        else:
            logger.info(f"[{country}] Skipping knowledge extraction (--skip-knowledge)")
            results['knowledge'] = status['knowledge']

        # Step 2: Build CLIP Index
        if not self.skip_clip:
            if status['clip'] and not self.force:
                logger.info(f"[{country}] CLIP index already exists, skipping (use --force to rebuild)")
                results['clip'] = True
            else:
                results['clip'] = self.build_clip_index(country)
        else:
            logger.info(f"[{country}] Skipping CLIP index (--skip-clip)")
            results['clip'] = status['clip']

        # Step 3: Build RAG Index
        if not self.skip_rag:
            if status['rag'] and not self.force:
                logger.info(f"[{country}] RAG index already exists, skipping (use --force to rebuild)")
                results['rag'] = True
            else:
                results['rag'] = self.build_rag_index(country)
        else:
            logger.info(f"[{country}] Skipping RAG index (--skip-rag)")
            results['rag'] = status['rag']

        # Summary
        logger.info("")
        logger.info(f"[{country}] Results:")
        logger.info(f"  Knowledge: {'✓' if results.get('knowledge') else '✗'}")
        logger.info(f"  CLIP:      {'✓' if results.get('clip') else '✗'}")
        logger.info(f"  RAG:       {'✓' if results.get('rag') else '✗'}")

        return results

    def run(self) -> Dict[str, Dict[str, bool]]:
        """Process all countries."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("MULTI-COUNTRY INDEX BUILDER")
        logger.info("=" * 80)
        logger.info(f"Countries: {len(self.countries)}")
        logger.info(f"Skip knowledge: {self.skip_knowledge}")
        logger.info(f"Skip CLIP: {self.skip_clip}")
        logger.info(f"Skip RAG: {self.skip_rag}")
        logger.info(f"Force rebuild: {self.force}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("")

        all_results = {}

        for i, country in enumerate(self.countries, 1):
            logger.info(f"Processing {i}/{len(self.countries)}: {country}")
            results = self.process_country(country)
            all_results[country] = results

        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)

        for country, results in all_results.items():
            status = "✓" if all(results.values()) else "✗"
            logger.info(f"{status} {country:15s} - Knowledge: {results.get('knowledge', False)}, "
                       f"CLIP: {results.get('clip', False)}, RAG: {results.get('rag', False)}")

        total_success = sum(1 for r in all_results.values() if all(r.values()))
        logger.info("")
        logger.info(f"Successfully completed: {total_success}/{len(self.countries)} countries")

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Build knowledge base, CLIP, and RAG indices for all countries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "ccub2-agent" / "data",
        help="Root data directory (default: ~/ccub2-agent/data)"
    )

    parser.add_argument(
        "--countries",
        nargs='+',
        help="Specific countries to process (default: all)"
    )

    parser.add_argument(
        "--skip-knowledge",
        action="store_true",
        help="Skip knowledge extraction (use existing knowledge files)"
    )

    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP index building"
    )

    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG index integration"
    )

    parser.add_argument(
        "--parallel-gpus",
        type=str,
        help="GPU IDs for parallel processing (comma-separated, e.g., '0,1')"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if indices already exist"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running"
    )

    args = parser.parse_args()

    # Parse GPU list
    parallel_gpus = []
    if args.parallel_gpus:
        parallel_gpus = [int(g.strip()) for g in args.parallel_gpus.split(',')]

    # Build indices
    builder = CountryIndexBuilder(
        data_root=args.data_root,
        countries=args.countries,
        skip_knowledge=args.skip_knowledge,
        skip_clip=args.skip_clip,
        skip_rag=args.skip_rag,
        parallel_gpus=parallel_gpus,
        force=args.force,
        dry_run=args.dry_run,
    )

    results = builder.run()

    # Exit with error if any country failed
    all_success = all(all(r.values()) for r in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
