#!/usr/bin/env python3
"""Merge curated and user-submitted datasets for a country.

Usage:
    python scripts/curation/05_merge_datasets.py --country korea --dry-run
    python scripts/curation/05_merge_datasets.py --country korea
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ccub2_agent.data.curation.merge_datasets import DatasetMerger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets for a country")
    parser.add_argument("--country", required=True, help="Country code")
    parser.add_argument("--data-dir", default=None, help="Base data directory")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't write")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parents[2] / "data"
    merger = DatasetMerger(data_dir)
    summary = merger.merge_country(args.country, dry_run=args.dry_run)

    print(f"\n{'='*50}")
    print(f"Merge Summary: {args.country}")
    print(f"{'='*50}")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
