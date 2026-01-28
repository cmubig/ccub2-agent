#!/usr/bin/env python3
"""Download curated cultural images from open-license platforms.

Usage:
    python scripts/curation/01_download_curated.py --country korea --category food --limit 10
    python scripts/curation/01_download_curated.py --country korea --limit 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ccub2_agent.data.curation.wikimedia_downloader import WikimediaDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download curated cultural images")
    parser.add_argument("--country", required=True, help="Country code (e.g., korea, japan)")
    parser.add_argument("--category", default=None, help="Cultural category (e.g., food, clothing)")
    parser.add_argument("--limit", type=int, default=50, help="Max images to download")
    parser.add_argument("--platform", default="wikimedia", choices=["wikimedia"], help="Source platform")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parents[2] / "data" / "curated"

    if args.platform == "wikimedia":
        downloader = WikimediaDownloader(
            output_dir=output_dir,
            country=args.country,
            category=args.category,
            limit=args.limit,
            delay=args.delay,
        )
    else:
        logger.error(f"Unsupported platform: {args.platform}")
        sys.exit(1)

    logger.info(f"Starting download: {args.country}/{args.category or 'all'} (limit={args.limit})")
    results = downloader.run()

    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    logger.info(f"Complete: {success} downloaded, {failed} failed")


if __name__ == "__main__":
    main()
