#!/usr/bin/env python3
"""Validate licenses for all curated images in a country.

Usage:
    python scripts/curation/02_validate_licenses.py --country korea
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ccub2_agent.data.curation.license_validator import LicenseValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate curated image licenses")
    parser.add_argument("--country", required=True, help="Country code")
    parser.add_argument("--curated-dir", default=None, help="Curated data directory")
    args = parser.parse_args()

    curated_dir = Path(args.curated_dir) if args.curated_dir else Path(__file__).parents[2] / "data" / "curated"
    validator = LicenseValidator(curated_dir)
    report = validator.validate_country(args.country)

    print(f"\n{'='*50}")
    print(f"License Validation Report: {args.country}")
    print(f"{'='*50}")
    print(f"Total:   {report.total}")
    print(f"Valid:   {report.valid}")
    print(f"Invalid: {report.invalid}")
    print(f"Rate:    {report.pass_rate:.0%}")

    if report.invalid > 0:
        print(f"\nInvalid records:")
        for r in report.results:
            if not r.valid:
                print(f"  - {r.image_id}: {', '.join(r.errors)}")


if __name__ == "__main__":
    main()
