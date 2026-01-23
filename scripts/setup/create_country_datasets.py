#!/usr/bin/env python3
"""
Create approved_dataset.json for multiple countries from contributions CSV.
"""

import csv
import json
import logging
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# JobId to country mapping
JOBID_TO_COUNTRY = {
    '1': 'korea',
    '2': 'china',
    '12': 'japan',
    '11': 'usa',
    '3': 'nigeria',
    '191': 'general',
    '7': 'mexico',
    '192': 'kenya',
    '51': 'italy',
    '186': 'france',
    '50': 'germany',
    '10': 'india',
    '85': 'canada',
    '176': 'uk',
    '26': 'egypt',
    '13': 'peru',
    '159': 'serbia',
    '14': 'qatar',
    '177': 'singapore',
}

# Reverse mapping
COUNTRY_TO_JOBIDS = defaultdict(list)
for jid, country in JOBID_TO_COUNTRY.items():
    COUNTRY_TO_JOBIDS[country].append(jid)


def create_dataset_for_country(contributions_csv: Path, country: str, output_json: Path):
    """Create approved_dataset.json for a specific country."""

    logger.info(f"Creating dataset for {country}...")

    # Get jobIds for this country
    target_jobids = COUNTRY_TO_JOBIDS.get(country, [])
    if not target_jobids:
        logger.warning(f"No jobId mapping found for {country}")
        return 0

    logger.info(f"Using jobIds: {target_jobids}")

    # Read contributions CSV
    contributions = []
    with open(contributions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = row.get('jobId', '').strip('"').strip()
            review_status = row.get('reviewStatus', '').strip('"').lower()

            # Filter by country (jobId) and status (approved or pending)
            if job_id in target_jobids and review_status in ['approved', 'pending']:
                contributions.append(row)

    logger.info(f"Found {len(contributions)} valid contributions for {country}")

    if len(contributions) == 0:
        logger.warning(f"No contributions found for {country}")
        return 0

    # Convert to dataset format
    items = []
    category_counts = defaultdict(int)

    for contrib in contributions:
        category = contrib.get('category', '').strip('"').strip() or 'general'
        category_lower = category.lower().replace(' & ', '_').replace(' ', '_')

        contrib_id = contrib.get('__id__', '').strip('"').strip()
        image_url = contrib.get('imageURL', '').strip('"').strip()
        description = contrib.get('description', '').strip('"').strip()

        # Generate item ID
        item_id = f"{country}_{category_lower}_{category_counts[category_lower]:04d}"
        category_counts[category_lower] += 1

        # Create item
        item = {
            "id": item_id,
            "category": category_lower,
            "subcategory": None,
            "image_url": image_url,
            "image_path": f"images/{category_lower}/{contrib_id}.jpg",
            "source": "worldccub",
            "contribution_id": contrib_id,
            "upload_date": contrib.get('timestamp', '').strip('"'),
            "quality_score": 0,
            "likes": int(contrib.get('likeCount', '0').strip('"') or 0),
            "points": int(contrib.get('points', '0').strip('"') or 0),
            "tags": [category_lower, country],
            "description": description,
            "description_lang": "auto",
            "cultural_notes": "",
            "key_features": [],
            "common_mistakes": [],
            "verified": True,
            "contributor_id": contrib.get('userId', '').strip('"')
        }

        items.append(item)

    # Create output directory
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Write dataset
    dataset = {"items": items}
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Created {output_json}")
    logger.info(f"  Total items: {len(items)}")
    logger.info(f"  Categories: {dict(category_counts)}")

    return len(items)


def main():
    """Create datasets for all countries with 10+ contributions."""

    data_dir = PROJECT_ROOT / "data"
    contributions_csv = data_dir / "_contributions.csv"

    # Countries to process (10+ valid contributions)
    countries_to_process = [
        'china', 'japan', 'usa', 'nigeria', 'general',
        'mexico', 'kenya', 'italy', 'france', 'germany'
    ]

    logger.info(f"Processing {len(countries_to_process)} countries")
    logger.info("=" * 80)

    for country in countries_to_process:
        output_json = data_dir / "country_packs" / country / "approved_dataset.json"

        try:
            count = create_dataset_for_country(contributions_csv, country, output_json)
            if count > 0:
                logger.info(f"✅ {country}: {count} items")
            else:
                logger.warning(f"⚠️  {country}: No items")
        except Exception as e:
            logger.error(f"❌ {country}: Failed - {e}")

        print()

    logger.info("=" * 80)
    logger.info("Dataset creation completed!")


if __name__ == "__main__":
    main()
