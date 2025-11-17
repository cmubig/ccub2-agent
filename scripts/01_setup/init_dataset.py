#!/usr/bin/env python3
"""
Initialize dataset from contributions.

This script sets up the complete cultural knowledge base:
1. Converts contributions.csv to approved_dataset.json
2. Downloads images from Firebase
3. Enhances captions with VLM
4. Extracts cultural knowledge
5. Builds RAG indices (text + CLIP)

Run once after cloning the repository.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_status(data_dir: Path, country: str) -> dict:
    """Check what needs to be initialized."""
    country_pack_dir = data_dir / "country_packs" / country

    status = {
        'contributions_csv': (PROJECT_ROOT / "data" / "_contributions.csv").exists(),
        'approved_dataset': (country_pack_dir / "approved_dataset.json").exists(),
        'images_dir': (country_pack_dir / "images").exists() and
                      len(list((country_pack_dir / "images").glob("**/*.jpg"))) > 0,
        'enhanced_dataset': (country_pack_dir / "approved_dataset_enhanced.json").exists(),
        'cultural_knowledge': (data_dir / "cultural_knowledge" / f"{country}_knowledge.json").exists(),
        'text_index': (data_dir / "cultural_index" / country / "faiss.index").exists(),
        'clip_index': (data_dir / "clip_index" / country / "clip.index").exists(),
    }

    # Check Firebase connectivity
    firebase_status = "unavailable"
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from ccub2_agent.modules.firebase_client import get_firebase_client
        firebase = get_firebase_client()
        if firebase.use_admin_sdk:
            firebase_status = "admin_sdk"
        else:
            firebase_status = "rest_api"
    except Exception:
        firebase_status = "unavailable"

    status['firebase_status'] = firebase_status

    return status


def print_status(status: dict):
    """Print current initialization status."""
    print("")
    print("="*80)
    print("DATASET INITIALIZATION STATUS")
    print("="*80)

    # Display Firebase connection status
    firebase_status = status.get('firebase_status', 'unavailable')
    if firebase_status == "admin_sdk":
        firebase_icon = "✅"
        firebase_msg = "Firebase Admin SDK (full access)"
    elif firebase_status == "rest_api":
        firebase_icon = "⚠️"
        firebase_msg = "Firebase REST API (read-only fallback)"
    else:
        firebase_icon = "❌"
        firebase_msg = "Firebase unavailable (CSV fallback)"

    print(f"\n🔥 Data Source: {firebase_icon} {firebase_msg}")
    if firebase_status == "unavailable":
        print("   ℹ️  To enable Firebase: See FIREBASE_SETUP.md")
    print("")

    steps = [
        ('contributions_csv', '1. Contributions CSV', 'Fallback data source'),
        ('approved_dataset', '2. Approved Dataset JSON', 'Auto-generated'),
        ('images_dir', '3. Images Downloaded', 'From Firebase'),
        ('enhanced_dataset', '4. Enhanced Captions', 'VLM processing'),
        ('cultural_knowledge', '5. Cultural Knowledge', 'VLM extraction'),
        ('text_index', '6. Text RAG Index', 'FAISS index'),
        ('clip_index', '7. CLIP Image Index', 'Image similarity'),
    ]

    for key, name, desc in steps:
        status_icon = "✅" if status[key] else "❌"
        print(f"{status_icon} {name:<30} ({desc})")

    print("="*80)
    print("")


def extract_job_metadata(job_description: str) -> dict:
    """
    Extract metadata from job description.

    Job descriptions created by agent have this format:
    ---
    📊 **Project Details:**
    • Country: korea
    • Category: traditional_clothing
    • Subcategory: jeogori_collar
    • Keywords: jeogori, collar, neckline
    • Target: 15 contributions
    ---

    Args:
        job_description: Job description string

    Returns:
        dict with: country, category, subcategory, keywords
    """
    import re

    metadata = {
        "country": None,
        "category": None,
        "subcategory": "general",
        "keywords": []
    }

    if not job_description:
        return metadata

    # Parse metadata section
    for line in job_description.split('\n'):
        line = line.strip()

        # Country (use word boundary to avoid matching in other words)
        match = re.search(r'\bCountry:\s*(\w+)', line, re.IGNORECASE)
        if match:
            metadata["country"] = match.group(1)

        # Category (must NOT be Subcategory!)
        match = re.search(r'^•?\s*Category:\s*([\w_]+)', line, re.IGNORECASE)
        if match and "subcategory" not in line.lower():
            metadata["category"] = match.group(1)

        # Subcategory
        match = re.search(r'\bSubcategory:\s*([\w_]+)', line, re.IGNORECASE)
        if match:
            metadata["subcategory"] = match.group(1)

        # Keywords
        match = re.search(r'\bKeywords:\s*(.+)', line, re.IGNORECASE)
        if match:
            keywords_str = match.group(1)
            # Remove "N/A" and split by comma
            if "N/A" not in keywords_str:
                metadata["keywords"] = [k.strip() for k in keywords_str.split(',')]

    return metadata


def convert_contributions_to_dataset(contributions_csv: Path, output_json: Path, country: str, use_firebase: bool = True):
    """
    Convert contributions to approved_dataset.json format.

    INCREMENTAL UPDATE: Preserves existing items and only adds new ones.

    Args:
        contributions_csv: Path to CSV (fallback if Firebase fails)
        output_json: Output JSON path
        country: Target country
        use_firebase: Use Firebase directly instead of CSV (default: True)
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "01_setup"))
    from detect_available_countries import JOBID_TO_COUNTRY

    logger.info(f"Converting contributions to approved_dataset.json for {country}...")

    # Load job information for metadata extraction
    jobs_dict = {}
    if use_firebase:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from ccub2_agent.modules.firebase_client import get_firebase_client
            firebase = get_firebase_client()
            jobs = firebase.get_jobs()
            jobs_dict = {job['__id__']: job for job in jobs}
            logger.info(f"Loaded {len(jobs_dict)} jobs for metadata extraction")
        except Exception as e:
            logger.warning(f"Failed to load jobs: {e}")

    # Load existing dataset if available (for incremental update)
    existing_items = []
    existing_urls = set()
    if output_json.exists():
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_items = existing_data.get('items', [])
                existing_urls = {item.get('image_url') for item in existing_items}
            logger.info(f"Found {len(existing_items)} existing items, will add only new ones")
        except Exception as e:
            logger.warning(f"Could not load existing dataset: {e}")

    # Reverse mapping: country -> jobIds
    target_jobids = [jid for jid, c in JOBID_TO_COUNTRY.items() if c == country]

    if not target_jobids:
        logger.warning(f"No jobId mapping found for country '{country}', processing all contributions")
        target_jobids = None

    # Get contributions from Firebase or CSV
    contributions_data = []

    if use_firebase:
        try:
            logger.info("Fetching contributions from Firebase...")
            sys.path.insert(0, str(PROJECT_ROOT))
            from ccub2_agent.modules.firebase_client import get_firebase_client

            firebase = get_firebase_client()
            contributions_data = firebase.get_contributions(country=country)
            logger.info(f"Fetched {len(contributions_data)} contributions from Firebase")

        except Exception as e:
            logger.warning(f"Firebase fetch failed: {e}")
            logger.info("Falling back to CSV...")
            use_firebase = False

    if not use_firebase:
        # Fallback to CSV
        import csv
        logger.info(f"Reading from CSV: {contributions_csv}")
        with open(contributions_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            contributions_data = list(reader)
            logger.info(f"Read {len(contributions_data)} contributions from CSV")

    # Process contributions
    new_items = []
    item_count = len(existing_items)  # Continue numbering from existing items

    for row in contributions_data:
        jobId = row.get('jobId', '').strip('"').strip()

        # Filter by country if we have jobId mapping
        if target_jobids and jobId not in target_jobids:
            continue

        image_url = row.get('imageURL', '')

        # Skip if already exists (incremental update)
        if image_url in existing_urls:
            continue

        # Extract metadata from job description if available
        metadata = {"subcategory": "general", "keywords": []}
        if jobId in jobs_dict:
            job = jobs_dict[jobId]
            job_description = job.get('description', '')
            metadata = extract_job_metadata(job_description)

        # Use extracted metadata or fallback to row data
        category = metadata.get("category") or row.get('category', 'general')
        subcategory = metadata.get("subcategory", "general")

        # Map columns properly
        item = {
            'id': f"{country}_{category.replace(' & ', '_').replace(' ', '_').lower()}_{item_count:04d}",
            'category': category,
            'image_url': image_url,
            'caption': row.get('description', ''),
            'quality_score': min(5, max(1, int(row.get('likeCount', '0') or 0))),
            'jobId': jobId,
            # NEW: metadata for subcategory tracking
            '_metadata': {
                'country': country,
                'category': category,
                'subcategory': subcategory,
                'keywords': metadata.get("keywords", []),
                'jobId': jobId
            }
        }
        new_items.append(item)
        item_count += 1

    # Combine existing + new items
    all_items = existing_items + new_items

    # Save
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({'items': all_items}, f, indent=2, ensure_ascii=False)

    if new_items:
        logger.info(f"✓ Added {len(new_items)} new items (total: {len(all_items)}) for {country}")
    else:
        logger.info(f"✓ No new items to add (existing: {len(all_items)})")

    return len(new_items)


def run_initialization(data_dir: Path, country: str, status: dict, skip_images: bool = False, use_firebase: bool = True):
    """Run missing initialization steps."""
    import subprocess

    country_pack_dir = data_dir / "country_packs" / country

    print("")
    print("="*80)
    print("RUNNING INITIALIZATION")
    print("="*80)
    print("")

    # Step 1: Check contributions.csv (only if not using Firebase)
    if not use_firebase and not status['contributions_csv']:
        logger.error("❌ data/_contributions.csv not found!")
        logger.error("Please add your contributions CSV file first.")
        return False

    # Step 2: Convert to approved_dataset.json (ALWAYS RUN - incremental update)
    if use_firebase:
        logger.info("▶ Step 2: Fetching latest data from Firebase...")
    else:
        logger.info("▶ Step 2: Loading data from local CSV...")

    new_items_added = convert_contributions_to_dataset(
        PROJECT_ROOT / "data" / "_contributions.csv",
        country_pack_dir / "approved_dataset.json",
        country,
        use_firebase=use_firebase
    )
    print("")

    # If no new items and everything exists, we're done
    if new_items_added == 0 and all([
        status['enhanced_dataset'],
        status['cultural_knowledge'],
        status['text_index'],
        status['clip_index']
    ]):
        logger.info("✅ No new data to process. All components up-to-date!")
        return True

    # Step 3: Download images (run if new items OR images missing)
    if (new_items_added > 0 or not status['images_dir']) and not skip_images:
        logger.info("▶ Step 3: Downloading images from Firebase...")
        logger.info("Note: This requires Firebase configuration")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "05_utils" / "download_country_images.py"),
            "--country", country,
            "--output-dir", str(country_pack_dir / "images")
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Images downloaded")
        except subprocess.CalledProcessError:
            logger.warning("⚠ Image download failed - you may need to add images manually")
        except FileNotFoundError:
            logger.warning("⚠ Download script not found - skipping image download")
        print("")
    elif skip_images:
        logger.info("⊘ Skipping image download (--skip-images)")
        print("")

    # Step 4: Enhance captions with VLM (run if new items OR not enhanced yet)
    if new_items_added > 0 or not status['enhanced_dataset']:
        logger.info("▶ Step 4: Enhancing captions with VLM...")
        logger.info("This may take 10-30 minutes depending on dataset size")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "02_data_processing" / "enhance_captions.py"),
            "--dataset", str(country_pack_dir / "approved_dataset.json"),
            "--images-dir", str(country_pack_dir / "images"),
            "--output", str(country_pack_dir / "approved_dataset_enhanced.json"),
            "--country", country
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Captions enhanced")
        except Exception as e:
            logger.error(f"❌ Caption enhancement failed: {e}")
            return False
        print("")

    # Step 5: Extract cultural knowledge (run if new items OR not extracted yet)
    if new_items_added > 0 or not status['cultural_knowledge']:
        logger.info("▶ Step 5: Extracting cultural knowledge from images...")
        logger.info("This may take 1-3 hours depending on dataset size")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "02_data_processing" / "extract_cultural_knowledge.py"),
            "--data-dir", str(country_pack_dir),
            "--output", str(data_dir / "cultural_knowledge" / f"{country}_knowledge.json")
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Cultural knowledge extracted")
        except Exception as e:
            logger.error(f"❌ Knowledge extraction failed: {e}")
            return False
        print("")

    # Step 6: Build text RAG index (run if new items OR not built yet)
    if new_items_added > 0 or not status['text_index']:
        logger.info("▶ Step 6: Building text RAG index...")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "03_indexing" / "integrate_knowledge_to_rag.py"),
            "--knowledge-file", str(data_dir / "cultural_knowledge" / f"{country}_knowledge.json"),
            "--index-dir", str(data_dir / "cultural_index" / country)
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ Text RAG index built")
        except Exception as e:
            logger.error(f"❌ Text index building failed: {e}")
            return False
        print("")

    # Step 7: Build CLIP image index (run if new items OR not built yet)
    if new_items_added > 0 or not status['clip_index']:
        logger.info("▶ Step 7: Building CLIP image index...")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "03_indexing" / "build_clip_image_index.py"),
            "--data-dir", str(country_pack_dir),
            "--output-dir", str(data_dir / "clip_index" / country)
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info("✓ CLIP image index built")
        except Exception as e:
            logger.error(f"❌ CLIP index building failed: {e}")
            return False
        print("")

    logger.info("")
    logger.info("="*80)
    logger.info("INITIALIZATION COMPLETE! ✅")
    logger.info("="*80)
    logger.info("")
    logger.info("You can now run:")
    logger.info("  python scripts/04_testing/test_model_agnostic_editing.py")
    logger.info("")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Initialize dataset and knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full initialization (recommended for first-time setup)
  python scripts/01_setup/init_dataset.py --country <country_name>

  # Skip image download (if you already have images)
  python scripts/01_setup/init_dataset.py --country <country_name> --skip-images

  # Check status only
  python scripts/01_setup/init_dataset.py --country <country_name> --check-only
        """
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name (e.g., korea, japan, china)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=PROJECT_ROOT / "data",
        help='Data directory (default: PROJECT_ROOT/data)'
    )
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='Skip image download step'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check status, do not run initialization'
    )

    args = parser.parse_args()

    # Check status
    status = check_status(args.data_dir, args.country)
    print_status(status)

    # Check if everything is already initialized
    if all(status.values()):
        logger.info("✅ All components are already initialized!")
        logger.info("You can run: python scripts/04_testing/test_model_agnostic_editing.py")
        return

    # Check-only mode
    if args.check_only:
        logger.info("Run without --check-only to initialize missing components")
        return

    # Ask about data source
    use_firebase = False
    firebase_status = status.get('firebase_status', 'unavailable')

    if firebase_status in ['admin_sdk', 'rest_api']:
        print("")
        print("📊 Data Source Selection")
        print("="*80)
        print("")
        print("Firebase is available. Would you like to fetch the latest data from Firebase?")
        print("")
        print("  [Y] Yes - Fetch latest contributions from Firebase (recommended)")
        print("  [N] No  - Use existing local CSV file (_contributions.csv)")
        print("")

        choice = input("Fetch from Firebase? [Y/n]: ").strip().lower()
        use_firebase = not choice or choice in ['y', 'yes']

        if use_firebase:
            print("✓ Will fetch latest data from Firebase")
        else:
            print("✓ Will use local CSV file")
            # Check if CSV exists when user chooses CSV mode
            if not status['contributions_csv']:
                logger.error("❌ data/_contributions.csv not found!")
                logger.error("Please add your contributions CSV file or choose Firebase.")
                return
        print("")
    else:
        print("")
        print("ℹ️  Firebase is not available. Will use local CSV file (_contributions.csv)")
        if not status['contributions_csv']:
            logger.error("❌ data/_contributions.csv not found!")
            logger.error("Please add your contributions CSV file first.")
            return
        print("")

    # Ask for confirmation
    print("This will initialize all missing components.")
    print("Depending on your dataset size, this may take 2-5 hours.")
    print("")
    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("Cancelled.")
        return

    # Run initialization
    success = run_initialization(args.data_dir, args.country, status, args.skip_images, use_firebase)

    if not success:
        logger.error("Initialization failed. Please check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
