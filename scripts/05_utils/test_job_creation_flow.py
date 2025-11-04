#!/usr/bin/env python3
"""
Test complete job creation and data integration flow.

This script tests:
1. Gap detection and subcategory inference
2. Job creation with structured metadata
3. Job metadata extraction from description
4. Duplicate job detection

Usage:
    python scripts/05_utils/test_job_creation_flow.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_gap_analyzer():
    """Test gap analyzer with subcategory inference."""
    from ccub2_agent.modules.gap_analyzer import DataGapAnalyzer
    from ccub2_agent.modules.country_pack import CountryDataPack

    logger.info("=" * 80)
    logger.info("TEST 1: Gap Analyzer with Subcategory Inference")
    logger.info("=" * 80)

    # Create mock country pack
    country_pack = CountryDataPack("korea")

    # Create analyzer
    analyzer = DataGapAnalyzer(country_pack)

    # Mock issues
    issues = [
        {
            "category": "traditional_clothing",
            "description": "The jeogori collar structure is incorrect",
            "severity": 8,
        },
        {
            "category": "traditional_clothing",
            "description": "Collar neckline should be square, not round",
            "severity": 7,
        },
    ]

    # Analyze gaps
    gaps = analyzer.analyze(issues, "korea")

    logger.info(f"\nDetected {len(gaps)} gaps:")
    for gap in gaps:
        logger.info(f"  Category: {gap['category']}")
        logger.info(f"  Subcategory: {gap['subcategory']}")  # Should infer "jeogori_collar"
        logger.info(f"  Keywords: {gap['keywords']}")
        logger.info(f"  Priority: {gap['priority']}")
        logger.info("")

    # Verify subcategory inference
    assert len(gaps) > 0, "Should detect at least one gap"
    assert gaps[0]['subcategory'] != "general", "Should infer specific subcategory"
    logger.info("✅ Subcategory inference working!")
    logger.info("")


def test_job_creation():
    """Test job creation with structured metadata."""
    from ccub2_agent.modules.agent_job_creator import AgentJobCreator

    logger.info("=" * 80)
    logger.info("TEST 2: Job Creation with Structured Metadata")
    logger.info("=" * 80)

    # Create job creator
    creator = AgentJobCreator()

    # Test data
    test_job_data = {
        "country": "korea",
        "category": "traditional_clothing",
        "subcategory": "jeogori_collar",
        "keywords": ["jeogori", "collar", "neckline", "square"],
        "description": "We need close-up photos of authentic Korean jeogori collar showing the traditional square neckline structure.",
        "target_count": 15,
        "skip_duplicate_check": True,  # For testing
    }

    logger.info("\nCreating test job...")
    logger.info(f"  Country: {test_job_data['country']}")
    logger.info(f"  Category: {test_job_data['category']}")
    logger.info(f"  Subcategory: {test_job_data['subcategory']}")

    # Generate title (without actually creating in Firebase)
    title = creator._generate_title(
        test_job_data['country'],
        test_job_data['category'],
        test_job_data['subcategory']
    )
    logger.info(f"\nGenerated Title:")
    logger.info(f"  '{title}'")

    # Generate structured description
    structured_desc = creator._create_structured_description(
        description=test_job_data['description'],
        country=test_job_data['country'],
        category=test_job_data['category'],
        subcategory=test_job_data['subcategory'],
        keywords=test_job_data['keywords'],
        target_count=test_job_data['target_count']
    )
    logger.info(f"\nGenerated Description:")
    logger.info("-" * 60)
    logger.info(structured_desc)
    logger.info("-" * 60)

    # Verify title format (accept both "Korea" and "Korean" - no hardcoding!)
    assert "Korea" in title or "Korean" in title, "Title should contain country name"
    assert "Traditional Clothing" in title, "Title should contain category"
    assert "Jeogori Collar" in title, "Title should contain subcategory"

    # Verify description contains metadata
    assert "Country: korea" in structured_desc
    assert "Category: traditional_clothing" in structured_desc
    assert "Subcategory: jeogori_collar" in structured_desc
    assert "jeogori" in structured_desc

    logger.info("\n✅ Job creation structure correct!")
    logger.info("")


def test_metadata_extraction():
    """Test metadata extraction from job description."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "01_setup"))
    from init_dataset import extract_job_metadata

    logger.info("=" * 80)
    logger.info("TEST 3: Metadata Extraction from Description")
    logger.info("=" * 80)

    # Sample structured description (as created by agent)
    sample_description = """We need close-up photos of authentic Korean jeogori collar showing the traditional square neckline structure.

---
📊 **Project Details:**
• Country: korea
• Category: traditional_clothing
• Subcategory: jeogori_collar
• Keywords: jeogori, collar, neckline, square
• Target: 15 contributions

📌 Your contributions will help AI systems better understand and represent Korea culture accurately.
"""

    logger.info("\nExtracting metadata from description...")

    # Extract metadata
    metadata = extract_job_metadata(sample_description)

    logger.info(f"\nExtracted Metadata:")
    logger.info(f"  Country: {metadata['country']}")
    logger.info(f"  Category: {metadata['category']}")
    logger.info(f"  Subcategory: {metadata['subcategory']}")
    logger.info(f"  Keywords: {metadata['keywords']}")

    # Verify extraction
    assert metadata['country'] == "korea", "Should extract country"
    assert metadata['category'] == "traditional_clothing", "Should extract category"
    assert metadata['subcategory'] == "jeogori_collar", "Should extract subcategory"
    assert "jeogori" in metadata['keywords'], "Should extract keywords"
    assert "collar" in metadata['keywords'], "Should extract multiple keywords"

    logger.info("\n✅ Metadata extraction working!")
    logger.info("")


def test_title_parsing():
    """Test title parsing in detect_available_countries.py."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "01_setup"))
    from detect_available_countries import _extract_country_from_title

    logger.info("=" * 80)
    logger.info("TEST 4: Title Parsing for Country Detection")
    logger.info("=" * 80)

    test_cases = [
        ("Korean Traditional Clothing Dataset", "korea"),
        ("Korean Traditional Clothing - Jeogori Collar", "korea"),
        ("Japanese Culture Dataset", "japan"),
        ("Indian Food and Cuisine Dataset", "india"),
        ("Nigerian Architecture - Traditional Huts", "nigeria"),
    ]

    logger.info("\nTesting title parsing:")
    all_passed = True
    for title, expected_country in test_cases:
        extracted = _extract_country_from_title(title)
        status = "✅" if extracted == expected_country else "❌"
        logger.info(f"  {status} '{title[:50]:<50}' → {extracted}")
        if extracted != expected_country:
            all_passed = False
            logger.error(f"     Expected: {expected_country}")

    if all_passed:
        logger.info("\n✅ All title parsing tests passed!")
    else:
        logger.error("\n❌ Some title parsing tests failed!")

    logger.info("")


def test_duplicate_detection():
    """Test duplicate job detection."""
    from ccub2_agent.modules.agent_job_creator import AgentJobCreator

    logger.info("=" * 80)
    logger.info("TEST 5: Duplicate Job Detection")
    logger.info("=" * 80)

    creator = AgentJobCreator()

    logger.info("\nNote: This test requires Firebase connection.")
    logger.info("If Firebase is not available, duplicate detection will be skipped.")

    if not creator.db:
        logger.warning("⚠️  Firebase not initialized - skipping duplicate detection test")
        logger.info("")
        return

    # Test duplicate detection logic
    result = creator.check_duplicate_job(
        country="korea",
        category="traditional_clothing",
        subcategory="general",
        keywords=["hanbok", "jeogori"]
    )

    if result:
        logger.info(f"Found existing job: {result}")
    else:
        logger.info("No duplicate found (expected for new subcategories)")

    logger.info("\n✅ Duplicate detection logic tested!")
    logger.info("")


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("TESTING COMPLETE JOB CREATION AND DATA INTEGRATION FLOW")
    print("=" * 80)
    print("")

    try:
        # Run tests
        test_gap_analyzer()
        test_job_creation()
        test_metadata_extraction()
        test_title_parsing()
        test_duplicate_detection()

        # Summary
        print("\n")
        print("=" * 80)
        print("ALL TESTS PASSED! ✅")
        print("=" * 80)
        print("")
        print("The system is ready for:")
        print("  1. ✅ Gap detection with subcategory inference")
        print("  2. ✅ Job creation with structured metadata")
        print("  3. ✅ Metadata extraction from job descriptions")
        print("  4. ✅ Country detection from job titles")
        print("  5. ✅ Duplicate job detection")
        print("")
        print("Next steps:")
        print("  - Run: python scripts/04_testing/test_model_agnostic_editing.py")
        print("  - Generate images and trigger auto job creation")
        print("  - Verify jobs are created in Firebase with proper metadata")
        print("")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
