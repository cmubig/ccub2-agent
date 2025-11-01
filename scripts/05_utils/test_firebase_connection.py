#!/usr/bin/env python3
"""
Test Firebase connection and data retrieval.

Usage:
    python scripts/05_utils/test_firebase_connection.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.firebase_client import get_firebase_client


def test_firebase_connection():
    """Test Firebase connection and basic operations."""
    print("=" * 80)
    print("Firebase Connection Test")
    print("=" * 80)
    print()

    try:
        # Initialize Firebase client
        print("1️⃣  Initializing Firebase client...")
        firebase = get_firebase_client()
        print("   ✅ Firebase client initialized")
        print()

        # Test: Get jobs
        print("2️⃣  Fetching jobs...")
        jobs = firebase.get_jobs()
        print(f"   ✅ Fetched {len(jobs)} jobs")

        if jobs:
            print()
            print("   Sample jobs:")
            for i, job in enumerate(jobs[:3], 1):
                print(f"      {i}. Job {job.get('__id__')}: {job.get('title')}")
                print(f"         Country: {job.get('country')}, Category: {job.get('category')}")
        print()

        # Test: Get contributions
        print("3️⃣  Fetching contributions...")
        contributions = firebase.get_contributions(limit=100)
        print(f"   ✅ Fetched {len(contributions)} contributions (limit: 100)")

        if contributions:
            print()
            print("   Sample contributions:")
            for i, contrib in enumerate(contributions[:3], 1):
                print(f"      {i}. Job {contrib.get('jobId')}: {contrib.get('description')[:50]}...")
                print(f"         Category: {contrib.get('category')}")
        print()

        # Test: Count by country
        print("4️⃣  Counting contributions by country...")

        # Group by jobId
        from collections import defaultdict
        job_counts = defaultdict(int)
        for contrib in contributions:
            job_id = contrib.get('jobId')
            if job_id:
                job_counts[job_id] += 1

        # Map to countries
        job_to_country = {job.get('__id__'): job.get('country') for job in jobs}
        country_counts = defaultdict(int)
        for job_id, count in job_counts.items():
            country = job_to_country.get(job_id, 'unknown')
            country_counts[country] += count

        print()
        for country, count in sorted(country_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"      • {country}: {count} contributions")
        print()

        # Summary
        print("=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        print()
        print("Firebase is correctly configured and accessible.")
        print("You can now run init_dataset.py with Firebase integration!")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Test failed!")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check if .firebase_config.json exists in project root")
        print("2. Verify Firebase project ID and credentials")
        print("3. If using Admin SDK, ensure service account key is available")
        print("4. Check internet connection")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_firebase_connection()
