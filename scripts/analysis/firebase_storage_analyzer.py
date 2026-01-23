#!/usr/bin/env python3
"""
Firebase Storage Structure Analyzer

Analyzes and reports on the structure of Firebase Storage for WorldCCUB project.
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ccub2_agent.modules.firebase_client import FirebaseClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_storage():
    """Analyze Firebase Storage structure."""
    print("=" * 60)
    print("Firebase Storage Structure Analyzer")
    print("=" * 60)
    
    # Initialize client
    try:
        client = FirebaseClient()
        print(f"✓ Connected to Firebase project: {client.project_id}")
        print(f"✓ Storage bucket: {client.storage_bucket}\n")
    except Exception as e:
        print(f"✗ Failed to initialize Firebase client: {e}")
        return
    
    # Get storage structure
    print("Analyzing storage structure...")
    structure = client.get_storage_structure()
    
    # Print summary
    print("\n" + "=" * 60)
    print("STORAGE SUMMARY")
    print("=" * 60)
    
    # Contributions
    contrib = structure["contributions"]
    print(f"\n📸 Contributions: {contrib['total_files']} files")
    if contrib["by_job"]:
        print(f"   Jobs: {len(contrib['by_job'])}")
        top_jobs = sorted(contrib["by_job"].items(), key=lambda x: x[1], reverse=True)[:10]
        for job_id, count in top_jobs:
            print(f"   - {job_id}: {count} images")
    
    # Notices
    notices = structure["notices"]
    print(f"\n📢 Notices: {notices['total_files']} files")
    
    # Qualification tests
    tests = structure["qualification_tests"]
    print(f"\n📝 Qualification Tests: {tests['total_files']} files")
    if tests["by_job"]:
        print(f"   Jobs: {len(tests['by_job'])}")
        for job_id, count in tests["by_job"].items():
            print(f"   - {job_id}: {count} images")
    
    # Other
    other = structure["other"]
    print(f"\n📁 Other: {other['total_files']} files")
    if other["paths"]:
        print("   Example paths:")
        for path in other["paths"][:5]:
            print(f"   - {path}")
    
    # Total
    total = (
        contrib["total_files"] +
        notices["total_files"] +
        tests["total_files"] +
        other["total_files"]
    )
    print(f"\n📊 Total files: {total}")
    
    # Save report
    report_path = PROJECT_ROOT / "data" / "firebase_storage_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Sample contribution images
    if contrib["by_job"]:
        print("\n" + "=" * 60)
        print("SAMPLE CONTRIBUTION IMAGES")
        print("=" * 60)
        
        sample_job = list(contrib["by_job"].keys())[0]
        print(f"\nJob: {sample_job}")
        
        images = client.get_contribution_images(sample_job, limit=5)
        for i, img in enumerate(images, 1):
            print(f"\n{i}. User: {img['user_id']}")
            print(f"   Filename: {img['filename']}")
            print(f"   Path: {img['storage_path']}")
            if img['download_url']:
                print(f"   URL: {img['download_url'][:80]}...")


if __name__ == "__main__":
    analyze_storage()
