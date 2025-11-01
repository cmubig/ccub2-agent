#!/usr/bin/env python3
"""
Detect available countries from contributions.csv and _jobs.csv
"""

import csv
from pathlib import Path
from collections import defaultdict
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_jobid_mapping(jobs_csv: Path, use_firebase: bool = True) -> dict:
    """
    Load jobId to country mapping from Firebase or _jobs.csv.

    This function dynamically generates the mapping, so new countries
    are automatically supported when added to Firebase/jobs collection.

    Args:
        jobs_csv: Path to _jobs.csv (fallback if Firebase fails)
        use_firebase: Use Firebase directly instead of CSV (default: True)

    Returns:
        dict: {jobId: country_name} mapping
    """
    jobid_to_country = {}

    # Try Firebase first
    if use_firebase:
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from ccub2_agent.modules.firebase_client import get_firebase_client

            firebase = get_firebase_client()
            jobs = firebase.get_jobs()

            for job in jobs:
                job_id = job.get('__id__', '')
                title = job.get('title', '')

                if not job_id or not title:
                    continue

                # Extract country name from title
                country = _extract_country_from_title(title)
                if country:
                    jobid_to_country[job_id] = country

            if jobid_to_country:
                print(f"Loaded {len(jobid_to_country)} job mappings from Firebase")
                return jobid_to_country

        except Exception as e:
            print(f"Failed to load from Firebase: {e}")
            print("Falling back to CSV...")

    # Fallback to CSV
    if not jobs_csv.exists():
        print("No jobs.csv found, using fallback mapping")
        return _get_fallback_mapping()

    with open(jobs_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = row.get('__id__', '').strip('"').strip()
            title = row.get('title', '').strip('"').strip()

            if not job_id or not title:
                continue

            # Extract country name from title
            country = _extract_country_from_title(title)
            if country:
                jobid_to_country[job_id] = country

    return jobid_to_country


def _extract_country_from_title(title: str) -> str:
    """
    Extract country name from job title.

    Examples:
        "Korea Culture Dataset" -> "korea"
        "China Culture Dataset" -> "china"
        "Funny Cat" -> "general"
        "British Culture Dataset" -> "uk"
    """
    title_lower = title.lower()

    # Special cases (including adjective forms)
    special_cases = {
        'funny cat': 'general',
        'british': 'uk',
        'united states': 'usa',
        'japanese': 'japan',
        'nigerian': 'nigeria',
        'mexican': 'mexico',
        'kenyan': 'kenya',
        'italian': 'italy',
        'french': 'france',
        'german': 'germany',
        'indian': 'india',
        'canadian': 'canada',
        'egyptian': 'egypt',
        'peruvian': 'peru',
        'serbian': 'serbia',
        'qatari': 'qatar',
        'singaporean': 'singapore',
        'chinese': 'china',
        'korean': 'korea',
    }

    for pattern, country in special_cases.items():
        if pattern in title_lower:
            return country

    # Standard pattern: "[Country] Culture Dataset"
    match = re.match(r'(\w+)\s+culture\s+dataset', title_lower)
    if match:
        return match.group(1)

    # Fallback: first word
    words = title_lower.split()
    if words:
        return words[0]

    return 'unknown'


def _get_fallback_mapping() -> dict:
    """
    Fallback mapping if _jobs.csv is not available.
    Based on current known jobIds from contributions.csv analysis.
    """
    return {
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


# Load mapping dynamically at module import
JOBID_TO_COUNTRY = load_jobid_mapping(PROJECT_ROOT / "data" / "_jobs.csv")


def detect_available_countries(contributions_csv: Path, use_firebase: bool = True) -> dict:
    """
    Detect which countries have contributions available.

    This function automatically detects all countries present in
    Firebase or contributions.csv without needing code changes when
    new countries are added.

    Args:
        contributions_csv: Path to CSV (fallback if Firebase fails)
        use_firebase: Use Firebase directly instead of CSV (default: True)

    Returns:
        dict: {country: count} mapping
    """
    country_counts = defaultdict(int)

    # Try Firebase first
    if use_firebase:
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from ccub2_agent.modules.firebase_client import get_firebase_client

            firebase = get_firebase_client()
            contributions = firebase.get_contributions()

            for contrib in contributions:
                jobId = contrib.get('jobId', '').strip()

                if jobId in JOBID_TO_COUNTRY:
                    country = JOBID_TO_COUNTRY[jobId]
                    country_counts[country] += 1

            if country_counts:
                return dict(country_counts)

        except Exception as e:
            print(f"Failed to detect from Firebase: {e}")
            print("Falling back to CSV...")

    # Fallback to CSV
    if not contributions_csv.exists():
        return {}

    with open(contributions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            jobId = row.get('jobId', '').strip('"').strip()

            if jobId in JOBID_TO_COUNTRY:
                country = JOBID_TO_COUNTRY[jobId]
                country_counts[country] += 1

    return dict(country_counts)


def get_country_display_name(country: str) -> str:
    """
    Get display name for country.

    Falls back to capitalized country name if not in mapping.
    """
    display_names = {
        'korea': 'Korea (한국)',
        'japan': 'Japan (日本)',
        'china': 'China (中国)',
        'india': 'India (भारत)',
        'usa': 'USA',
        'nigeria': 'Nigeria',
        'kenya': 'Kenya',
        'mexico': 'Mexico',
        'italy': 'Italy',
        'france': 'France',
        'germany': 'Germany',
        'egypt': 'Egypt',
        'canada': 'Canada',
        'uk': 'United Kingdom',
        'peru': 'Peru',
        'serbia': 'Serbia',
        'qatar': 'Qatar',
        'singapore': 'Singapore',
        'general': 'General/Mixed',
    }
    return display_names.get(country, country.capitalize())


if __name__ == "__main__":
    contributions_csv = PROJECT_ROOT / "data" / "_contributions.csv"

    if contributions_csv.exists():
        countries = detect_available_countries(contributions_csv)

        print("Available countries in contributions.csv:")
        print()
        for country, count in sorted(countries.items(), key=lambda x: -x[1]):
            display = get_country_display_name(country)
            print(f"  {display}: {count} contributions")
        print()
        print(f"Total: {len(countries)} countries")
    else:
        print("contributions.csv not found")
