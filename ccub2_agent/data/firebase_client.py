"""
Firebase Client

CSV-based data client with optional Firebase admin integration.
Reads _contributions.csv and _jobs.csv as the primary data source.
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Data directory (where CSV files live)
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Category normalization map: raw CSV values -> canonical categories
CATEGORY_NORMALIZE: Dict[str, str] = {
    "": "uncategorized",
    "0": "uncategorized",
    # City
    "City": "city_street",
    "City & Street": "city_street",
    # Nature
    "Nature": "nature_landscape",
    "Nature & Landscape": "nature_landscape",
    "Sunset/Sunrise": "nature_landscape",
    "Travel": "nature_landscape",
    "Flowers": "nature_landscape",
    # Food
    "Food & Drink": "food_drink",
    # Architecture
    "Architecture": "architecture",
    # People
    "People & Action": "people_action",
    # Clothing
    "Clothing": "traditional_clothing",
    "Fashion": "traditional_clothing",
    # Religion / Festival
    "Religion & Festival": "religion_festival",
    "Events": "religion_festival",
    # Arts
    "Dance, Music & Visual Arts": "arts",
    # Entertainment
    "Entertainment": "entertainment",
    "Funny": "entertainment",
    # Sports
    "Sports": "sports",
    # Daily life
    "Utensils & Tools": "daily_life",
    "Vehicles": "daily_life",
    "Hobbies": "daily_life",
    # Animals
    "Animals": "animals",
}


def normalize_category(raw: str) -> str:
    """Normalize a raw category string to a canonical category."""
    return CATEGORY_NORMALIZE.get(raw.strip(), raw.strip().lower().replace(" ", "_"))


def _extract_country_from_title(title: str) -> Optional[str]:
    """Extract country name from job title like 'Korea Culture Dataset'."""
    m = re.match(r"^(.+?)\s+Culture\s+Dataset", title, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower().replace(" ", "_")
    return None


class FirebaseClient:
    """CSV-based data client (Firebase admin optional)."""

    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = data_dir or DATA_DIR
        self._contributions: Optional[List[Dict[str, Any]]] = None
        self._jobs: Optional[List[Dict[str, Any]]] = None
        self._job_country_map: Optional[Dict[str, str]] = None
        logger.info(f"FirebaseClient initialized (data_dir={self._data_dir})")

    # ------------------------------------------------------------------
    # Internal CSV loading (lazy)
    # ------------------------------------------------------------------

    def _load_contributions(self) -> List[Dict[str, Any]]:
        if self._contributions is not None:
            return self._contributions
        csv_path = self._data_dir / "_contributions.csv"
        if not csv_path.exists():
            logger.warning(f"Contributions CSV not found: {csv_path}")
            self._contributions = []
            return self._contributions
        rows: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["category_raw"] = row.get("category", "")
                row["category_normalized"] = normalize_category(row.get("category", ""))
                row["status"] = row.get("reviewStatus", "pending")
                rows.append(row)
        self._contributions = rows
        logger.info(f"Loaded {len(rows)} contributions from CSV")
        return self._contributions

    def _load_jobs(self) -> List[Dict[str, Any]]:
        if self._jobs is not None:
            return self._jobs
        csv_path = self._data_dir / "_jobs.csv"
        if not csv_path.exists():
            logger.warning(f"Jobs CSV not found: {csv_path}")
            self._jobs = []
            return self._jobs
        rows: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["country"] = _extract_country_from_title(row.get("title", ""))
                rows.append(row)
        self._jobs = rows
        logger.info(f"Loaded {len(rows)} jobs from CSV")
        return self._jobs

    def _get_job_country_map(self) -> Dict[str, str]:
        """Build jobId -> country mapping from jobs CSV."""
        if self._job_country_map is not None:
            return self._job_country_map
        jobs = self._load_jobs()
        self._job_country_map = {}
        for job in jobs:
            job_id = job.get("__id__", "")
            country = job.get("country")
            if job_id and country:
                self._job_country_map[job_id] = country
        logger.info(f"Built job→country map: {len(self._job_country_map)} entries")
        return self._job_country_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_contributions(
        self,
        country: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get contributions, optionally filtered by country and/or status."""
        all_contribs = self._load_contributions()
        job_map = self._get_job_country_map()

        result = []
        for c in all_contribs:
            # Resolve country from jobId
            job_id = c.get("jobId", "")
            c_country = job_map.get(str(job_id))

            if country and c_country != country.lower().replace(" ", "_"):
                continue
            if status and c.get("status") != status:
                continue

            c["country"] = c_country
            result.append(c)
        return result

    def get_jobs(self, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get jobs, optionally filtered by country."""
        all_jobs = self._load_jobs()
        if country is None:
            return all_jobs
        target = country.lower().replace(" ", "_")
        return [j for j in all_jobs if j.get("country") == target]

    def create_job(self, **kwargs) -> str:
        """Create a job locally (saves to JSON file)."""
        jobs_dir = self._data_dir / "agent_jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)

        import uuid
        job_id = f"agent_{uuid.uuid4().hex[:8]}"
        job_data = {"job_id": job_id, **kwargs}

        job_path = jobs_dir / f"{job_id}.json"
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Created local job: {job_path}")
        return job_id

    def download_image(self, url: str, dest_path: Path) -> Path:
        """Download an image from URL to dest_path. Skips if already exists."""
        dest_path = Path(dest_path)
        if dest_path.exists():
            return dest_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import requests
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            logger.debug(f"Downloaded: {dest_path.name}")
        except Exception as e:
            logger.error(f"Download failed ({url}): {e}")
            raise

        return dest_path

    def get_all_countries(self) -> List[str]:
        """Get list of all countries from jobs."""
        jobs = self._load_jobs()
        countries = sorted(set(j["country"] for j in jobs if j.get("country")))
        return countries


# Singleton
_client_instance: Optional[FirebaseClient] = None


def get_firebase_client() -> FirebaseClient:
    """Get Firebase client singleton instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = FirebaseClient()
    return _client_instance
