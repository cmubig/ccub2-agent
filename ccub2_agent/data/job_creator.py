"""
Agent Job Creator

Creates WorldCCUB collection jobs as local JSON files.
Used by JobAgent to convert detected gaps into actionable collection tasks.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Default output directory for agent-created jobs
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "agent_jobs"


class AgentJobCreator:
    """Creates and manages local collection jobs."""

    def __init__(self, output_dir: Optional[Path] = None):
        self._output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AgentJobCreator initialized (output={self._output_dir})")

    def create_job(
        self,
        country: str,
        category: str,
        missing_elements: List[str],
        target_count: int = 20,
    ) -> str:
        """
        Create a collection job.

        Args:
            country: Target country.
            category: Target category.
            missing_elements: List of missing cultural elements.
            target_count: Number of images to collect.

        Returns:
            Job ID string.
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job_data = {
            "job_id": job_id,
            "country": country,
            "category": category,
            "missing_elements": missing_elements,
            "target_count": target_count,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "description": (
                f"Collect {target_count} images of {category} for {country}. "
                f"Missing elements: {', '.join(missing_elements[:5]) if missing_elements else 'general coverage'}"
            ),
        }

        job_path = self._output_dir / f"{job_id}.json"
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Created job {job_id}: {category} for {country} (target={target_count})")
        return job_id

    def create_jobs_from_gaps(
        self,
        gaps: List[Dict[str, Any]],
        country: str,
    ) -> List[str]:
        """
        Create jobs from a list of gap analysis results.

        Args:
            gaps: List of gap dicts (from DataGapAnalyzer).
            country: Country name.

        Returns:
            List of created job IDs.
        """
        job_ids = []
        for gap in gaps:
            category = gap.get("category", "uncategorized")
            needed = gap.get("needed", 20)
            element = gap.get("element", "")
            job_id = self.create_job(
                country=country,
                category=category,
                missing_elements=[element] if element else [],
                target_count=max(needed, 5),
            )
            job_ids.append(job_id)

        logger.info(f"Created {len(job_ids)} jobs from gaps for {country}")
        return job_ids

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all local jobs."""
        jobs = []
        for job_file in sorted(self._output_dir.glob("job_*.json")):
            with open(job_file, "r", encoding="utf-8") as f:
                jobs.append(json.load(f))
        return jobs
