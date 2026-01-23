"""
Gap-Based Job Creator

Creates data collection jobs based on coverage gaps.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .coverage_analyzer import CoverageReport

logger = logging.getLogger(__name__)


class GapBasedJobCreator:
    """
    Creates data collection jobs based on identified gaps.
    
    Prioritizes jobs based on coverage analysis.
    """
    
    def __init__(self):
        """Initialize job creator."""
        logger.info("GapBasedJobCreator initialized")
    
    def create_job_from_gap(
        self,
        coverage_report: CoverageReport,
        target_count: int = 20,
        priority: str = "medium",
    ) -> Dict[str, Any]:
        """
        Create a job from a coverage gap.
        
        Args:
            coverage_report: Coverage analysis report
            target_count: Target number of contributions
            priority: Job priority ("low", "medium", "high")
            
        Returns:
            Job creation result
        """
        from ..firebase_client import FirebaseClient
        firebase = FirebaseClient()
        
        # Determine priority based on coverage
        if coverage_report.coverage_score < 0.3:
            priority = "high"
        elif coverage_report.coverage_score < 0.6:
            priority = "medium"
        else:
            priority = "low"
        
        # Create job
        job_data = {
            "country": coverage_report.country,
            "category": coverage_report.category,
            "missing_elements": coverage_report.priority_gaps,
            "target_count": target_count,
            "priority": priority,
            "reason": "coverage_gap",
            "coverage_score": coverage_report.coverage_score,
        }
        
        try:
            job_id = firebase.create_job(**job_data)
            
            logger.info(f"Job created from gap: job_id={job_id}, country={coverage_report.country}")
            
            return {
                "success": True,
                "job_id": job_id,
                "job_data": job_data,
            }
        except Exception as e:
            logger.error(f"Error creating job: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
    
    def create_jobs_from_gaps(
        self,
        coverage_reports: List[CoverageReport],
        target_count: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple jobs from coverage gaps.
        
        Args:
            coverage_reports: List of coverage reports
            target_count: Target count per job
            
        Returns:
            List of job creation results
        """
        results = []
        
        # Sort by priority (low coverage first)
        sorted_reports = sorted(
            coverage_reports,
            key=lambda r: r.coverage_score,
        )
        
        for report in sorted_reports:
            if report.priority_gaps:
                result = self.create_job_from_gap(
                    coverage_report=report,
                    target_count=target_count,
                )
                results.append(result)
        
        logger.info(f"Created {len(results)} jobs from {len(coverage_reports)} reports")
        return results


def create_job_from_gap(
    coverage_report: CoverageReport,
    target_count: int = 20,
) -> Dict[str, Any]:
    """
    Convenience function to create a job from a gap.
    
    Args:
        coverage_report: Coverage report
        target_count: Target count
        
    Returns:
        Job creation result
    """
    creator = GapBasedJobCreator()
    return creator.create_job_from_gap(
        coverage_report=coverage_report,
        target_count=target_count,
    )
