"""
Gap Analysis System

Analyzes data coverage gaps and creates collection jobs.
"""

from .coverage_analyzer import (
    CoverageAnalyzer,
    CoverageReport,
    analyze_coverage,
)

from .job_creator import (
    GapBasedJobCreator,
    create_job_from_gap,
)

__all__ = [
    "CoverageAnalyzer",
    "CoverageReport",
    "analyze_coverage",
    "GapBasedJobCreator",
    "create_job_from_gap",
]
