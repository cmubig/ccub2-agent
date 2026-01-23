"""
Data Management System

Manages cultural data, Firebase integration, and data collection.
"""

# Gap analysis components (implemented)
from .gap_analysis import (
    CoverageAnalyzer,
    CoverageReport,
    analyze_coverage,
    GapBasedJobCreator,
    create_job_from_gap,
)

# Placeholder implementations (raise NotImplementedError)
from .country_pack import CountryDataPack
from .firebase_client import FirebaseClient, get_firebase_client
from .gap_analyzer import DataGapAnalyzer
from .data_gap_detector import DataGapDetector, create_data_gap_detector
from .job_creator import AgentJobCreator

__all__ = [
    # Gap analysis (implemented)
    "CoverageAnalyzer",
    "CoverageReport",
    "analyze_coverage",
    "GapBasedJobCreator",
    "create_job_from_gap",
    # Placeholders (need implementation)
    "CountryDataPack",
    "FirebaseClient",
    "get_firebase_client",
    "DataGapAnalyzer",
    "DataGapDetector",
    "create_data_gap_detector",
    "AgentJobCreator",
]
