"""WorldCCUB curated data pipeline.

Hybrid data collection: User 60% + Curated 30% + Partner 10%.
"""

from ccub2_agent.data.curation.base_downloader import BaseDownloader
from ccub2_agent.data.curation.license_validator import LicenseValidator

__all__ = ["BaseDownloader", "LicenseValidator"]
