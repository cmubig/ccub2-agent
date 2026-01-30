"""Provenance schema for WorldCCUB-Global dataset.

Tracks the origin, licensing, and verification status of every image
in the dataset, whether user-submitted, curated, or partner-sourced.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class DataSource(str, Enum):
    USER_SUBMITTED = "user_submitted"
    CURATED = "curated"
    PARTNER = "partner"


class License(str, Enum):
    CC0_1_0 = "CC0-1.0"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    PUBLIC_DOMAIN = "Public Domain"
    UNSPLASH = "Unsplash License"
    PIXABAY = "Pixabay License"


ALLOWED_LICENSES = set(License)


class CurationPlatform(str, Enum):
    WIKIMEDIA_COMMONS = "wikimedia_commons"
    PIXABAY = "pixabay"
    UNSPLASH = "unsplash"
    PARTNER = "partner"


class UserSubmittedProvenance(BaseModel):
    """Provenance record for user-submitted images."""

    source: Literal[DataSource.USER_SUBMITTED] = Field(default=DataSource.USER_SUBMITTED)
    contributor_id: str = Field(..., description="Firebase UID of the contributor")
    consent_documented: bool = Field(default=True)
    country_rep_approved: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    review_status: str = Field(default="pending", description="pending | approved | rejected")


class CuratedProvenance(BaseModel):
    """Provenance record for curated (open-license) images."""

    source: Literal[DataSource.CURATED] = Field(default=DataSource.CURATED)
    original_url: str = Field(..., description="Original URL of the image")
    license: License = Field(..., description="License of the source image")
    attribution: str = Field(..., description="Required attribution text")
    country_rep_verified: bool = Field(default=False)
    cultural_accuracy_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="VLM-assessed cultural accuracy"
    )
    source_platform: CurationPlatform = Field(..., description="Platform the image was sourced from")
    source_id: str = Field(..., description="Platform-specific identifier (e.g., File:Example.jpg)")
    download_timestamp: datetime = Field(default_factory=datetime.utcnow)


class PartnerProvenance(BaseModel):
    """Provenance record for partner-institution images."""

    source: Literal[DataSource.PARTNER] = Field(default=DataSource.PARTNER)
    institution: str = Field(..., description="Name of the partner institution")
    agreement_id: str = Field(..., description="Data sharing agreement reference")
    license: License = Field(...)
    attribution: str = Field(...)
    country_rep_verified: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ImageRecord(BaseModel):
    """Unified image record combining metadata and provenance."""

    image_id: str = Field(..., description="Unique image identifier")
    image_path: str = Field(..., description="Relative path to the image file")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    category: str = Field(..., description="Cultural category (e.g., food, clothing)")
    sub_category: Optional[str] = Field(default=None)
    caption_original: str = Field(..., description="Caption in original language")
    caption_english: Optional[str] = Field(default=None, description="English translation")
    provenance: UserSubmittedProvenance | CuratedProvenance | PartnerProvenance = Field(
        ..., discriminator="source"
    )
    cultural_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    failure_modes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
