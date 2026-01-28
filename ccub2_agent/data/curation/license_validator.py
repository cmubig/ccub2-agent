"""License validator for curated images.

Validates that all images in the curated pipeline have acceptable licenses
and proper attribution before they can move from staging to approved.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ccub2_agent.schemas.provenance_schema import ALLOWED_LICENSES, License

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of license validation for a single image."""

    image_id: str
    valid: bool
    license: str
    has_attribution: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Aggregated validation report for a batch."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.valid / self.total if self.total > 0 else 0.0


class LicenseValidator:
    """Validate licenses and attribution for curated images."""

    def __init__(self, curated_dir: Path):
        self.curated_dir = Path(curated_dir)
        self.provenance_file = self.curated_dir / "metadata" / "provenance.jsonl"

    def validate_country(self, country: str) -> ValidationReport:
        """Validate all curated images for a given country."""
        report = ValidationReport()

        if not self.provenance_file.exists():
            logger.warning(f"No provenance file found at {self.provenance_file}")
            return report

        with open(self.provenance_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if record.get("country", "").lower() != country.lower():
                    continue

                result = self._validate_record(record)
                report.results.append(result)
                report.total += 1
                if result.valid:
                    report.valid += 1
                else:
                    report.invalid += 1

        logger.info(
            f"Validation for {country}: {report.valid}/{report.total} passed "
            f"({report.pass_rate:.0%})"
        )
        return report

    def _validate_record(self, record: dict) -> ValidationResult:
        """Validate a single provenance record."""
        errors: list[str] = []
        image_id = record.get("image_id", "unknown")
        license_str = record.get("license", "")
        attribution = record.get("attribution", "")

        # Check license is in allowed set
        try:
            license_val = License(license_str)
            valid_license = license_val in ALLOWED_LICENSES
        except ValueError:
            valid_license = False
            errors.append(f"Unknown license: {license_str}")

        if not valid_license and not errors:
            errors.append(f"License not in allowed set: {license_str}")

        # Check attribution exists
        has_attribution = bool(attribution and attribution.strip())
        if not has_attribution:
            errors.append("Missing attribution")

        # Check image file exists
        image_path = record.get("image_path", "")
        if image_path and not Path(image_path).exists():
            errors.append(f"Image file not found: {image_path}")

        # Check required fields
        for required_field in ["original_url", "source_platform", "source_id"]:
            if not record.get(required_field):
                errors.append(f"Missing required field: {required_field}")

        valid = valid_license and has_attribution and len(errors) == 0

        return ValidationResult(
            image_id=image_id,
            valid=valid,
            license=license_str,
            has_attribution=has_attribution,
            errors=errors,
        )

    def get_invalid_records(self, country: str) -> list[ValidationResult]:
        """Get all invalid records for a country."""
        report = self.validate_country(country)
        return [r for r in report.results if not r.valid]
