"""
Data Validator Agent - Data quality and schema validation.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
import hashlib
from collections import defaultdict
from PIL import Image
import numpy as np

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...data.firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class DataValidatorAgent(BaseAgent):
    """
    Validates data quality and schema compliance.
    
    Responsibilities:
    - Schema validation
    - Image quality checks
    - Caption quality checks
    - Duplicate detection
    - Safety screening
    - Integrity verification
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.firebase = FirebaseClient()
        self.schema_v1_0 = self._load_schema()
        self.duplicate_index = {}  # In-memory duplicate index
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema v1.0 definition."""
        return {
            "required": {
                "item_id": {"type": "string", "pattern": r"^[A-Z]{2}_[A-Z]+_\d{6}$"},
                "country": {"type": "string"},
                "category": {"type": "string"},
                "image_path": {"type": "string", "exists": True},
                "caption_raw": {"type": "string", "min_length": 5},
                "consent_attestation": {"type": "boolean", "value": True}
            },
            "optional": {
                "subcategory": {"type": "string"},
                "caption_normalized": {"type": "string"},
                "era": {"type": "string"}
            }
        }
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Validate data quality and schema.
        
        Args:
            input_data: {
                "action": str,  # "validate_item" | "validate_batch" | "audit" | "duplicate_check"
                "item": Dict (optional, for single item),
                "batch_file": str (optional, for batch),
                "country": str (optional, for audit)
            }
            
        Returns:
            AgentResult with validation results
        """
        try:
            action = input_data.get("action", "validate_item")
            
            if action == "validate_item":
                return self._validate_single_item(input_data.get("item", {}))
            elif action == "validate_batch":
                return self._validate_batch(input_data.get("batch_file", ""))
            elif action == "audit":
                return self._audit_country(input_data.get("country", self.config.country))
            elif action == "duplicate_check":
                return self._check_duplicate(input_data.get("image_path", ""))
            else:
                return AgentResult(
                    success=False,
                    data={},
                    message=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Data validation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Validation error: {str(e)}"
            )
    
    def _validate_single_item(self, item: Dict[str, Any]) -> AgentResult:
        """Validate a single item."""
        errors = []
        warnings = []
        checks = {}
        
        # Schema validation
        schema_result = self._validate_schema(item)
        errors.extend(schema_result.get("errors", []))
        warnings.extend(schema_result.get("warnings", []))
        checks["schema"] = schema_result
        
        # Image quality
        if "image_path" in item:
            image_result = self._check_image_quality(item["image_path"])
            checks["image_quality"] = image_result
            if not image_result.get("passed", False):
                errors.append(f"Image quality check failed: {image_result.get('reason', '')}")
        
        # Caption quality
        if "caption_raw" in item:
            caption_result = self._check_caption_quality(item["caption_raw"])
            checks["caption_quality"] = caption_result
            if not caption_result.get("passed", False):
                warnings.append(f"Caption quality issues: {caption_result.get('issues', [])}")
        
        # Duplicate check
        if "image_path" in item:
            duplicate_result = self._check_duplicate(item["image_path"])
            checks["duplicate"] = duplicate_result
            if duplicate_result.get("is_duplicate", False):
                errors.append(f"Duplicate detected: {duplicate_result.get('matched_id', '')}")
        
        # Safety screening
        safety_result = self._screen_safety(item.get("image_path"), item.get("caption_raw", ""))
        checks["safety"] = safety_result
        if not safety_result.get("safe", True):
            errors.append("Safety screening failed")
        
        # Integrity verification
        integrity_result = self._verify_integrity(item)
        checks["integrity"] = integrity_result
        if not integrity_result.get("valid", False):
            errors.append("Integrity check failed")
        
        return AgentResult(
            success=len(errors) == 0,
            data={
                "item_id": item.get("item_id"),
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "checks": checks
            },
            message=f"Validation: {'PASSED' if len(errors) == 0 else 'FAILED'} ({len(errors)} errors, {len(warnings)} warnings)"
        )
    
    def _validate_batch(self, batch_file: str) -> AgentResult:
        """Validate multiple items from batch file."""
        batch_path = Path(batch_file)
        if not batch_path.exists():
            return AgentResult(
                success=False,
                data={},
                message=f"Batch file not found: {batch_file}"
            )
        
        items = []
        with open(batch_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        results = []
        for item in items:
            result = self._validate_single_item(item)
            results.append(result.data)
        
        valid_count = sum(1 for r in results if r.get("valid", False))
        
        return AgentResult(
            success=True,
            data={
                "total": len(items),
                "valid": valid_count,
                "invalid": len(items) - valid_count,
                "results": results
            },
            message=f"Batch validation: {valid_count}/{len(items)} valid"
        )
    
    def _audit_country(self, country: str) -> AgentResult:
        """Audit all items for a country."""
        contributions = self.firebase.get_contributions(country=country)
        
        audit_results = {
            "total_items": len(contributions),
            "valid": 0,
            "invalid": 0,
            "issues_by_type": defaultdict(int),
            "issues_by_category": defaultdict(int)
        }
        
        for contrib in contributions:
            result = self._validate_single_item(contrib)
            if result.data.get("valid", False):
                audit_results["valid"] += 1
            else:
                audit_results["invalid"] += 1
                for error in result.data.get("errors", []):
                    audit_results["issues_by_type"][error] += 1
                category = contrib.get("category", "unknown")
                audit_results["issues_by_category"][category] += 1
        
        return AgentResult(
            success=True,
            data=audit_results,
            message=f"Audit complete: {audit_results['valid']}/{audit_results['total_items']} valid"
        )
    
    def _validate_schema(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Validate item against schema."""
        errors = []
        warnings = []
        
        # Check required fields
        for field, rules in self.schema_v1_0["required"].items():
            if field not in item:
                errors.append(f"Missing required field: {field}")
            else:
                value = item[field]
                # Type check
                if rules.get("type") == "string" and not isinstance(value, str):
                    errors.append(f"Invalid type for {field}: expected string")
                elif rules.get("type") == "boolean" and not isinstance(value, bool):
                    errors.append(f"Invalid type for {field}: expected boolean")
                
                # Pattern check
                if "pattern" in rules:
                    import re
                    if not re.match(rules["pattern"], str(value)):
                        errors.append(f"Invalid format for {field}")
                
                # Min length check
                if "min_length" in rules and len(str(value)) < rules["min_length"]:
                    errors.append(f"{field} too short (min {rules['min_length']})")
                
                # Exists check
                if "exists" in rules and rules["exists"]:
                    path = Path(value)
                    if not path.exists():
                        errors.append(f"File not found: {value}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _check_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Check image quality."""
        try:
            path = Path(image_path)
            if not path.exists():
                return {"passed": False, "reason": "File not found"}
            
            image = Image.open(path)
            
            checks = {
                "resolution": {
                    "min_width": 512,
                    "min_height": 512,
                    "actual": (image.width, image.height),
                    "passed": image.width >= 512 and image.height >= 512
                },
                "file_size": {
                    "min_kb": 50,
                    "max_mb": 20,
                    "actual_kb": path.stat().st_size / 1024,
                    "passed": 50 <= path.stat().st_size / 1024 <= 20480
                },
                "format": {
                    "allowed": ["JPEG", "PNG", "WebP"],
                    "actual": image.format,
                    "passed": image.format in ["JPEG", "PNG", "WEBP"]
                }
            }
            
            all_passed = all(c["passed"] for c in checks.values())
            
            return {
                "passed": all_passed,
                "checks": checks,
                "reason": "All checks passed" if all_passed else "Some checks failed"
            }
            
        except Exception as e:
            return {"passed": False, "reason": f"Image check error: {str(e)}"}
    
    def _check_caption_quality(self, caption: str) -> Dict[str, Any]:
        """Check caption quality."""
        checks = {
            "length": {
                "min": 10,
                "max": 1000,
                "actual": len(caption),
                "passed": 10 <= len(caption) <= 1000
            },
            "meaningful": {
                "min_words": 3,
                "actual_words": len(caption.split()),
                "passed": len(caption.split()) >= 3
            }
        }
        
        all_passed = all(c["passed"] for c in checks.values())
        issues = [k for k, v in checks.items() if not v["passed"]]
        
        return {
            "passed": all_passed,
            "checks": checks,
            "issues": issues
        }
    
    def _check_duplicate(self, image_path: str) -> Dict[str, Any]:
        """Check for duplicate images."""
        try:
            path = Path(image_path)
            if not path.exists():
                return {"is_duplicate": False, "reason": "File not found"}
            
            # Calculate perceptual hash (simplified)
            image = Image.open(path)
            image_hash = self._calculate_image_hash(image)
            
            # Check against index
            if image_hash in self.duplicate_index:
                return {
                    "is_duplicate": True,
                    "match_type": "perceptual",
                    "matched_id": self.duplicate_index[image_hash]
                }
            
            # Add to index
            item_id = path.stem
            self.duplicate_index[image_hash] = item_id
            
            return {"is_duplicate": False}
            
        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return {"is_duplicate": False, "error": str(e)}
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash for duplicate detection."""
        # Simplified: use file hash
        # In real implementation, would use perceptual hashing (pHash)
        image_bytes = image.tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _screen_safety(self, image_path: Optional[str], caption: str) -> Dict[str, Any]:
        """Screen for safety issues."""
        flags = []
        
        # Check caption for PII
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',  # Phone
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        import re
        for pattern in pii_patterns:
            if re.search(pattern, caption):
                flags.append({"type": "pii", "severity": "MEDIUM"})
        
        # In real implementation, would use NSFW detection models
        # For now, just check caption
        
        return {
            "safe": len([f for f in flags if f.get("severity") == "HIGH"]) == 0,
            "flags": flags
        }
    
    def _verify_integrity(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Verify data integrity."""
        checks = {
            "consent": item.get("consent_attestation") == True,
            "timestamp": "upload_timestamp" in item,
            "country_match": item.get("country") == self.config.country if self.config.country else True
        }
        
        return {
            "valid": all(checks.values()),
            "checks": checks
        }
