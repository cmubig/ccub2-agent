"""
Model Registry

Tracks model versions, hashes, and metadata for reproducibility.
Essential for answering "which model version was used?" in NeurIPS reviews.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """
    Model version information.
    
    Tracks exact model version, hash, and metadata for reproducibility.
    """
    model_name: str  # e.g., "Qwen3-VL-8B", "FLUX-1.1", "SDXL"
    model_type: str  # "vlm", "i2i", "t2i", "clip"
    version: str  # Version string (e.g., "1.0", "2024-01-15")
    model_hash: Optional[str] = None  # Hash of model weights/config
    config_hash: Optional[str] = None  # Hash of model configuration
    checkpoint_path: Optional[str] = None  # Path to checkpoint
    source: str = ""  # Source (e.g., "huggingface", "local", "api")
    source_url: Optional[str] = None  # URL if from remote source
    loaded_at: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata and timestamp."""
        if self.metadata is None:
            self.metadata = {}
        if not self.loaded_at:
            self.loaded_at = datetime.now().isoformat()
    
    def compute_hash(self, model_path: Optional[Path] = None) -> str:
        """
        Compute hash of model file or checkpoint.
        
        Args:
            model_path: Path to model file. If None, uses checkpoint_path.
            
        Returns:
            SHA256 hash
        """
        path = model_path or (Path(self.checkpoint_path) if self.checkpoint_path else None)
        if path is None or not path.exists():
            return ""
        
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelRegistry:
    """
    Centralized model version registry.
    
    Tracks all models used in experiments for full reproducibility.
    """
    
    def __init__(self, registry_file: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            registry_file: Path to registry JSON file. If None, uses default.
        """
        if registry_file is None:
            registry_file = Path("models/registry.json")
        
        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelVersion] = {}
        
        # Load existing registry
        if self.registry_file.exists():
            self.load()
        else:
            logger.info(f"Creating new model registry: {self.registry_file}")
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        version: str,
        checkpoint_path: Optional[str] = None,
        source: str = "huggingface",
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compute_hash: bool = True,
    ) -> ModelVersion:
        """
        Register a model version.
        
        Args:
            model_name: Name of the model
            model_type: Type (vlm, i2i, t2i, clip)
            version: Version string
            checkpoint_path: Path to checkpoint
            source: Source (huggingface, local, api)
            source_url: URL if from remote
            metadata: Additional metadata
            compute_hash: Whether to compute model hash
            
        Returns:
            Registered ModelVersion
        """
        model_key = f"{model_name}_{model_type}_{version}"
        
        model_version = ModelVersion(
            model_name=model_name,
            model_type=model_type,
            version=version,
            checkpoint_path=checkpoint_path,
            source=source,
            source_url=source_url,
            metadata=metadata or {},
        )
        
        # Compute hash if requested
        if compute_hash and checkpoint_path:
            model_version.model_hash = model_version.compute_hash()
        
        self.models[model_key] = model_version
        
        logger.info(f"Model registered: {model_key}")
        
        # Save registry
        self.save()
        
        return model_version
    
    def get_model(self, model_name: str, model_type: str, version: str) -> Optional[ModelVersion]:
        """
        Get a registered model version.
        
        Args:
            model_name: Name of the model
            model_type: Type
            version: Version string
            
        Returns:
            ModelVersion if found, None otherwise
        """
        model_key = f"{model_name}_{model_type}_{version}"
        return self.models.get(model_key)
    
    def list_models(self, model_type: Optional[str] = None) -> List[ModelVersion]:
        """
        List all registered models.
        
        Args:
            model_type: Filter by type. If None, returns all.
            
        Returns:
            List of ModelVersion objects
        """
        if model_type is None:
            return list(self.models.values())
        return [
            model for model in self.models.values()
            if model.model_type == model_type
        ]
    
    def save(self):
        """Save registry to JSON file."""
        registry_data = {
            "updated_at": datetime.now().isoformat(),
            "models": {
                key: model.to_dict()
                for key, model in self.models.items()
            },
        }
        
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model registry saved: {self.registry_file}")
    
    def load(self):
        """Load registry from JSON file."""
        with open(self.registry_file, "r", encoding="utf-8") as f:
            registry_data = json.load(f)
        
        self.models = {
            key: ModelVersion(**model_dict)
            for key, model_dict in registry_data.get("models", {}).items()
        }
        
        logger.info(f"Model registry loaded: {len(self.models)} models")


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry(registry_file: Optional[Path] = None) -> ModelRegistry:
    """Get or create global model registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry(registry_file=registry_file)
    return _global_registry


def register_model(
    model_name: str,
    model_type: str,
    version: str,
    checkpoint_path: Optional[str] = None,
    source: str = "huggingface",
    source_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ModelVersion:
    """Convenience function to register a model."""
    registry = get_model_registry()
    return registry.register_model(
        model_name=model_name,
        model_type=model_type,
        version=version,
        checkpoint_path=checkpoint_path,
        source=source,
        source_url=source_url,
        metadata=metadata,
    )
