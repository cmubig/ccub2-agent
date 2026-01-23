"""
Index/Release Agent - RAG indices and versioned releases.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
import shutil
from datetime import datetime

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...retrieval.clip_image_rag import CLIPImageRAG, create_clip_rag
from ...data.country_pack import CountryDataPack

logger = logging.getLogger(__name__)


class IndexReleaseAgent(BaseAgent):
    """
    Updates retrieval indices and prepares versioned releases.
    
    Responsibilities:
    - Build/update CLIP image indices
    - Build/update text knowledge base indices
    - Prepare versioned dataset releases
    - Generate changelogs
    - Manage train/val/test splits
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.data_root = Path(__file__).parent.parent.parent.parent / "data"
        self.releases_dir = self.data_root / "releases"
        self.releases_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Update indices or prepare release.
        
        Args:
            input_data: {
                "action": str,  # "update_indices" | "create_release" | "rebuild_indices"
                "country": str (optional),
                "incremental": bool (optional),
                "version": str (optional, for releases),
                "changelog": str (optional, for releases)
            }
            
        Returns:
            AgentResult with index/release status
        """
        try:
            action = input_data.get("action", "update_indices")
            country = input_data.get("country", self.config.country)
            
            if action == "update_indices":
                return self._update_indices(country, input_data.get("incremental", True))
            elif action == "rebuild_indices":
                return self._rebuild_indices(country)
            elif action == "create_release":
                return self._create_release(
                    input_data.get("version", "1.0.0"),
                    input_data.get("changelog", "")
                )
            else:
                return AgentResult(
                    success=False,
                    data={},
                    message=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Index/Release execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Index/Release error: {str(e)}"
            )
    
    def _update_indices(self, country: str, incremental: bool) -> AgentResult:
        """Update CLIP and text indices incrementally."""
        results = {}
        
        # Update CLIP index
        clip_result = self._update_clip_index(country, incremental)
        results["clip_index"] = clip_result
        
        # Update text KB index
        text_result = self._update_text_index(country, incremental)
        results["text_index"] = text_result
        
        return AgentResult(
            success=True,
            data=results,
            message=f"Indices updated for {country} (incremental={incremental})"
        )
    
    def _update_clip_index(self, country: str, incremental: bool) -> Dict[str, Any]:
        """Update CLIP image index."""
        try:
            clip_index_dir = self.data_root / "clip_index" / country
            images_dir = self.data_root / "country_packs" / country / "images"
            
            if not images_dir.exists():
                return {"status": "skipped", "reason": "Images directory not found"}
            
            # Load existing index if incremental
            if incremental and clip_index_dir.exists():
                logger.info(f"Incremental update of CLIP index for {country}")
                # In real implementation, would add only new images
                # For now, rebuild if needed
                pass
            
            # Build/rebuild index
            logger.info(f"Building CLIP index for {country}")
            clip_rag = create_clip_rag(
                country=country,
                images_dir=images_dir,
                index_dir=clip_index_dir
            )
            
            # Count indexed images
            index_count = clip_rag.index.ntotal if hasattr(clip_rag, 'index') else 0
            
            return {
                "status": "success",
                "index_path": str(clip_index_dir),
                "images_indexed": index_count,
                "incremental": incremental
            }
            
        except Exception as e:
            logger.error(f"CLIP index update failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _update_text_index(self, country: str, incremental: bool) -> Dict[str, Any]:
        """Update text knowledge base index."""
        try:
            text_index_dir = self.data_root / "cultural_index" / country
            knowledge_file = self.data_root / "cultural_knowledge" / f"{country}_knowledge.json"
            
            if not knowledge_file.exists():
                return {"status": "skipped", "reason": "Knowledge file not found"}
            
            # In real implementation, would use sentence-transformers and FAISS
            # For now, just check if index exists
            if text_index_dir.exists():
                return {
                    "status": "exists",
                    "index_path": str(text_index_dir),
                    "incremental": incremental
                }
            else:
                return {
                    "status": "needs_build",
                    "index_path": str(text_index_dir),
                    "message": "Text index needs to be built"
                }
                
        except Exception as e:
            logger.error(f"Text index update failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _rebuild_indices(self, country: str) -> AgentResult:
        """Rebuild indices from scratch."""
        # Delete existing indices
        clip_index_dir = self.data_root / "clip_index" / country
        text_index_dir = self.data_root / "cultural_index" / country
        
        if clip_index_dir.exists():
            shutil.rmtree(clip_index_dir)
        if text_index_dir.exists():
            shutil.rmtree(text_index_dir)
        
        # Rebuild
        return self._update_indices(country, incremental=False)
    
    def _create_release(self, version: str, changelog: str) -> AgentResult:
        """Create versioned dataset release."""
        try:
            release_dir = self.releases_dir / f"v{version}"
            release_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect dataset files
            dataset_files = []
            for country_dir in (self.data_root / "country_packs").iterdir():
                if country_dir.is_dir():
                    dataset_file = country_dir / "approved_dataset_enhanced.json"
                    if dataset_file.exists():
                        dataset_files.append((country_dir.name, dataset_file))
            
            # Create dataset archive structure
            release_data_dir = release_dir / "data"
            release_data_dir.mkdir(exist_ok=True)
            
            # Copy dataset files
            for country, dataset_file in dataset_files:
                country_release_dir = release_data_dir / country
                country_release_dir.mkdir(exist_ok=True)
                shutil.copy2(dataset_file, country_release_dir / "approved_dataset.json")
            
            # Copy indices
            indices_dir = release_dir / "indices"
            indices_dir.mkdir(exist_ok=True)
            
            clip_indices = indices_dir / "clip_index"
            text_indices = indices_dir / "cultural_index"
            clip_indices.mkdir(exist_ok=True)
            text_indices.mkdir(exist_ok=True)
            
            # Copy index directories
            source_clip = self.data_root / "clip_index"
            source_text = self.data_root / "cultural_index"
            
            if source_clip.exists():
                for country_dir in source_clip.iterdir():
                    if country_dir.is_dir():
                        shutil.copytree(country_dir, clip_indices / country_dir.name, dirs_exist_ok=True)
            
            if source_text.exists():
                for country_dir in source_text.iterdir():
                    if country_dir.is_dir():
                        shutil.copytree(country_dir, text_indices / country_dir.name, dirs_exist_ok=True)
            
            # Generate changelog
            changelog_path = release_dir / "CHANGELOG.md"
            with open(changelog_path, 'w', encoding='utf-8') as f:
                f.write(f"# WorldCCUB-Global v{version}\n\n")
                f.write(f"Release Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
                f.write(f"## Changes\n\n{changelog}\n\n")
                f.write("## Dataset Statistics\n\n")
                f.write(f"- Countries: {len(dataset_files)}\n")
                f.write(f"- Total items: [to be calculated]\n")
            
            # Generate dataset card
            datacard_path = release_dir / "DATACARD.md"
            self._generate_datacard(datacard_path, version, dataset_files)
            
            # Generate checksums
            checksums_path = release_dir / "checksums.sha256"
            # In real implementation, would calculate SHA256 checksums
            
            return AgentResult(
                success=True,
                data={
                    "version": version,
                    "release_dir": str(release_dir),
                    "countries": len(dataset_files),
                    "changelog_path": str(changelog_path),
                    "datacard_path": str(datacard_path)
                },
                message=f"Release v{version} created successfully"
            )
            
        except Exception as e:
            logger.error(f"Release creation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Release error: {str(e)}"
            )
    
    def _generate_datacard(self, datacard_path: Path, version: str, dataset_files: List):
        """Generate dataset card in NeurIPS D&B format."""
        datacard = f"""# WorldCCUB-Global Dataset Card

## Dataset Summary

**Version**: {version}
**Release Date**: {datetime.now().strftime('%Y-%m-%d')}
**License**: [To be specified]
**Authors**: WorldCCUB Global Contributors

## Dataset Description

WorldCCUB-Global is a provenance-attested, peer-reviewed cultural dataset for evaluating and improving cultural fidelity in generative image models.

## Dataset Composition

- **Countries**: {len(dataset_files)}
- **Categories**: 8 core categories
- **Collection Method**: Peer-reviewed platform contributions
- **Consent**: All items include consent attestation

## Intended Use

- Research on cultural bias in generative models
- Training culturally-aware image generation models
- Benchmarking cultural fidelity metrics
- Developing cultural improvement systems

## Limitations

- Coverage varies by country (Tier-0 vs Tier-1)
- Cultural subjectivity in evaluation
- Potential biases in contributor demographics

## Ethics & Governance

- Consent-first collection
- Peer review process
- Withdrawal mechanism
- Versioned releases with changelogs

## Citation

[To be added upon publication]
"""
        with open(datacard_path, 'w', encoding='utf-8') as f:
            f.write(datacard)
