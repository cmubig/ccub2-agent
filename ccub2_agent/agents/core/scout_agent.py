"""
Scout Agent - Detects coverage gaps and failure modes.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...data.gap_analyzer import DataGapAnalyzer
from ...data.country_pack import CountryDataPack
from ...retrieval.clip_image_rag import CLIPImageRAG

logger = logging.getLogger(__name__)

# Default index directory
DEFAULT_INDEX_BASE = Path(__file__).parent.parent.parent.parent / "data" / "clip_index"


class ScoutAgent(BaseAgent):
    """
    Detects coverage gaps and recurring failure modes.

    Responsibilities:
    - Analyze failure modes from Judge Agent
    - Query RAG indices for coverage gaps
    - Retrieve reference images via CLIP RAG
    - Identify missing cultural references
    - Prioritize data collection needs
    """

    def __init__(self, config: AgentConfig, index_base: Optional[Path] = None, shared_clip_rag: Optional[CLIPImageRAG] = None):
        super().__init__(config)
        # Initialize country pack for gap analysis
        country_pack = CountryDataPack(config.country)
        self.gap_analyzer = DataGapAnalyzer(country_pack)

        # CLIP RAG initialization (lazy loading or shared)
        self.index_base = index_base or DEFAULT_INDEX_BASE
        self._clip_rag: Optional[CLIPImageRAG] = shared_clip_rag

    def _get_clip_rag(self, country: str) -> Optional[CLIPImageRAG]:
        """Get or initialize CLIP RAG for the specified country."""
        index_dir = self.index_base / country

        if not index_dir.exists():
            logger.warning(f"No CLIP index found for country: {country} at {index_dir}")
            return None

        # Initialize if not cached or country changed
        if self._clip_rag is None:
            logger.info(f"Initializing CLIP RAG for {country}")
            self._clip_rag = CLIPImageRAG(index_dir=index_dir)

        return self._clip_rag

    def _retrieve_references(
        self,
        image_path: str,
        country: str,
        category: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve reference images using CLIP RAG."""
        clip_rag = self._get_clip_rag(country)

        if clip_rag is None:
            return []

        try:
            references = clip_rag.retrieve_similar_images(
                image_path=Path(image_path),
                k=k,
                category=category
            )
            logger.info(f"Retrieved {len(references)} references for {country}/{category}")
            return references
        except Exception as e:
            logger.error(f"Reference retrieval failed: {e}")
            return []

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze gaps and retrieve reference images.

        Args:
            input_data: {
                "image_path": str (query image for reference retrieval),
                "failure_modes": List[Dict],
                "country": str,
                "category": str (optional),
                "k": int (number of references, default 5)
            }

        Returns:
            AgentResult with gap analysis and retrieved references
        """
        try:
            image_path = input_data.get("image_path")
            failure_modes = input_data.get("failure_modes", [])
            country = input_data.get("country", self.config.country)
            category = input_data.get("category", self.config.category)
            k = input_data.get("k", 5)

            # Analyze gaps
            gaps = self.gap_analyzer.analyze(failure_modes, country)

            # Retrieve references if image_path provided
            references = []
            if image_path:
                references = self._retrieve_references(
                    image_path=image_path,
                    country=country,
                    category=category,
                    k=k
                )

            # Need more data if gaps found AND no references retrieved
            needs_more_data = len(gaps) > 0 and len(references) == 0

            return AgentResult(
                success=True,
                data={
                    "gaps": gaps,
                    "needs_more_data": needs_more_data,
                    "missing_elements": [g.get("element") for g in gaps],
                    "references": references,  # NEW: Retrieved references
                    "category": category,
                    "country": country
                },
                message=f"Found {len(gaps)} gaps, retrieved {len(references)} references"
            )

        except Exception as e:
            logger.error(f"Scout execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Scout error: {str(e)}"
            )
