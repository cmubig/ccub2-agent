"""
Scout Agent - Detects coverage gaps and failure modes.
"""

from typing import Dict, Any, List
import logging

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...data.gap_analyzer import DataGapAnalyzer
from ...data.country_pack import CountryDataPack
from ...retrieval.clip_image_rag import CLIPImageRAG

logger = logging.getLogger(__name__)


class ScoutAgent(BaseAgent):
    """
    Detects coverage gaps and recurring failure modes.
    
    Responsibilities:
    - Analyze failure modes from Judge Agent
    - Query RAG indices for coverage gaps
    - Identify missing cultural references
    - Prioritize data collection needs
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Initialize country pack for gap analysis
        country_pack = CountryDataPack(config.country)
        self.gap_analyzer = DataGapAnalyzer(country_pack)
        # CLIP RAG will be initialized on demand
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Analyze gaps and identify missing references.
        
        Args:
            input_data: {
                "failure_modes": List[Dict],
                "country": str,
                "category": str (optional)
            }
            
        Returns:
            AgentResult with gap analysis and reference recommendations
        """
        try:
            failure_modes = input_data.get("failure_modes", [])
            country = input_data.get("country", self.config.country)
            category = input_data.get("category", self.config.category)
            
            # Analyze gaps
            gaps = self.gap_analyzer.analyze(failure_modes, country)
            
            # Check if references exist
            needs_more_data = len(gaps) > 0
            
            return AgentResult(
                success=True,
                data={
                    "gaps": gaps,
                    "needs_more_data": needs_more_data,
                    "missing_elements": [g.get("element") for g in gaps],
                    "category": category,
                    "country": country
                },
                message=f"Found {len(gaps)} coverage gaps"
            )
            
        except Exception as e:
            logger.error(f"Scout execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Scout error: {str(e)}"
            )
