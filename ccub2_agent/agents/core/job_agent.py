"""
Job Agent - Creates WorldCCUB collection jobs.
"""

from typing import Dict, Any
import logging

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...data.job_creator import AgentJobCreator

logger = logging.getLogger(__name__)


class JobAgent(BaseAgent):
    """
    Converts detected coverage gaps into WorldCCUB collection jobs.
    
    Responsibilities:
    - Create Firebase jobs with clear specifications
    - Manage job lifecycle
    - Track job completion
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.job_creator = AgentJobCreator()
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Create a collection job.
        
        Args:
            input_data: {
                "country": str,
                "category": str,
                "missing_elements": List[str],
                "target_count": int (optional)
            }
            
        Returns:
            AgentResult with job creation status
        """
        try:
            country = input_data.get("country", self.config.country)
            category = input_data.get("category", self.config.category)
            missing_elements = input_data.get("missing_elements", [])
            target_count = input_data.get("target_count", 20)
            
            # Create job
            job_id = self.job_creator.create_job(
                country=country,
                category=category,
                missing_elements=missing_elements,
                target_count=target_count
            )
            
            return AgentResult(
                success=True,
                data={
                    "job_id": job_id,
                    "country": country,
                    "category": category,
                    "target_count": target_count
                },
                message=f"Job {job_id} created successfully"
            )
            
        except Exception as e:
            logger.error(f"Job creation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Job creation error: {str(e)}"
            )
