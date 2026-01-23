"""
Country Lead Agent - CL coordination and management.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...data.firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class CountryLeadAgent(BaseAgent):
    """
    Manages Country Lead coordination.
    
    Responsibilities:
    - CL recruitment
    - Onboarding
    - Communication management
    - Contribution tracking
    - Tier management
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.firebase = FirebaseClient()
        self.cl_data_file = Path(__file__).parent.parent.parent.parent / "data" / "country_leads.json"
        self._load_cl_data()
    
    def _load_cl_data(self):
        """Load Country Lead data."""
        if self.cl_data_file.exists():
            with open(self.cl_data_file, 'r', encoding='utf-8') as f:
                self.cl_data = json.load(f)
        else:
            self.cl_data = {}
    
    def _save_cl_data(self):
        """Save Country Lead data."""
        self.cl_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cl_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.cl_data, f, indent=2, ensure_ascii=False)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Manage Country Lead operations.
        
        Args:
            input_data: {
                "action": str,  # "recruit" | "onboard" | "notify" | "track" | "tier_status"
                "country": str (optional),
                "lead_id": str (optional),
                "notification_type": str (optional),
                "context": Dict (optional)
            }
            
        Returns:
            AgentResult with CL operation results
        """
        try:
            action = input_data.get("action", "track")
            country = input_data.get("country", self.config.country)
            
            if action == "recruit":
                return self._recruit_cl(country, input_data.get("candidates", []))
            elif action == "onboard":
                return self._onboard_cl(
                    input_data.get("lead_id", ""),
                    country,
                    input_data.get("lead_data", {})
                )
            elif action == "notify":
                return self._notify_cl(
                    country,
                    input_data.get("notification_type", ""),
                    input_data.get("context", {})
                )
            elif action == "track":
                return self._track_contributions(country)
            elif action == "tier_status":
                return self._check_tier_status(country)
            else:
                return AgentResult(
                    success=False,
                    data={},
                    message=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Country Lead execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Country Lead error: {str(e)}"
            )
    
    def _recruit_cl(self, country: str, candidates: List[Dict[str, Any]]) -> AgentResult:
        """Recruit Country Lead candidates."""
        # In real implementation, would search networks, send outreach
        # For now, just log candidates
        
        logger.info(f"Recruiting CL for {country}: {len(candidates)} candidates")
        
        return AgentResult(
            success=True,
            data={
                "country": country,
                "candidates": candidates,
                "status": "recruitment_initiated"
            },
            message=f"Recruitment initiated for {country}"
        )
    
    def _onboard_cl(self, lead_id: str, country: str, lead_data: Dict[str, Any]) -> AgentResult:
        """Onboard new Country Lead."""
        if not lead_id:
            lead_id = f"CL_{country.upper()}_{datetime.now().strftime('%Y%m%d')}"
        
        cl_info = {
            "lead_id": lead_id,
            "country": country,
            "joined": datetime.now().isoformat(),
            "status": "onboarding",
            "contribution_score": 0.0,
            **lead_data
        }
        
        # Save to data file
        if country not in self.cl_data:
            self.cl_data[country] = {}
        self.cl_data[country][lead_id] = cl_info
        self._save_cl_data()
        
        logger.info(f"Onboarded CL {lead_id} for {country}")
        
        return AgentResult(
            success=True,
            data=cl_info,
            message=f"CL {lead_id} onboarded successfully"
        )
    
    def _notify_cl(self, country: str, notification_type: str, context: Dict[str, Any]) -> AgentResult:
        """Send notification to Country Lead."""
        # Get CL info
        cl_info = self.cl_data.get(country, {})
        if not cl_info:
            return AgentResult(
                success=False,
                data={},
                message=f"No CL found for {country}"
            )
        
        # Notification templates
        templates = {
            "new_job": f"New collection job available for {country}: {context.get('job_title', '')}",
            "milestone": f"Milestone reached for {country}: {context.get('milestone', '')}",
            "quality_alert": f"Review quality concern for {country}: {context.get('issue', '')}",
            "weekly_digest": self._generate_weekly_digest(country)
        }
        
        message = templates.get(notification_type, f"Notification for {country}")
        
        logger.info(f"Notifying CL for {country}: {notification_type}")
        
        return AgentResult(
            success=True,
            data={
                "country": country,
                "notification_type": notification_type,
                "message": message,
                "sent_at": datetime.now().isoformat()
            },
            message=f"Notification sent to CL for {country}"
        )
    
    def _track_contributions(self, country: str) -> AgentResult:
        """Track CL contributions for authorship credit."""
        contributions = self.firebase.get_contributions(country=country)
        
        # Get CL info
        cl_info = self.cl_data.get(country, {})
        if not cl_info:
            return AgentResult(
                success=False,
                data={},
                message=f"No CL data for {country}"
            )
        
        # Calculate contribution metrics
        lead_id = list(cl_info.keys())[0] if cl_info else None
        if not lead_id:
            return AgentResult(
                success=False,
                data={},
                message=f"No CL ID found for {country}"
            )
        
        metrics = {
            "uploads_approved": sum(1 for c in contributions if c.get("status") == "approved"),
            "reviews_completed": 0,  # Would calculate from review logs
            "contributors_recruited": 0,  # Would track from CL actions
            "documentation_contributed": 0
        }
        
        # Calculate authorship score
        authorship_score = self._calculate_authorship_score(metrics)
        
        # Update CL data
        cl_info[lead_id]["contribution_score"] = authorship_score
        cl_info[lead_id]["metrics"] = metrics
        self._save_cl_data()
        
        return AgentResult(
            success=True,
            data={
                "lead_id": lead_id,
                "country": country,
                "metrics": metrics,
                "authorship_score": authorship_score,
                "eligible": authorship_score >= 0.60
            },
            message=f"Contribution tracking: score {authorship_score:.2f}"
        )
    
    def _calculate_authorship_score(self, metrics: Dict[str, int]) -> float:
        """Calculate authorship eligibility score."""
        weights = {
            "uploads": 0.25,
            "reviews": 0.20,
            "recruitment": 0.20,
            "documentation": 0.15,
            "validation": 0.20
        }
        
        scores = {
            "uploads": min(1.0, metrics["uploads_approved"] / 50),
            "reviews": min(1.0, metrics["reviews_completed"] / 100),
            "recruitment": min(1.0, metrics["contributors_recruited"] / 20),
            "documentation": min(1.0, metrics["documentation_contributed"] / 10),
            "validation": 0.0  # Would calculate from validation sessions
        }
        
        return sum(weights[k] * scores[k] for k in weights)
    
    def _check_tier_status(self, country: str) -> AgentResult:
        """Check Tier-0/Tier-1 status for country."""
        contributions = self.firebase.get_contributions(country=country)
        approved = [c for c in contributions if c.get("status") == "approved"]
        
        # Count by category
        categories = {}
        for contrib in approved:
            cat = contrib.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        tier_0_requirements = {
            "total_approved": 50,
            "categories_covered": 6,
            "min_per_category": 5
        }
        
        tier_1_requirements = {
            "total_approved": 200,
            "categories_covered": 8,
            "min_per_category": 15
        }
        
        tier_0_met = (
            len(approved) >= tier_0_requirements["total_approved"] and
            len(categories) >= tier_0_requirements["categories_covered"] and
            all(count >= tier_0_requirements["min_per_category"] for count in categories.values())
        )
        
        tier_1_met = (
            len(approved) >= tier_1_requirements["total_approved"] and
            len(categories) >= tier_1_requirements["categories_covered"] and
            all(count >= tier_1_requirements["min_per_category"] for count in categories.values())
        )
        
        return AgentResult(
            success=True,
            data={
                "country": country,
                "tier_0_met": tier_0_met,
                "tier_1_met": tier_1_met,
                "current_stats": {
                    "total_approved": len(approved),
                    "categories_covered": len(categories),
                    "categories": categories
                },
                "tier_0_progress": {
                    "total": len(approved) / tier_0_requirements["total_approved"],
                    "categories": len(categories) / tier_0_requirements["categories_covered"]
                }
            },
            message=f"Tier status: Tier-0={'✓' if tier_0_met else '✗'}, Tier-1={'✓' if tier_1_met else '✗'}"
        )
    
    def _generate_weekly_digest(self, country: str) -> str:
        """Generate weekly digest for CL."""
        contributions = self.firebase.get_contributions(country=country)
        approved_this_week = [
            c for c in contributions
            if c.get("status") == "approved"
            # Would filter by timestamp for actual week
        ]
        
        return f"""WorldCCUB Weekly Digest - {country}
Week of {datetime.now().strftime('%Y-%m-%d')}

Progress Summary:
- Approved this week: {len(approved_this_week)} images
- Total approved: {len([c for c in contributions if c.get('status') == 'approved'])}
- Active contributors: [to be calculated]

Action Items:
- [To be populated from pending tasks]
"""
