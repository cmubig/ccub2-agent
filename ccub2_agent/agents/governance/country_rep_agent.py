"""
Country Representative Agent - Country Rep coordination and management.

Each country has 3 equal Country Representatives who share responsibilities
for data collection oversight, cultural review, and community coordination.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...data.firebase_client import FirebaseClient

logger = logging.getLogger(__name__)

# Country configuration constants
TOTAL_COUNTRIES = 20
REPS_PER_COUNTRY = 3
IMAGES_PER_COUNTRY = 300
MIN_PER_CATEGORY = 30
TOTAL_CATEGORIES = 8

CATEGORIES = [
    "food",
    "clothing",
    "architecture",
    "city",
    "nature",
    "religion",
    "art",
    "people",
]


class CountryRepAgent(BaseAgent):
    """
    Manages Country Representative coordination.

    Responsibilities:
    - Rep recruitment (3 equal slots per country)
    - Onboarding
    - Communication management
    - Contribution tracking
    - Category balance monitoring
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.firebase = FirebaseClient()
        self.rep_data_file = Path(__file__).parent.parent.parent.parent / "data" / "country_reps.json"
        self._load_rep_data()

    def _load_rep_data(self):
        """Load Country Representative data."""
        if self.rep_data_file.exists():
            with open(self.rep_data_file, 'r', encoding='utf-8') as f:
                self.rep_data = json.load(f)
        else:
            self.rep_data = {}

    def _save_rep_data(self):
        """Save Country Representative data."""
        self.rep_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rep_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.rep_data, f, indent=2, ensure_ascii=False)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Manage Country Representative operations.

        Args:
            input_data: {
                "action": str,  # "recruit" | "onboard" | "notify" | "track" | "progress"
                "country": str (optional),
                "rep_id": str (optional),
                "slot": int (optional, 1-3),
                "notification_type": str (optional),
                "context": Dict (optional)
            }

        Returns:
            AgentResult with operation results
        """
        try:
            action = input_data.get("action", "track")
            country = input_data.get("country", self.config.country)

            if action == "recruit":
                return self._recruit_rep(country, input_data.get("candidates", []))
            elif action == "onboard":
                return self._onboard_rep(
                    input_data.get("rep_id", ""),
                    country,
                    input_data.get("slot", None),
                    input_data.get("rep_data", {})
                )
            elif action == "notify":
                return self._notify_reps(
                    country,
                    input_data.get("notification_type", ""),
                    input_data.get("context", {})
                )
            elif action == "track":
                return self._track_contributions(country)
            elif action == "progress":
                return self._check_country_progress(country)
            else:
                return AgentResult(
                    success=False,
                    data={},
                    message=f"Unknown action: {action}"
                )

        except Exception as e:
            logger.error(f"Country Rep execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Country Rep error: {str(e)}"
            )

    def _recruit_rep(self, country: str, candidates: List[Dict[str, Any]]) -> AgentResult:
        """Recruit Country Representative candidates."""
        logger.info(f"Recruiting rep for {country}: {len(candidates)} candidates")

        # Check how many slots are already filled
        country_reps = self.rep_data.get(country, {})
        filled_slots = len(country_reps)
        available_slots = REPS_PER_COUNTRY - filled_slots

        return AgentResult(
            success=True,
            data={
                "country": country,
                "candidates": candidates,
                "filled_slots": filled_slots,
                "available_slots": available_slots,
                "status": "recruitment_initiated"
            },
            message=f"Recruitment initiated for {country} ({available_slots} slots available)"
        )

    def _onboard_rep(self, rep_id: str, country: str, slot: Optional[int], rep_data: Dict[str, Any]) -> AgentResult:
        """Onboard new Country Representative with slot validation."""
        # Validate slot limit (max 3 per country)
        country_reps = self.rep_data.get(country, {})
        if len(country_reps) >= REPS_PER_COUNTRY:
            return AgentResult(
                success=False,
                data={"country": country, "filled_slots": len(country_reps)},
                message=f"Country {country} already has {REPS_PER_COUNTRY} representatives"
            )

        # Determine slot
        if slot is None:
            used_slots = {v.get("slot") for v in country_reps.values()}
            for s in [1, 2, 3]:
                if s not in used_slots:
                    slot = s
                    break

        if not rep_id:
            rep_id = f"REP_{country.upper()}_{slot}_{datetime.now().strftime('%Y%m%d')}"

        rep_info = {
            "rep_id": rep_id,
            "country": country,
            "slot": slot,
            "joined": datetime.now().isoformat(),
            "status": "onboarding",
            **rep_data
        }

        if country not in self.rep_data:
            self.rep_data[country] = {}
        self.rep_data[country][rep_id] = rep_info
        self._save_rep_data()

        logger.info(f"Onboarded Rep {rep_id} (slot {slot}) for {country}")

        return AgentResult(
            success=True,
            data=rep_info,
            message=f"Rep {rep_id} onboarded (slot {slot}/{REPS_PER_COUNTRY})"
        )

    def _notify_reps(self, country: str, notification_type: str, context: Dict[str, Any]) -> AgentResult:
        """Send notification to all Country Representatives for a country."""
        rep_info = self.rep_data.get(country, {})
        if not rep_info:
            return AgentResult(
                success=False,
                data={},
                message=f"No representatives found for {country}"
            )

        templates = {
            "new_job": f"New collection job available for {country}: {context.get('job_title', '')}",
            "milestone": f"Milestone reached for {country}: {context.get('milestone', '')}",
            "quality_alert": f"Review quality concern for {country}: {context.get('issue', '')}",
            "category_balance": self._generate_category_alert(country),
            "weekly_digest": self._generate_weekly_digest(country)
        }

        message = templates.get(notification_type, f"Notification for {country}")

        logger.info(f"Notifying {len(rep_info)} reps for {country}: {notification_type}")

        return AgentResult(
            success=True,
            data={
                "country": country,
                "notification_type": notification_type,
                "message": message,
                "recipients": list(rep_info.keys()),
                "sent_at": datetime.now().isoformat()
            },
            message=f"Notification sent to {len(rep_info)} reps for {country}"
        )

    def _track_contributions(self, country: str) -> AgentResult:
        """Track contributions for a country."""
        contributions = self.firebase.get_contributions(country=country)
        approved = [c for c in contributions if c.get("status") == "approved"]

        # Count by category
        categories: Dict[str, int] = {}
        for contrib in approved:
            cat = contrib.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return AgentResult(
            success=True,
            data={
                "country": country,
                "total_approved": len(approved),
                "total_target": IMAGES_PER_COUNTRY,
                "progress_pct": round(len(approved) / IMAGES_PER_COUNTRY * 100, 1),
                "categories": categories,
                "categories_covered": len(categories),
                "categories_target": TOTAL_CATEGORIES,
            },
            message=f"Progress: {len(approved)}/{IMAGES_PER_COUNTRY} images"
        )

    def _check_country_progress(self, country: str) -> AgentResult:
        """Check country progress toward 300-image target with category balance."""
        contributions = self.firebase.get_contributions(country=country)
        approved = [c for c in contributions if c.get("status") == "approved"]

        # Count by category
        categories: Dict[str, int] = {}
        for contrib in approved:
            cat = contrib.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        total = len(approved)

        # Check category minimums
        categories_meeting_min = sum(
            1 for cat in CATEGORIES if categories.get(cat, 0) >= MIN_PER_CATEGORY
        )

        # Identify underrepresented categories
        underrepresented = [
            {"category": cat, "count": categories.get(cat, 0), "needed": MIN_PER_CATEGORY - categories.get(cat, 0)}
            for cat in CATEGORIES
            if categories.get(cat, 0) < MIN_PER_CATEGORY
        ]

        # Check if any category exceeds 50% of total (imbalance warning)
        imbalance_warnings = []
        if total > 0:
            for cat, count in categories.items():
                if count / total > 0.5:
                    imbalance_warnings.append(
                        f"{cat} has {count}/{total} ({round(count/total*100)}%) - "
                        f"prioritize other categories"
                    )

        target_met = total >= IMAGES_PER_COUNTRY and categories_meeting_min >= TOTAL_CATEGORIES

        return AgentResult(
            success=True,
            data={
                "country": country,
                "total_approved": total,
                "total_target": IMAGES_PER_COUNTRY,
                "progress_pct": round(total / IMAGES_PER_COUNTRY * 100, 1),
                "target_met": target_met,
                "categories": categories,
                "categories_meeting_min": categories_meeting_min,
                "categories_target": TOTAL_CATEGORIES,
                "underrepresented": underrepresented,
                "imbalance_warnings": imbalance_warnings,
            },
            message=(
                f"Progress: {total}/{IMAGES_PER_COUNTRY} images, "
                f"{categories_meeting_min}/{TOTAL_CATEGORIES} categories at minimum"
            )
        )

    def _generate_category_alert(self, country: str) -> str:
        """Generate category balance alert."""
        contributions = self.firebase.get_contributions(country=country)
        approved = [c for c in contributions if c.get("status") == "approved"]

        categories: Dict[str, int] = {}
        for contrib in approved:
            cat = contrib.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        underrepresented = [
            cat for cat in CATEGORIES if categories.get(cat, 0) < MIN_PER_CATEGORY
        ]

        if underrepresented:
            return (
                f"Category balance alert for {country}: "
                f"{len(underrepresented)} categories below minimum ({MIN_PER_CATEGORY}): "
                f"{', '.join(underrepresented)}"
            )
        return f"All categories meet minimum requirements for {country}"

    def _generate_weekly_digest(self, country: str) -> str:
        """Generate weekly digest for Country Representatives."""
        contributions = self.firebase.get_contributions(country=country)
        approved = [c for c in contributions if c.get("status") == "approved"]

        categories: Dict[str, int] = {}
        for contrib in approved:
            cat = contrib.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        cat_summary = "\n".join(
            f"  - {cat}: {categories.get(cat, 0)}/{MIN_PER_CATEGORY}"
            for cat in CATEGORIES
        )

        return f"""WorldCCUB Weekly Digest - {country}
Week of {datetime.now().strftime('%Y-%m-%d')}

Progress Summary:
- Total approved: {len(approved)}/{IMAGES_PER_COUNTRY} images
- Progress: {round(len(approved) / IMAGES_PER_COUNTRY * 100, 1)}%

Category Coverage (min {MIN_PER_CATEGORY} each):
{cat_summary}

Action Items:
- [To be populated from pending tasks]
"""
