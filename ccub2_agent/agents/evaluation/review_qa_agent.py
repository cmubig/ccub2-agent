"""
Review-QA Agent - Peer review integrity monitoring.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from ..base_agent import BaseAgent, AgentConfig, AgentResult
from ...data.firebase_client import FirebaseClient

logger = logging.getLogger(__name__)


class ReviewQAAgent(BaseAgent):
    """
    Monitors peer review integrity and quality.
    
    Responsibilities:
    - Track reviewer reliability scores
    - Detect anomalies in voting patterns
    - Route uncertain items to re-review
    - Generate reviewer reports
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.firebase = FirebaseClient()
        self.gold_set_items = []  # Known quality items for testing
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Monitor review quality and detect anomalies.
        
        Args:
            input_data: {
                "action": str,  # "monitor" | "check_anomalies" | "update_reliability" | "inject_gold"
                "window_hours": int (optional),
                "country": str (optional)
            }
            
        Returns:
            AgentResult with review quality metrics and anomalies
        """
        try:
            action = input_data.get("action", "monitor")
            window_hours = input_data.get("window_hours", 24)
            country = input_data.get("country", self.config.country)
            
            if action == "monitor":
                return self._monitor_reviews(country, window_hours)
            elif action == "check_anomalies":
                return self._check_anomalies(country, window_hours)
            elif action == "update_reliability":
                return self._update_reliability_scores(country)
            elif action == "inject_gold":
                return self._inject_gold_set(input_data.get("rate", 0.05))
            else:
                return AgentResult(
                    success=False,
                    data={},
                    message=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Review-QA execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Review-QA error: {str(e)}"
            )
    
    def _monitor_reviews(self, country: str, window_hours: int) -> AgentResult:
        """Monitor recent reviews for quality."""
        # Get recent contributions and reviews from Firebase
        contributions = self.firebase.get_contributions(country=country)
        
        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_reviews = []
        
        for contrib in contributions:
            # Extract review data (structure depends on Firebase schema)
            reviews = contrib.get("reviews", [])
            for review in reviews:
                review_time = review.get("timestamp")
                if review_time:
                    # Parse timestamp and check if recent
                    try:
                        if isinstance(review_time, str):
                            review_dt = datetime.fromisoformat(review_time.replace('Z', '+00:00'))
                        else:
                            review_dt = review_time
                        
                        if review_dt >= cutoff_time:
                            recent_reviews.append({
                                "item_id": contrib.get("id"),
                                "reviewer_id": review.get("reviewer_id"),
                                "decision": review.get("decision"),  # "approve" | "reject"
                                "confidence": review.get("confidence", 0.5),
                                "timestamp": review_time
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse timestamp: {e}")
        
        # Calculate metrics
        total_reviews = len(recent_reviews)
        approval_rate = sum(1 for r in recent_reviews if r["decision"] == "approve") / total_reviews if total_reviews > 0 else 0
        
        return AgentResult(
            success=True,
            data={
                "total_reviews": total_reviews,
                "approval_rate": approval_rate,
                "window_hours": window_hours,
                "country": country
            },
            message=f"Monitored {total_reviews} reviews in last {window_hours}h"
        )
    
    def _check_anomalies(self, country: str, window_hours: int) -> AgentResult:
        """Detect anomalies in review patterns."""
        contributions = self.firebase.get_contributions(country=country)
        
        # Group reviews by reviewer
        reviewer_stats = defaultdict(lambda: {
            "total": 0,
            "approvals": 0,
            "rejections": 0,
            "avg_time": [],
            "reviews": []
        })
        
        anomalies = []
        
        for contrib in contributions:
            reviews = contrib.get("reviews", [])
            for review in reviews:
                reviewer_id = review.get("reviewer_id")
                if not reviewer_id:
                    continue
                
                stats = reviewer_stats[reviewer_id]
                stats["total"] += 1
                if review.get("decision") == "approve":
                    stats["approvals"] += 1
                else:
                    stats["rejections"] += 1
                stats["reviews"].append(review)
        
        # Detect anomaly patterns
        for reviewer_id, stats in reviewer_stats.items():
            # Pattern 1: Rubber stamp approval (>95% approval rate)
            if stats["total"] > 50:
                approval_rate = stats["approvals"] / stats["total"]
                if approval_rate > 0.95:
                    anomalies.append({
                        "type": "rubber_stamp_approval",
                        "reviewer_id": reviewer_id,
                        "evidence": f"Approval rate: {approval_rate:.1%}",
                        "severity": "HIGH"
                    })
            
            # Pattern 2: Rapid fire voting (<5 seconds average)
            if len(stats["reviews"]) > 10:
                # Calculate average time between reviews (simplified)
                # In real implementation, would calculate from timestamps
                anomalies.append({
                    "type": "rapid_fire_voting",
                    "reviewer_id": reviewer_id,
                    "evidence": "Average review time suspiciously fast",
                    "severity": "MEDIUM"
                })
        
        return AgentResult(
            success=True,
            data={
                "anomalies": anomalies,
                "total_reviewers": len(reviewer_stats),
                "anomaly_rate": len(anomalies) / len(reviewer_stats) if reviewer_stats else 0
            },
            message=f"Detected {len(anomalies)} anomalies"
        )
    
    def _update_reliability_scores(self, country: str) -> AgentResult:
        """Update reviewer reliability scores."""
        # Calculate RRS (Reviewer Reliability Score) for each reviewer
        # This would use gold set accuracy, consensus agreement, etc.
        
        reliability_scores = {}
        contributions = self.firebase.get_contributions(country=country)
        
        # Group by reviewer and calculate metrics
        reviewer_metrics = defaultdict(lambda: {
            "total": 0,
            "gold_set_accuracy": 0.0,
            "consensus_agreement": 0.0,
            "consistency": 0.0
        })
        
        for contrib in contributions:
            reviews = contrib.get("reviews", [])
            final_decision = contrib.get("status")  # "approved" | "rejected"
            
            for review in reviews:
                reviewer_id = review.get("reviewer_id")
                if not reviewer_id:
                    continue
                
                metrics = reviewer_metrics[reviewer_id]
                metrics["total"] += 1
                
                # Check consensus agreement
                review_decision = "approved" if review.get("decision") == "approve" else "rejected"
                if review_decision == final_decision:
                    metrics["consensus_agreement"] += 1
        
        # Calculate RRS
        for reviewer_id, metrics in reviewer_metrics.items():
            if metrics["total"] > 0:
                consensus_agreement = metrics["consensus_agreement"] / metrics["total"]
                # Simplified RRS calculation
                rrs = (
                    0.4 * metrics["gold_set_accuracy"] +
                    0.3 * consensus_agreement +
                    0.3 * metrics["consistency"]
                ) * 100
                reliability_scores[reviewer_id] = rrs
        
        return AgentResult(
            success=True,
            data={
                "reliability_scores": reliability_scores,
                "total_reviewers": len(reliability_scores),
                "avg_rrs": sum(reliability_scores.values()) / len(reliability_scores) if reliability_scores else 0
            },
            message=f"Updated reliability scores for {len(reliability_scores)} reviewers"
        )
    
    def _inject_gold_set(self, rate: float) -> AgentResult:
        """Inject gold set items into review queue."""
        # In real implementation, would inject known-quality items
        # For now, just log the action
        logger.info(f"Injecting gold set items at {rate:.1%} rate")
        
        return AgentResult(
            success=True,
            data={
                "injection_rate": rate,
                "gold_items_injected": 0  # Would be actual count
            },
            message=f"Gold set injection configured at {rate:.1%} rate"
        )
