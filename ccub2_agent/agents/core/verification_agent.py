"""
Verification Agent - Reference relevance verification.

Based on MA-RAG / MAIN-RAG pattern for filtering false positive references
and preventing cultural hallucination.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import sys

from ..base_agent import BaseAgent, AgentConfig, AgentResult

# Add metric module to path for VLM
METRIC_PATH = Path(__file__).parent.parent.parent.parent / "evaluation" / "metrics" / "cultural_metric"
if str(METRIC_PATH) not in sys.path:
    sys.path.insert(0, str(METRIC_PATH))

logger = logging.getLogger(__name__)


class VerificationAgent(BaseAgent):
    """
    Verifies that retrieved references actually support the proposed fix.
    
    Responsibilities:
    - Check reference relevance to detected failure modes
    - Filter false positive references
    - Provide confidence-weighted reference selection
    - Detect "cultural hallucination" (references that don't match)
    
    Based on MA-RAG / MAIN-RAG verification pattern.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._vlm_client = None
        self.relevance_threshold = 0.7  # Configurable
    
    def _get_vlm_client(self):
        """Lazy load VLM client."""
        if self._vlm_client is None:
            try:
                from enhanced_cultural_metric_pipeline import EnhancedVLMClient
                self._vlm_client = EnhancedVLMClient(
                    model_name="Qwen/Qwen3-VL-8B-Instruct",
                    load_in_4bit=True,
                    debug=self.config.verbose
                )
                logger.info("VLM client initialized for verification")
            except Exception as e:
                logger.error(f"Failed to initialize VLM client: {e}")
                raise
        return self._vlm_client
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Verify reference relevance.
        
        Args:
            input_data: {
                "references": List[Dict],  # From Reference Selector
                    # Each ref: {"image_path": str, "score": float, "metadata": Dict}
                "failure_modes": List[Dict],  # From Judge/Metric
                    # Each fm: {"mode": str, "evidence": str, "severity": str}
                "prompt": str,
                "country": str,
                "category": str,
                "original_image_path": str (optional)
            }
            
        Returns:
            AgentResult with verified references and confidence scores
        """
        try:
            references = input_data.get("references", [])
            failure_modes = input_data.get("failure_modes", [])
            prompt = input_data.get("prompt", "")
            country = input_data.get("country", self.config.country)
            category = input_data.get("category", self.config.category)
            original_image = input_data.get("original_image_path")
            
            if not references:
                return AgentResult(
                    success=True,
                    data={
                        "verified_references": [],
                        "filtered_count": 0,
                        "verification_confidence": 0.0
                    },
                    message="No references to verify"
                )
            
            # Verify each reference
            verified = []
            filtered = []
            
            for ref in references:
                verification_result = self._verify_reference(
                    ref=ref,
                    failure_modes=failure_modes,
                    prompt=prompt,
                    country=country,
                    category=category,
                    original_image=original_image
                )
                
                if verification_result["verified"]:
                    verified.append({
                        **ref,
                        "verification_score": verification_result["score"],
                        "verification_reason": verification_result["reason"],
                        "verified": True
                    })
                else:
                    filtered.append({
                        **ref,
                        "verification_score": verification_result["score"],
                        "filter_reason": verification_result["reason"],
                        "verified": False
                    })
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(verified)
            
            return AgentResult(
                success=True,
                data={
                    "verified_references": verified,
                    "filtered_references": filtered,
                    "filtered_count": len(filtered),
                    "verification_confidence": confidence,
                    "verification_rate": len(verified) / len(references) if references else 0.0
                },
                message=f"Verified {len(verified)}/{len(references)} references"
            )
            
        except Exception as e:
            logger.error(f"Verification execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Verification error: {str(e)}"
            )
    
    def _verify_reference(
        self,
        ref: Dict[str, Any],
        failure_modes: List[Dict[str, Any]],
        prompt: str,
        country: str,
        category: Optional[str],
        original_image: Optional[str]
    ) -> Dict[str, Any]:
        """
        Verify a single reference's relevance.
        
        Returns:
            {
                "verified": bool,
                "score": float (0-1),
                "reason": str
            }
        """
        try:
            vlm = self._get_vlm_client()
            ref_image_path = Path(ref.get("image_path", ""))
            
            if not ref_image_path.exists():
                return {
                    "verified": False,
                    "score": 0.0,
                    "reason": "Reference image not found"
                }
            
            # Build verification prompt
            failure_summary = self._summarize_failure_modes(failure_modes)
            
            verification_prompt = f"""Analyze if this reference image is relevant for fixing the cultural issues.

Context:
- Country: {country}
- Category: {category or 'general'}
- Original prompt: "{prompt}"
- Detected issues: {failure_summary}

Reference image metadata: {ref.get('metadata', {})}

Questions:
1. Does this reference image show authentic {country} {category or 'cultural'} elements?
2. Does it address the detected cultural issues?
3. Is it culturally appropriate and accurate?
4. Could using this reference lead to cultural misrepresentation or hallucination?

Respond with:
- RELEVANT: [yes/no]
- SCORE: [0.0-1.0]
- REASON: [brief explanation]"""
            
            # Query VLM
            from PIL import Image
            ref_image = Image.open(ref_image_path).convert("RGB")
            
            response = vlm.query_with_context(
                image=ref_image,
                query=verification_prompt,
                context="",
                max_tokens=200
            )
            
            # Parse response
            verified, score, reason = self._parse_verification_response(response)
            
            # Additional check: compare with original if available
            if original_image and Path(original_image).exists():
                similarity_check = self._check_cultural_similarity(
                    ref_image_path,
                    Path(original_image),
                    country,
                    category
                )
                # Adjust score based on similarity
                score = (score + similarity_check) / 2.0
            
            return {
                "verified": verified and score >= self.relevance_threshold,
                "score": score,
                "reason": reason
            }
            
        except Exception as e:
            logger.warning(f"Reference verification failed: {e}")
            return {
                "verified": False,
                "score": 0.0,
                "reason": f"Verification error: {str(e)}"
            }
    
    def _summarize_failure_modes(self, failure_modes: List[Dict[str, Any]]) -> str:
        """Summarize failure modes for verification prompt."""
        if not failure_modes:
            return "No specific issues detected"
        
        summaries = []
        for fm in failure_modes[:3]:  # Top 3
            mode = fm.get("mode", "unknown")
            evidence = fm.get("evidence", "")[:100]
            summaries.append(f"- {mode}: {evidence}")
        
        return "\n".join(summaries)
    
    def _parse_verification_response(self, response: str) -> tuple[bool, float, str]:
        """Parse VLM verification response."""
        response_lower = response.lower()
        
        # Extract RELEVANT
        relevant = "yes" in response_lower or "relevant: yes" in response_lower
        
        # Extract SCORE
        score = 0.5  # Default
        if "score:" in response_lower:
            try:
                score_part = response_lower.split("score:")[1].split()[0]
                score = float(score_part)
                score = max(0.0, min(1.0, score))
            except:
                pass
        
        # Extract REASON
        reason = "No reason provided"
        if "reason:" in response_lower:
            reason = response_lower.split("reason:")[1].strip()[:200]
        elif len(response) > 50:
            reason = response[:200]
        
        return relevant, score, reason
    
    def _check_cultural_similarity(
        self,
        ref_image: Path,
        original_image: Path,
        country: str,
        category: Optional[str]
    ) -> float:
        """
        Check if reference is culturally similar to original.
        
        Returns similarity score (0-1).
        """
        try:
            vlm = self._get_vlm_client()
            
            from PIL import Image
            ref_img = Image.open(ref_image).convert("RGB")
            orig_img = Image.open(original_image).convert("RGB")
            
            similarity_prompt = f"""Compare these two {country} {category or 'cultural'} images.

Do they show:
1. Similar cultural elements?
2. Compatible time periods (both traditional or both contemporary)?
3. Consistent cultural authenticity?

Respond with SIMILARITY: [0.0-1.0]"""
            
            # Use VLM to compare
            response = vlm.query_with_context(
                image=ref_img,  # Use ref as primary
                query=similarity_prompt,
                context=f"Original image context: {country} {category}",
                max_tokens=100
            )
            
            # Extract similarity score
            if "similarity:" in response.lower():
                try:
                    score_part = response.lower().split("similarity:")[1].split()[0]
                    return max(0.0, min(1.0, float(score_part)))
                except:
                    pass
            
            return 0.5  # Default moderate similarity
            
        except Exception as e:
            logger.warning(f"Cultural similarity check failed: {e}")
            return 0.5
    
    def _calculate_confidence(self, verified_refs: List[Dict[str, Any]]) -> float:
        """Calculate overall verification confidence."""
        if not verified_refs:
            return 0.0
        
        scores = [ref.get("verification_score", 0.0) for ref in verified_refs]
        return sum(scores) / len(scores) if scores else 0.0
