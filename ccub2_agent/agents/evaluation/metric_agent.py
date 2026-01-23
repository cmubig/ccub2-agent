"""
Metric Agent - Cultural Metric Toolkit execution.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import sys

from ..base_agent import BaseAgent, AgentConfig, AgentResult

# Add metric module to path
METRIC_PATH = Path(__file__).parent.parent.parent / "evaluation" / "metrics" / "cultural_metric"
if str(METRIC_PATH) not in sys.path:
    sys.path.insert(0, str(METRIC_PATH))

logger = logging.getLogger(__name__)


class MetricAgent(BaseAgent):
    """
    Executes Cultural Metric Toolkit for detailed cultural fidelity scoring.
    
    Responsibilities:
    - Run multi-dimensional cultural scoring
    - Generate failure mode tags
    - Produce structured rationales
    - Use RAG-enhanced cultural knowledge base
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Lazy initialization of VLM and KB
        self._vlm_client = None
        self._knowledge_base = None
        self._index_dir = None
    
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
                logger.info("VLM client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize VLM client: {e}")
                raise
        return self._vlm_client
    
    def _get_knowledge_base(self):
        """Lazy load cultural knowledge base."""
        if self._knowledge_base is None:
            # Find index directory for this country
            data_root = Path(__file__).parent.parent.parent.parent / "data"
            index_dir = data_root / "cultural_index" / self.config.country
            
            if not index_dir.exists():
                logger.warning(f"Knowledge base not found at {index_dir}, using basic mode")
                return None
            
            try:
                from enhanced_cultural_metric_pipeline import EnhancedCulturalKnowledgeBase
                self._knowledge_base = EnhancedCulturalKnowledgeBase(index_dir)
                self._index_dir = index_dir
                logger.info(f"Knowledge base loaded from {index_dir}")
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
                return None
        return self._knowledge_base
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute cultural metric scoring.
        
        Args:
            input_data: {
                "image_path": str,
                "prompt": str,
                "country": str,
                "category": str (optional),
                "editing_prompt": str (optional),
                "iteration_number": int (optional),
                "previous_scores": Tuple[int, int] (optional)
            }
            
        Returns:
            AgentResult with detailed scores, failure modes, and rationales
        """
        try:
            image_path = Path(input_data["image_path"])
            prompt = input_data["prompt"]
            country = input_data.get("country", self.config.country)
            category = input_data.get("category", self.config.category)
            editing_prompt = input_data.get("editing_prompt")
            iteration_number = input_data.get("iteration_number", 0)
            previous_scores = input_data.get("previous_scores")
            
            # Get VLM client
            vlm = self._get_vlm_client()
            
            # Get knowledge base
            kb = self._get_knowledge_base()
            
            # Retrieve cultural context
            context_text = ""
            if kb:
                try:
                    # Create sample for retrieval
                    from enhanced_cultural_metric_pipeline import EnhancedCulturalEvalSample
                    sample = EnhancedCulturalEvalSample(
                        uid="metric_eval",
                        group_id="single",
                        step="0",
                        prompt=prompt,
                        country=country,
                        image_path=image_path,
                        editing_prompt=editing_prompt,
                        category=category
                    )
                    docs = kb.retrieve_contextual(prompt, sample, top_k=5)
                    context_text = "\n".join(f"[Doc {idx}] {doc.text}" for idx, doc in enumerate(docs, 1))
                    logger.debug(f"Retrieved {len(docs)} context documents")
                except Exception as e:
                    logger.warning(f"Failed to retrieve context: {e}")
            
            # Evaluate cultural scores
            previous_cultural = previous_scores[0] if previous_scores else None
            previous_prompt = previous_scores[1] if previous_scores else None
            
            cultural_score, prompt_score = vlm.evaluate_cultural_scores(
                image_path=image_path,
                prompt=prompt,
                editing_prompt=editing_prompt or "",
                context=context_text,
                country=country,
                iteration_number=iteration_number,
                previous_cultural_score=previous_cultural,
                previous_prompt_score=previous_prompt
            )
            
            # Detect failure modes using VLM
            failure_modes = self._detect_failure_modes(
                vlm, image_path, prompt, country, category, context_text
            )
            
            # Generate structured rationale
            rationale = self._generate_rationale(
                cultural_score, prompt_score, failure_modes, context_text
            )
            
            return AgentResult(
                success=True,
                data={
                    "cultural_score": cultural_score,
                    "prompt_score": prompt_score,
                    "overall_score": (cultural_score + prompt_score) / 2.0,
                    "failure_modes": failure_modes,
                    "rationale": rationale,
                    "context_used": len(context_text) > 0,
                    "iteration": iteration_number
                },
                message=f"Cultural metric: {cultural_score}/10, Prompt: {prompt_score}/10"
            )
            
        except Exception as e:
            logger.error(f"Metric execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Metric error: {str(e)}"
            )
    
    def _detect_failure_modes(
        self, vlm, image_path: Path, prompt: str, country: str, 
        category: Optional[str], context: str
    ) -> List[Dict[str, Any]]:
        """Detect specific failure modes."""
        failure_modes = []
        
        try:
            # Use VLM to detect failure modes
            detection_prompt = f"""Analyze this {country} {category or 'cultural'} image for cultural accuracy issues.

Check for these failure modes:
1. DE_IDENTIFICATION: Country/culture of origin unrecognizable
2. OVER_MODERNIZATION: Traditional elements replaced with modern ones
3. SUPERFICIAL_CUES: Only surface-level cultural markers
4. STEREOTYPE_RELIANCE: Generic stereotypes instead of authentic details
5. CULTURAL_CONFUSION: Mixed cultural elements

For each detected issue, provide:
- Mode name
- Evidence
- Severity (HIGH/MEDIUM/LOW)

Format: MODE: [name] | EVIDENCE: [observation] | SEVERITY: [level]"""
            
            response = vlm.query_with_context(
                image=image_path,
                query=detection_prompt,
                context=context,
                max_tokens=300
            )
            
            # Parse response
            for line in response.split('\n'):
                if 'MODE:' in line:
                    parts = line.split('|')
                    mode_info = {}
                    for part in parts:
                        if 'MODE:' in part:
                            mode_info['mode'] = part.split('MODE:')[1].strip()
                        elif 'EVIDENCE:' in part:
                            mode_info['evidence'] = part.split('EVIDENCE:')[1].strip()
                        elif 'SEVERITY:' in part:
                            mode_info['severity'] = part.split('SEVERITY:')[1].strip()
                    
                    if mode_info:
                        failure_modes.append(mode_info)
            
        except Exception as e:
            logger.warning(f"Failure mode detection failed: {e}")
        
        return failure_modes
    
    def _generate_rationale(
        self, cultural_score: int, prompt_score: int,
        failure_modes: List[Dict], context: str
    ) -> str:
        """Generate structured rationale for scores."""
        rationale_parts = []
        
        rationale_parts.append(
            f"Cultural authenticity score: {cultural_score}/10. "
            f"Prompt alignment score: {prompt_score}/10."
        )
        
        if failure_modes:
            rationale_parts.append(f"Detected {len(failure_modes)} failure mode(s):")
            for i, fm in enumerate(failure_modes[:3], 1):  # Top 3
                mode = fm.get('mode', 'unknown')
                evidence = fm.get('evidence', '')[:100]
                rationale_parts.append(f"{i}. {mode}: {evidence}")
        else:
            rationale_parts.append("No significant failure modes detected.")
        
        if context:
            rationale_parts.append("Cultural context from knowledge base was used in evaluation.")
        
        return " ".join(rationale_parts)
