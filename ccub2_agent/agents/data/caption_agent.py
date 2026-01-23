"""
Caption Agent - Caption normalization pipeline.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
import sys
from datetime import datetime

from ..base_agent import BaseAgent, AgentConfig, AgentResult

# Add metric module to path for VLM
METRIC_PATH = Path(__file__).parent.parent.parent.parent / "evaluation" / "metrics" / "cultural_metric"
if str(METRIC_PATH) not in sys.path:
    sys.path.insert(0, str(METRIC_PATH))

logger = logging.getLogger(__name__)


class CaptionAgent(BaseAgent):
    """
    Runs caption normalization pipeline.
    
    Responsibilities:
    - Translate captions to English
    - VLM analysis for detailed descriptions
    - LLM refinement for structured format
    - Schema enforcement
    - Quality validation
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self._vlm_client = None
        self.pipeline_version = "1.0.0"
    
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
                logger.info("VLM client initialized for caption processing")
            except Exception as e:
                logger.error(f"Failed to initialize VLM client: {e}")
                raise
        return self._vlm_client
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process caption normalization.
        
        Args:
            input_data: {
                "item_id": str,
                "image_path": str,
                "caption_raw": str,
                "country": str,
                "category": str (optional),
                "batch_mode": bool (optional)
            }
            
        Returns:
            AgentResult with normalized caption and metadata
        """
        try:
            if input_data.get("batch_mode", False):
                return self._process_batch(input_data)
            else:
                return self._process_single(input_data)
                
        except Exception as e:
            logger.error(f"Caption processing failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                data={},
                message=f"Caption error: {str(e)}"
            )
    
    def _process_single(self, input_data: Dict[str, Any]) -> AgentResult:
        """Process a single caption."""
        item_id = input_data["item_id"]
        image_path = Path(input_data["image_path"])
        caption_raw = input_data["caption_raw"]
        country = input_data.get("country", self.config.country)
        category = input_data.get("category", self.config.category)
        
        # Step 1: Language detection
        caption_lang = self._detect_language(caption_raw)
        
        # Step 2: Translation (if needed)
        caption_translated = None
        if caption_lang != "en":
            caption_translated = self._translate_caption(caption_raw, caption_lang, "en")
        
        # Step 3: VLM analysis
        vlm = self._get_vlm_client()
        caption_vlm = self._vlm_analysis(vlm, image_path, caption_raw, country, category)
        
        # Step 4: LLM refinement (simplified - would use LLM for structured format)
        caption_normalized = self._refine_caption(
            caption_raw, caption_translated, caption_vlm, country, category
        )
        
        # Step 5: Extract cultural elements
        cultural_elements = self._extract_cultural_elements(caption_vlm, category)
        
        # Step 6: Schema validation
        validation_result = self._validate_schema({
            "item_id": item_id,
            "caption_raw": caption_raw,
            "caption_normalized": caption_normalized,
            "cultural_elements": cultural_elements
        })
        
        if not validation_result["valid"]:
            return AgentResult(
                success=False,
                data={"validation_errors": validation_result["errors"]},
                message="Schema validation failed"
            )
        
        # Step 7: Quality check
        confidence_score = self._calculate_confidence(caption_normalized, caption_vlm)
        
        return AgentResult(
            success=True,
            data={
                "item_id": item_id,
                "caption_raw": caption_raw,
                "caption_raw_language": caption_lang,
                "caption_translated": caption_translated,
                "caption_vlm": caption_vlm,
                "caption_normalized": caption_normalized,
                "cultural_elements": cultural_elements,
                "era": self._classify_era(caption_vlm),
                "confidence_score": confidence_score,
                "pipeline_version": self.pipeline_version,
                "processing_timestamp": datetime.now().isoformat()
            },
            message="Caption normalized successfully"
        )
    
    def _process_batch(self, input_data: Dict[str, Any]) -> AgentResult:
        """Process multiple captions in batch."""
        batch_file = Path(input_data.get("batch_file", ""))
        if not batch_file.exists():
            return AgentResult(
                success=False,
                data={},
                message=f"Batch file not found: {batch_file}"
            )
        
        # Load batch items
        items = []
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        # Process each item
        results = []
        for item in items:
            result = self._process_single(item)
            if result.success:
                results.append(result.data)
        
        # Save results
        output_file = input_data.get("output_file", batch_file.parent / f"{batch_file.stem}_normalized.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        return AgentResult(
            success=True,
            data={
                "processed": len(results),
                "total": len(items),
                "output_file": str(output_file)
            },
            message=f"Processed {len(results)}/{len(items)} captions"
        )
    
    def _detect_language(self, text: str) -> str:
        """Detect language of caption."""
        # Simplified: check for common patterns
        # In real implementation, would use language detection library
        if any(ord(char) > 127 for char in text[:50]):
            # Contains non-ASCII, likely Korean/Chinese/Japanese
            if any('\uAC00' <= char <= '\uD7A3' for char in text):  # Hangul
                return "ko"
            elif any('\u4E00' <= char <= '\u9FFF' for char in text):  # CJK
                return "zh"
            elif any('\u3040' <= char <= '\u309F' for char in text):  # Hiragana
                return "ja"
        return "en"
    
    def _translate_caption(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate caption (simplified - would use translation API)."""
        # Placeholder: in real implementation, would use translation service
        logger.info(f"Translation {source_lang} -> {target_lang} (placeholder)")
        return text  # Return original for now
    
    def _vlm_analysis(
        self, vlm, image_path: Path, caption: str, country: str, category: Optional[str]
    ) -> str:
        """Use VLM to generate detailed visual description."""
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            
            prompt = f"""Look at this {country} {category or 'cultural'} image. The original caption is: "{caption}"

Describe what you see in the image in English (2-3 sentences):
1. Describe the visual details: objects, people, colors, setting, actions
2. Use cultural terms with English translations when relevant
3. Be specific and descriptive about what is visible
4. Focus on describing what you observe, not analyzing cultural significance

Output only the image description."""
            
            response = vlm.query_with_context(
                image=image,
                query=prompt,
                context="",
                max_tokens=200
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return caption  # Fallback to original
    
    def _refine_caption(
        self, raw: str, translated: Optional[str], vlm_desc: str,
        country: str, category: Optional[str]
    ) -> str:
        """Refine caption into structured format."""
        # Combine information into normalized caption
        parts = []
        
        # Use VLM description as base
        if vlm_desc:
            parts.append(vlm_desc)
        elif translated:
            parts.append(translated)
        else:
            parts.append(raw)
        
        # Add country context
        normalized = f"{parts[0]} (Traditional {country} {category or 'cultural'} element)."
        
        # Ensure length constraints
        if len(normalized) > 500:
            normalized = normalized[:497] + "..."
        
        return normalized
    
    def _extract_cultural_elements(self, vlm_desc: str, category: Optional[str]) -> List[Dict[str, Any]]:
        """Extract cultural elements from VLM description."""
        elements = []
        
        # Simplified extraction (would use NLP for better parsing)
        # Look for common cultural terms
        cultural_keywords = {
            "traditional_clothing": ["hanbok", "jeogori", "chima", "goreum", "dongjeong"],
            "food": ["kimchi", "bulgogi", "banchan", "rice"],
            "architecture": ["palace", "temple", "hanok", "roof"]
        }
        
        keywords = cultural_keywords.get(category or "", [])
        vlm_lower = vlm_desc.lower()
        
        for keyword in keywords:
            if keyword in vlm_lower:
                elements.append({
                    "name": keyword,
                    "type": category or "general",
                    "description": f"Found in description: {keyword}",
                    "authenticity": 0.9  # Placeholder
                })
        
        return elements
    
    def _classify_era(self, vlm_desc: str) -> str:
        """Classify era from description."""
        desc_lower = vlm_desc.lower()
        if any(word in desc_lower for word in ["traditional", "ancient", "historic"]):
            return "traditional"
        elif any(word in desc_lower for word in ["modern", "contemporary", "current"]):
            return "contemporary"
        else:
            return "historical"
    
    def _validate_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate caption against schema v1.0."""
        errors = []
        
        required_fields = ["item_id", "caption_raw", "caption_normalized"]
        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Check caption length
        if "caption_normalized" in data:
            caption = data["caption_normalized"]
            if len(caption) < 50:
                errors.append("Caption too short (min 50 chars)")
            if len(caption) > 500:
                errors.append("Caption too long (max 500 chars)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _calculate_confidence(self, normalized: str, vlm_desc: str) -> float:
        """Calculate confidence score for normalization."""
        # Simplified: based on length and VLM description quality
        if not normalized or not vlm_desc:
            return 0.5
        
        length_score = min(1.0, len(normalized) / 200)  # Prefer 200+ chars
        vlm_score = 0.9 if len(vlm_desc) > 100 else 0.7
        
        return (length_score + vlm_score) / 2.0
