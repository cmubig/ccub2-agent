"""
Cultural RAG Retriever

Retrieves relevant cultural knowledge for metric evaluation.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CulturalRAGRetriever:
    """
    RAG-based cultural knowledge retriever.
    
    Retrieves relevant cultural knowledge chunks for evaluation.
    """
    
    def __init__(self, country: str, category: Optional[str] = None):
        """
        Initialize RAG retriever.
        
        Args:
            country: Target country
            category: Optional category filter
        """
        self.country = country
        self.category = category
        self._knowledge_base = None
        
        logger.info(f"CulturalRAGRetriever initialized: country={country}, category={category}")
    
    @property
    def knowledge_base(self):
        """Lazy load knowledge base."""
        if self._knowledge_base is None:
            from ...enhanced_cultural_metric_pipeline import EnhancedCulturalKnowledgeBase
            self._knowledge_base = EnhancedCulturalKnowledgeBase(
                country=self.country,
                category=self.category,
            )
        return self._knowledge_base
    
    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant cultural knowledge.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of knowledge chunks with scores
        """
        chunks = self.knowledge_base.retrieve(
            query=query,
            top_k=top_k,
        )
        
        return [
            {
                "text": chunk.get("text", ""),
                "score": chunk.get("score", 0.0),
                "source": chunk.get("source", ""),
                "section": chunk.get("section", ""),
            }
            for chunk in chunks
        ]
    
    def get_cultural_context(
        self,
        image_description: str,
        country: str,
        category: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Get cultural context for an image.
        
        Args:
            image_description: Description of the image
            country: Target country
            category: Cultural category
            
        Returns:
            Dictionary with cultural context
        """
        query = f"{image_description} {category} {country}"
        
        chunks = self.retrieve_knowledge(query, top_k=top_k)
        
        # Aggregate context
        context_text = "\n".join([chunk["text"] for chunk in chunks])
        
        return {
            "context": context_text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "country": country,
            "category": category,
        }


def create_rag_retriever(
    country: str,
    category: Optional[str] = None,
) -> CulturalRAGRetriever:
    """Create a cultural RAG retriever."""
    return CulturalRAGRetriever(country=country, category=category)
