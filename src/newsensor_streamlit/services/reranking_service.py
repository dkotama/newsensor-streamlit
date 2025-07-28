"""
Re-ranking service for improving document retrieval relevance using cross-encoder models.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
import torch

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of document re-ranking."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    processing_time: float
    model_used: str
    best_score: float


class ReRankingService:
    """Service for re-ranking retrieved documents using cross-encoder models."""
    
    def __init__(self):
        self._model: Optional[CrossEncoder] = None
        self._model_name = settings.reranking_model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model(self) -> None:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self._model_name}")
            start_time = time.time()
            
            try:
                self._model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                    max_length=512  # Optimized for our chunk size
                )
                load_time = time.time() - start_time
                logger.info(f"Cross-encoder model loaded in {load_time:.2f}s on {self._device}")
                
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> RerankResult:
        """
        Re-rank documents using cross-encoder model.
        
        Args:
            query: User query
            documents: List of document chunks with content and metadata
            top_k: Number of top documents to return (default: reranking_final_k)
            
        Returns:
            RerankResult with re-ranked documents and scores
        """
        start_time = time.time()
        
        if not documents:
            return RerankResult(
                documents=[],
                scores=[],
                processing_time=0.0,
                model_used=self._model_name,
                best_score=0.0
            )
        
        # Use default top_k if not specified
        if top_k is None:
            top_k = settings.reranking_final_k
        
        # Ensure we don't request more than available
        top_k = min(top_k, len(documents))
        
        try:
            # Load model if needed
            self._load_model()
            
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for doc in documents:
                content = doc.get('content', '')
                if isinstance(content, str) and content.strip():
                    query_doc_pairs.append([query, content])
                else:
                    # Handle case where content might be missing or empty
                    query_doc_pairs.append([query, str(doc)])
            
            if not query_doc_pairs:
                logger.warning("No valid query-document pairs for re-ranking")
                return RerankResult(
                    documents=[],
                    scores=[],
                    processing_time=time.time() - start_time,
                    model_used=self._model_name,
                    best_score=0.0
                )
            
            # Get relevance scores
            logger.debug(f"Re-ranking {len(query_doc_pairs)} documents for query: {query[:50]}...")
            scores = self._model.predict(query_doc_pairs)
            
            # Convert to list if numpy array
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            # Create scored documents list
            scored_docs = list(zip(documents, scores))
            
            # Sort by relevance score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k results
            top_docs = scored_docs[:top_k]
            
            # Extract documents and scores
            reranked_documents = [doc for doc, _ in top_docs]
            reranked_scores = [float(score) for _, score in top_docs]
            
            processing_time = time.time() - start_time
            best_score = max(reranked_scores) if reranked_scores else 0.0
            
            logger.info(
                f"Re-ranked {len(documents)} → {len(reranked_documents)} documents "
                f"in {processing_time:.3f}s, best score: {best_score:.3f}"
            )
            
            return RerankResult(
                documents=reranked_documents,
                scores=reranked_scores,
                processing_time=processing_time,
                model_used=self._model_name,
                best_score=best_score
            )
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # Fallback: return original documents without scores
            processing_time = time.time() - start_time
            return RerankResult(
                documents=documents[:top_k],
                scores=[0.0] * min(top_k, len(documents)),
                processing_time=processing_time,
                model_used=f"{self._model_name} (fallback)",
                best_score=0.0
            )
    
    def is_model_loaded(self) -> bool:
        """Check if the cross-encoder model is loaded."""
        return self._model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self._model_name,
            "device": self._device,
            "loaded": self.is_model_loaded(),
            "cuda_available": torch.cuda.is_available()
        }
