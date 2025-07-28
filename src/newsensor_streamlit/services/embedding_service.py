from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

from loguru import logger
from google import genai
from google.genai.types import EmbedContentConfig

from newsensor_streamlit.config import settings

if TYPE_CHECKING:
    from typing import List


class EmbeddingService:
    """Service for handling Google Gemini embeddings with metadata enhancement."""
    
    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.google_api_key)
        
    def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Generate embedding for a single text."""
        result = self.client.models.embed_content(
            model=settings.embedding_model,
            contents=[text],
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768  # Optimized size
            )
        )
        
        return result.embeddings[0].values
    
    def generate_metadata_aware_embedding(self, chunk: Dict[str, Any]) -> List[float]:
        """Generate embedding with metadata context for better retrieval."""
        content = chunk["content"]
        metadata = chunk["metadata"]
        
        # Create enhanced text with metadata context
        enhanced_text = self._create_enhanced_text(content, metadata)
        
        return self.generate_embedding(enhanced_text, "RETRIEVAL_DOCUMENT")
    
    def _create_enhanced_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create enhanced text by prepending relevant metadata context."""
        # Extract key identification metadata
        sensor_model = metadata.get("sensor_model", "")
        manufacturer = metadata.get("manufacturer", "")
        sensor_type = metadata.get("sensor_type", "")
        chunk_context = metadata.get("chunk_context", "")
        
        # Create metadata prefix
        metadata_parts = []
        
        if sensor_model and sensor_model != "unknown":
            metadata_parts.append(f"Sensor Model: {sensor_model}")
        
        if manufacturer and manufacturer != "unknown":
            metadata_parts.append(f"Manufacturer: {manufacturer}")
            
        if sensor_type and sensor_type != "unknown":
            metadata_parts.append(f"Type: {sensor_type}")
            
        if chunk_context and chunk_context != "general":
            metadata_parts.append(f"Section: {chunk_context}")
        
        # Add alternate models for searchability
        alternate_models = metadata.get("alternate_models", [])
        if alternate_models:
            model_variants = ", ".join(alternate_models[:3])  # Limit to avoid token bloat
            metadata_parts.append(f"Also known as: {model_variants}")
        
        # Combine metadata prefix with content
        if metadata_parts:
            metadata_prefix = " | ".join(metadata_parts)
            enhanced_text = f"[{metadata_prefix}]\n\n{content}"
        else:
            enhanced_text = content
        
        return enhanced_text
        
    def generate_embeddings(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
            
        processed_texts = [t.strip() for t in texts if t.strip()]
        if not processed_texts:
            return []
            
        result = self.client.models.embed_content(
            model=settings.embedding_model,
            contents=processed_texts,
            config=EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )
        
        return [e.values for e in result.embeddings]
    
    def generate_query_embedding(self, query: str, context_hint: str = None) -> List[float]:
        """Generate optimized query embedding with optional context hint."""
        # Enhance query with context if provided
        if context_hint:
            enhanced_query = f"Query about {context_hint}: {query}"
        else:
            enhanced_query = query
            
        return self.generate_embedding(enhanced_query, "RETRIEVAL_QUERY")