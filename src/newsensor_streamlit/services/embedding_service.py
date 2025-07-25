from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from google import genai
from google.genai.types import EmbedContentConfig

from newsensor_streamlit.config import settings

if TYPE_CHECKING:
    from typing import List


class EmbeddingService:
    """Service for handling Google Gemini embeddings."""
    
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