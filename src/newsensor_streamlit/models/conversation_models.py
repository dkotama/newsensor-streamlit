from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class RetrievalMetadata(BaseModel):
    """Metadata for retrieval process."""
    reranking_enabled: bool = False
    reranking_model: Optional[str] = None
    initial_vector_search_count: int = 5
    final_chunk_count: int = 5
    relevance_scores: List[float] = []
    best_relevance_score: float = 0.0
    metadata_filtering_used: bool = False
    detected_models: List[str] = []
    processing_time: float = 0.0
    metadata_filters: Dict[str, Any] = {}


class LegacyMetrics(BaseModel):
    """Legacy metrics structure for backward compatibility."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


class ConversationMessage(BaseModel):
    """Individual message in a conversation."""
    timestamp: str
    question: str
    answer: str
    context: str = ""
    sources: List[str] = []
    ragas_metrics: Dict[str, Any] = {}
    legacy_metrics: LegacyMetrics = LegacyMetrics()
    retrieval_metadata: RetrievalMetadata = RetrievalMetadata()


class RagasSummary(BaseModel):
    """RAGAS evaluation summary for the conversation."""
    total_questions: int = 0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0


class ConversationDocument(BaseModel):
    """Complete conversation document structure."""
    conversation_id: str
    doc_id: str = "collection_wide"
    document_name: str = "Collection" 
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    messages: List[ConversationMessage] = []
    ragas_summary: RagasSummary = RagasSummary()
    
    class Config:
        json_encoders = {
            ObjectId: str
        }
