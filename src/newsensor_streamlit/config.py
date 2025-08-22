from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}
    google_api_key: str = Field(description="Google API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    llama_parse_api_key: str = Field(default="", description="LlamaParse API key")
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_api_key: str = Field(default="", description="Qdrant API key for authentication")
    qdrant_collection: str = Field(default="newsensor_datasheets", description="Qdrant collection name")
    
    # Application mode
    mode: str = Field(default="insert", description="Application mode: 'evaluation' or 'insert'")
    env: str = Field(default="development", description="Environment: 'development' or 'production'")
    
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    docs_dir: Path = Field(default=Path("data/docs"), description="Storage for processed documents")
    uploads_dir: Path = Field(default=Path("data/uploads"), description="Temporary upload directory")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    conversations_dir: Path = Field(default=Path("data/conversations"), description="Conversation history storage")
    
    chunk_size: int = Field(default=512, description="Text chunk size for embedding (optimized for small document set)")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks")
    
    max_context_chunks: int = Field(default=4, description="Maximum context chunks for LLM (reduced for focused retrieval)")
    llm_model: str = Field(default="openai/gpt-4o", description="Default LLM model")
    embedding_model: str = Field(default="models/text-embedding-004", description="Google embedding model")
    
    # Re-ranking settings for enhanced retrieval
    reranking_enabled_default: bool = Field(default=False, description="Default re-ranking state")
    reranking_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2", description="Cross-encoder model")
    reranking_initial_k: int = Field(default=12, description="Initial vector search results (optimized for small dataset)")
    reranking_final_k: int = Field(default=4, description="Final re-ranked results")
    
    ragas_metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        description="RAGAS metrics to calculate"
    )
    
    # RAGAS evaluation settings
    ragas_evaluator_model: str = Field(default="gpt-4o", description="Model for RAGAS evaluation")
    ragas_test_dataset_path: str = Field(default="data/evaluation/test_dataset.csv", description="Path to test dataset")
    ragas_fallback_silent: bool = Field(default=True, description="Silent fallback on RAGAS errors")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

settings = Settings()