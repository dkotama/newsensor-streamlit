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
    qdrant_collection: str = Field(default="newsensor_datasheets", description="Qdrant collection name")
    
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    docs_dir: Path = Field(default=Path("data/docs"), description="Storage for processed documents")
    uploads_dir: Path = Field(default=Path("data/uploads"), description="Temporary upload directory")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    
    chunk_size: int = Field(default=1000, description="Text chunk size for embedding")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    max_context_chunks: int = Field(default=5, description="Maximum context chunks for LLM")
    llm_model: str = Field(default="openai/gpt-4o", description="Default LLM model")
    embedding_model: str = Field(default="models/text-embedding-004", description="Google embedding model")
    
    ragas_metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        description="RAGAS metrics to calculate"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

settings = Settings()