from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Union
from enum import Enum

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class PDFParserBackend(str, Enum):
    """Available PDF parser backends."""
    LLAMAPARSE = "llamaparse"
    PYMUPDF4LLM = "pymupdf4llm"
    PYPDF2 = "pypdf2"


class ResponseLanguage(str, Enum):
    """Available response languages."""
    INDONESIAN = "id"
    ENGLISH = "en"


class RAGProvider(str, Enum):
    """Available RAG providers."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}
    
    # API Keys
    google_api_key: str = Field(description="Google API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    llama_parse_api_key: str = Field(default="", description="LlamaParse API key")
    
    # Database Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_api_key: str = Field(default="", description="Qdrant API key for authentication")
    qdrant_collection: str = Field(default="newsensor_datasheets", description="Qdrant collection name")
    
    # MongoDB Configuration for Conversations
    mongodb_uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URI")
    mongodb_database: str = Field(default="newsensor_conversations", description="MongoDB database name")
    mongodb_conversations_collection: str = Field(default="conversations", description="MongoDB collection for conversations")
    
    # Application mode
    mode: str = Field(default="insert", description="Application mode: 'evaluation' or 'insert'")
    env: str = Field(default="development", description="Environment: 'development' or 'production'")
    
    # PDF Parser Configuration
    pdf_parser_backend: str = Field(default="pymupdf4llm", description="Primary PDF parser backend")
    pymupdf_write_images: bool = Field(default=False, description="Write images to files")
    pymupdf_embed_images: bool = Field(default=False, description="Embed images in markdown")
    pymupdf_page_chunks: bool = Field(default=False, description="Chunk content by pages")
    pymupdf_show_progress: bool = Field(default=False, description="Show processing progress")
    pdf_parser_fallback_chain: str = Field(default="pymupdf4llm,pypdf2", description="Fallback parsers")
    
    # Language Configuration
    response_language: str = Field(default="id", description="Language for LLM responses")
    
    # RAG Model Configuration
    rag_provider: str = Field(default="openai", description="RAG provider (openai or openrouter)")
    rag_chat_model: str = Field(default="gpt-4o-mini", description="Model for RAG chat responses")
    rag_temperature: float = Field(default=0.1, description="Temperature for RAG model")
    # Directory Paths
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    docs_dir: Path = Field(default=Path("data/docs"), description="Storage for processed documents")
    uploads_dir: Path = Field(default=Path("data/uploads"), description="Temporary upload directory")
    cache_dir: Path = Field(default=Path("data/cache"), description="Cache directory")
    conversations_dir: Path = Field(default=Path("data/conversations"), description="Conversation history storage")
    
    # Chunking Configuration
    chunk_size: int = Field(default=512, description="Text chunk size for embedding (optimized for small document set)")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks")
    
    # Context Configuration - Handle both int and percentage string
    max_context_chunks: Union[int, str] = Field(default=4, description="Maximum context chunks for LLM (can be int or percentage like '5%')")
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

    @validator('max_context_chunks', pre=True)
    def validate_max_context_chunks(cls, v):
        """Handle both int and percentage string values for max_context_chunks"""
        if isinstance(v, str) and v.endswith('%'):
            return v  # Keep percentage strings as-is
        try:
            return int(v)  # Convert to int if possible
        except (ValueError, TypeError):
            return v  # Return as-is if conversion fails

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data_dir.mkdir(exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
    
    @property
    def pdf_parser_fallback_list(self) -> List[PDFParserBackend]:
        """Get fallback chain as list of enum values."""
        return [
            PDFParserBackend(backend.strip()) 
            for backend in self.pdf_parser_fallback_chain.split(",") 
            if backend.strip()
        ]
    
    @property
    def response_language_enum(self) -> ResponseLanguage:
        """Get response language as enum value."""
        try:
            return ResponseLanguage(self.response_language)
        except ValueError:
            return ResponseLanguage.INDONESIAN
    
    @property 
    def rag_provider_enum(self) -> RAGProvider:
        """Get RAG provider as enum value."""
        try:
            return RAGProvider(self.rag_provider)
        except ValueError:
            return RAGProvider.OPENAI

settings = Settings()