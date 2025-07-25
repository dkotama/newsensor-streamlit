from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from newsensor_streamlit.config import settings

if TYPE_CHECKING:
    from typing import List

    from langchain.schema import Document


class QdrantService:
    """Service for interacting with Qdrant vector database."""
    
    def __init__(self) -> None:
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection_name = settings.qdrant_collection
        self._ensure_collection()
        
    def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
            
    def store_document(self, chunks: List[Document], embeddings: List[List[float]], source_path: str) -> str:
        """Store document chunks and embeddings."""
        doc_id = f"doc_{hash(source_path)}"
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=idx + 1_000_000,  # Use sequential integers starting from 1M
                vector=embedding,
                payload={
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "source_path": source_path,
                    "chunk_index": idx,
                    "doc_id": doc_id
                }
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return doc_id
        
    def search_similar(self, query: str, doc_id: str, k: int = 5) -> List[Document]:
        """Search for similar context chunks."""
        try:
            from newsensor_streamlit.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            query_embedding = embedding_service.generate_embedding(
                query, task_type="RETRIEVAL_QUERY"
            )
            
            # Search for similar documents
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                query_filter={
                    "must": [
                        {"key": "doc_id", "match": {"value": doc_id}}
                    ]
                } if doc_id else None
            )
            
            # Convert to Documents
            documents = []
            for hit in search_result:
                if hit.payload:
                    from langchain.schema import Document
                    documents.append(
                        Document(
                            page_content=hit.payload.get("content", ""),
                            metadata=hit.payload.get("metadata", {})
                        )
                    )
            
            logger.info(f"Found {len(documents)} relevant chunks for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
        
    def list_collections(self) -> List[dict[str, str]]:
        """List all documents in the collection."""
        try:
            collections = self.client.get_collections()
            return [{"id": c.name, "name": c.name} for c in collections.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
            
    def get_conversation_history(self, doc_id: str, limit: int = 50) -> List[dict[str, str]]:
        """Get conversation history for a document."""
        return []