from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain.schema import Document
from loguru import logger

from newsensor_streamlit.config import settings
from newsensor_streamlit.services.conversation_service import ConversationService
from newsensor_streamlit.services.document_processor import DocumentProcessor
from newsensor_streamlit.services.embedding_service import EmbeddingService
from newsensor_streamlit.services.qdrant_service import QdrantService
from newsensor_streamlit.services.rag_service import RagService

if TYPE_CHECKING:
    from langchain.schema import BaseChatPromptTemplate


class ChatEngine:
    """Main orchestrator for the datasheet Q&A system."""
    
    def __init__(self) -> None:
        self.document_processor = DocumentProcessor(settings.docs_dir)
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.rag_service = RagService()
        self.conversation_service = ConversationService()
        
    def upload_document(self, file_path: str) -> str:
        """Process and upload a new document."""
        logger.info(f"Uploading document: {file_path}")
        
        # Process PDF
        result = self.document_processor.process_pdf(Path(file_path))
        
        # Create chunks
        chunks = self._create_chunks(result["content"], result["source"])
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings(
            [chunk.page_content for chunk in chunks],
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        # Store in vector database
        doc_id = self.qdrant_service.store_document(
            chunks=chunks,
            embeddings=embeddings,
            source_path=result["source"]
        )
        
        # Create conversation for this document
        document_name = Path(file_path).name
        conversation_id = self.conversation_service.create_conversation(doc_id, document_name)
        
        return {"doc_id": doc_id, "conversation_id": conversation_id}
        
    def ask_question(self, question: str, doc_id: str, conversation_id: str = None) -> dict[str, str]:
        """Ask a question about a specific document."""
        # Get relevant context
        context_chunks = self.qdrant_service.search_similar(
            query=question,
            doc_id=doc_id,
            k=settings.max_context_chunks
        )
        
        # Generate answer
        answer = self.rag_service.generate_answer(
            question=question,
            context=context_chunks
        )
        
        # Calculate RAGAS metrics
        metrics = self.rag_service.evaluate_quality(
            question=question,
            answer=answer["answer"],
            context=context_chunks,
            reference_answer=answer.get("reference")
        )
        
        sources = [Path(c.metadata.get("source", "Unknown")).name for c in context_chunks]
        
        # Save to conversation if conversation_id provided
        if conversation_id:
            self.conversation_service.add_message(
                conversation_id=conversation_id,
                question=question,
                answer=answer["answer"],
                sources=sources,
                metrics=metrics,
                context=answer["context"]
            )
        
        return {
            "question": question,
            "answer": answer["answer"],
            "context": answer["context"],
            "metrics": metrics,
            "sources": sources
        }
        
    def _create_chunks(self, text: str, source: str) -> list[Document]:
        """Split text into chunks for embedding."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Create chunks with source metadata
        chunks = text_splitter.create_documents([text])
        
        # Add source metadata to each chunk
        for chunk in chunks:
            chunk.metadata["source"] = source
            
        return chunks
        
    def list_documents(self) -> list[dict[str, str]]:
        """List all available documents."""
        return self.qdrant_service.list_collections()
        
    def get_conversation_history(self, doc_id: str, limit: int = 50) -> list[dict[str, str]]:
        """Get conversation history for a document."""
        return self.qdrant_service.get_conversation_history(doc_id, limit)
    
    def list_conversations(self, doc_id: str = None) -> list[dict]:
        """List conversations, optionally filtered by document."""
        return self.conversation_service.list_conversations(doc_id)
        
    def export_conversation(self, conversation_id: str, format: str = "json") -> str:
        """Export a conversation."""
        return self.conversation_service.export_conversation(conversation_id, format)