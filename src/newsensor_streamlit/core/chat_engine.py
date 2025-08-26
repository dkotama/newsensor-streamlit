from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain.schema import Document
from loguru import logger

from newsensor_streamlit.config import settings
from newsensor_streamlit.services.conversation_service import ConversationService
from newsensor_streamlit.services.document_processor import DocumentProcessor
from newsensor_streamlit.services.embedding_service import EmbeddingService
from newsensor_streamlit.services.qdrant_service import QdrantService
from newsensor_streamlit.services.rag_service import RagService

try:
    from newsensor_streamlit.services.reranking_service import ReRankingService
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False

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
        
        # Initialize re-ranking service if available
        self.reranking_service = None
        if RERANKING_AVAILABLE:
            try:
                self.reranking_service = ReRankingService()
                logger.info("Re-ranking service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize re-ranking service: {e}")
        
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
        
    def upload_document_enhanced(self, file_path: str) -> str:
        """Process and upload a document using enhanced metadata system."""
        logger.info(f"Uploading document with enhanced metadata: {file_path}")
        
        # Process PDF with enhanced metadata extraction
        document = self.document_processor.process_pdf(Path(file_path))
        
        # Create metadata-aware chunks
        chunks = self.document_processor.create_metadata_aware_chunks(
            document, chunk_size=settings.chunk_size
        )
        
        # Generate enhanced embeddings
        embeddings = []
        for chunk in chunks:
            embedding = self.embedding_service.generate_metadata_aware_embedding(chunk)
            embeddings.append(embedding)
        
        # Store in vector database with enhanced metadata
        doc_id = self.qdrant_service.store_enhanced_chunks(
            chunks=chunks,
            embeddings=embeddings,
            source_path=document["source"]
        )
        
        # Create conversation for this document
        document_name = Path(file_path).name
        sensor_model = document["metadata"].get("sensor_model", "unknown")
        manufacturer = document["metadata"].get("manufacturer", "unknown")
        
        conversation_id = self.conversation_service.create_conversation(
            doc_id, 
            f"{document_name} ({sensor_model} - {manufacturer})"
        )
        
        logger.info(f"Enhanced upload completed - Model: {sensor_model}, Manufacturer: {manufacturer}")
        return {
            "doc_id": doc_id, 
            "conversation_id": conversation_id,
            "metadata": {
                "sensor_model": sensor_model,
                "manufacturer": manufacturer,
                "chunks_created": len(chunks)
            }
        }
        
    def ask_question_enhanced(
        self, 
        question: str, 
        sensor_model: str = None,
        manufacturer: str = None,
        doc_id: str = None,
        conversation_id: str = None,
        use_reranking: bool = False
    ) -> dict[str, str]:
        """Ask a question with enhanced metadata filtering."""
        start_time = time.time()
        
        # Determine retrieval strategy with metadata awareness
        if use_reranking and self.reranking_service:
            context_chunks, retrieval_metadata = self._retrieve_with_enhanced_reranking(
                question, sensor_model, manufacturer, doc_id
            )
        else:
            context_chunks, retrieval_metadata = self._retrieve_enhanced_standard(
                question, sensor_model, manufacturer, doc_id
            )
        
        # Generate answer
        answer = self.rag_service.generate_answer(
            question=question,
            context=context_chunks
        )
        
        # Legacy metrics for backward compatibility
        legacy_metrics = self.rag_service.evaluate_quality(
            question=question,
            answer=answer["answer"],
            context=context_chunks,
            reference_answer=answer.get("reference")
        )
        
        sources = [Path(c.metadata.get("source", "Unknown")).name for c in context_chunks]
        
        # Add processing time and metadata context
        total_time = time.time() - start_time
        retrieval_metadata["processing_time"] = round(total_time, 3)
        retrieval_metadata["metadata_filters"] = {
            "sensor_model": sensor_model,
            "manufacturer": manufacturer,
            "doc_id": doc_id
        }
        
        # Save to conversation if conversation_id provided
        if conversation_id:
            self.conversation_service.add_message(
                conversation_id=conversation_id,
                question=question,
                answer=answer["answer"],
                sources=sources,
                metrics={},  # Empty metrics dict to remove metrics from saved conversations
                context=answer["context"],
                ragas_metrics={},  # Empty ragas_metrics dict to remove metrics from saved conversations
                retrieval_metadata={}  # Empty retrieval_metadata dict to remove metadata from saved conversations
            )
        
        return {
            "question": question,
            "answer": answer["answer"],
            "context": answer["context"],
            "metrics": legacy_metrics,
            "sources": sources,
            "retrieval_metadata": retrieval_metadata
        }
    
    def _retrieve_enhanced_standard(self, question: str, sensor_model: str = None, 
                                  manufacturer: str = None, doc_id: str = None) -> tuple[list[Document], dict]:
        """Enhanced metadata-aware standard retrieval."""
        # Use metadata filtering if available
        if sensor_model or manufacturer:
            enhanced_docs = self.qdrant_service.search_with_metadata_filter(
                query=question,
                sensor_model=sensor_model,
                manufacturer=manufacturer,
                k=settings.max_context_chunks
            )
            
            # Convert to LangChain Documents
            context_chunks = []
            for doc_data in enhanced_docs:
                from langchain.schema import Document
                metadata = doc_data.get('metadata', {}).copy()
                metadata["source"] = doc_data.get('source_path', 'Unknown')
                metadata["chunk_index"] = doc_data.get('chunk_index', 0)
                metadata["vector_score"] = doc_data.get('vector_score', 0)
                
                context_chunks.append(
                    Document(
                        page_content=doc_data.get('content', ''),
                        metadata=metadata
                    )
                )
        else:
            # Fall back to standard retrieval
            context_chunks = self.qdrant_service.search_similar(
                query=question,
                doc_id=doc_id or "",
                k=settings.max_context_chunks
            )
        
        metadata = {
            "reranking_enabled": False,
            "reranking_model": None,
            "initial_vector_search_count": len(context_chunks),
            "final_chunk_count": len(context_chunks),
            "relevance_scores": [],
            "best_relevance_score": 0.0,
            "metadata_filtering_used": bool(sensor_model or manufacturer),
            "detected_models": list(set([
                chunk.metadata.get("sensor_model", "unknown") 
                for chunk in context_chunks
            ]))
        }
        
        return context_chunks, metadata
    
    def _retrieve_with_enhanced_reranking(self, question: str, sensor_model: str = None,
                                        manufacturer: str = None, doc_id: str = None) -> tuple[list[Document], dict]:
        """Enhanced two-stage retrieval with metadata filtering and re-ranking."""
        if not self.reranking_service:
            logger.warning("Re-ranking requested but service not available, falling back to enhanced standard retrieval")
            return self._retrieve_enhanced_standard(question, sensor_model, manufacturer, doc_id)
        
        try:
            # Stage 1: Initial vector search with metadata filtering and larger k
            initial_docs = self.qdrant_service.search_for_reranking(
                query=question,
                doc_id=doc_id,
                k=settings.reranking_initial_k,
                sensor_model=sensor_model,
                manufacturer=manufacturer
            )
            
            if not initial_docs:
                logger.warning("No documents found in enhanced initial search")
                return [], {
                    "reranking_enabled": True,
                    "reranking_model": settings.reranking_model,
                    "initial_vector_search_count": 0,
                    "final_chunk_count": 0,
                    "relevance_scores": [],
                    "best_relevance_score": 0.0,
                    "metadata_filtering_used": bool(sensor_model or manufacturer),
                    "detected_models": []
                }
            
            # Stage 2: Re-ranking
            rerank_result = self.reranking_service.rerank_documents(
                query=question,
                documents=initial_docs,
                top_k=settings.reranking_final_k
            )
            
            # Convert back to LangChain Documents
            context_chunks = []
            detected_models = set()
            
            for doc_data in rerank_result.documents:
                from langchain.schema import Document
                metadata = doc_data.get('metadata', {}).copy()
                metadata["source"] = doc_data.get('source_path', 'Unknown')
                metadata["chunk_index"] = doc_data.get('chunk_index', 0)
                
                # Track detected models
                model = doc_data.get('sensor_model', 'unknown')
                if model != 'unknown':
                    detected_models.add(model)
                
                context_chunks.append(
                    Document(
                        page_content=doc_data.get('content', ''),
                        metadata=metadata
                    )
                )
            
            metadata = {
                "reranking_enabled": True,
                "reranking_model": rerank_result.model_used,
                "initial_vector_search_count": len(initial_docs),
                "final_chunk_count": len(rerank_result.documents),
                "relevance_scores": rerank_result.scores,
                "best_relevance_score": rerank_result.best_score,
                "metadata_filtering_used": bool(sensor_model or manufacturer),
                "detected_models": list(detected_models)
            }
            
            logger.info(f"Enhanced re-ranking completed: {len(initial_docs)} → {len(context_chunks)} chunks")
            logger.info(f"Detected models: {list(detected_models)}")
            return context_chunks, metadata
            
        except Exception as e:
            logger.error(f"Enhanced re-ranking failed: {e}, falling back to enhanced standard retrieval")
            return self._retrieve_enhanced_standard(question, sensor_model, manufacturer, doc_id)
        
    def ask_question(
        self, 
        question: str, 
        doc_id: str, 
        conversation_id: str = None,
        use_reranking: bool = False
    ) -> dict[str, str]:
        """Ask a question about a specific document."""
        start_time = time.time()
        
        # Determine retrieval strategy
        if use_reranking and self.reranking_service:
            context_chunks, retrieval_metadata = self._retrieve_with_reranking(
                question, doc_id
            )
        else:
            context_chunks, retrieval_metadata = self._retrieve_standard(
                question, doc_id
            )
        
        # Generate answer
        answer = self.rag_service.generate_answer(
            question=question,
            context=context_chunks
        )
        
        # Legacy metrics for backward compatibility
        legacy_metrics = self.rag_service.evaluate_quality(
            question=question,
            answer=answer["answer"],
            context=context_chunks,
            reference_answer=answer.get("reference")
        )
        
        sources = [Path(c.metadata.get("source", "Unknown")).name for c in context_chunks]
        
        # Add processing time
        total_time = time.time() - start_time
        retrieval_metadata["processing_time"] = round(total_time, 3)
        
        # Save to conversation if conversation_id provided
        if conversation_id:
            self.conversation_service.add_message(
                conversation_id=conversation_id,
                question=question,
                answer=answer["answer"],
                sources=sources,
                metrics={},  # Empty metrics dict to remove metrics from saved conversations
                context=answer["context"],
                ragas_metrics={},  # Empty ragas_metrics dict to remove metrics from saved conversations
                retrieval_metadata={}  # Empty retrieval_metadata dict to remove metadata from saved conversations
            )
        
        return {
            "question": question,
            "answer": answer["answer"],
            "context": answer["context"],
            "metrics": legacy_metrics,
            "sources": sources,
            "retrieval_metadata": retrieval_metadata
        }
    
    def _retrieve_standard(self, question: str, doc_id: str) -> tuple[list[Document], dict]:
        """Standard vector similarity retrieval."""
        context_chunks = self.qdrant_service.search_similar(
            query=question,
            doc_id=doc_id,
            k=settings.max_context_chunks
        )
        
        metadata = {
            "reranking_enabled": False,
            "reranking_model": None,
            "initial_vector_search_count": len(context_chunks),
            "final_chunk_count": len(context_chunks),
            "relevance_scores": [],
            "best_relevance_score": 0.0
        }
        
        return context_chunks, metadata
    
    def _retrieve_with_reranking(self, question: str, doc_id: str) -> tuple[list[Document], dict]:
        """Two-stage retrieval with re-ranking."""
        if not self.reranking_service:
            logger.warning("Re-ranking requested but service not available, falling back to standard retrieval")
            return self._retrieve_standard(question, doc_id)
        
        try:
            # Stage 1: Initial vector search with larger k
            initial_docs = self.qdrant_service.search_for_reranking(
                query=question,
                doc_id=doc_id,
                k=settings.reranking_initial_k
            )
            
            if not initial_docs:
                logger.warning("No documents found in initial search")
                return [], {
                    "reranking_enabled": True,
                    "reranking_model": settings.reranking_model,
                    "initial_vector_search_count": 0,
                    "final_chunk_count": 0,
                    "relevance_scores": [],
                    "best_relevance_score": 0.0
                }
            
            # Stage 2: Re-ranking
            rerank_result = self.reranking_service.rerank_documents(
                query=question,
                documents=initial_docs,
                top_k=settings.reranking_final_k
            )
            
            # Convert back to LangChain Documents
            context_chunks = []
            for doc_data in rerank_result.documents:
                from langchain.schema import Document
                metadata = doc_data.get('metadata', {}).copy()
                metadata["source"] = doc_data.get('source_path', 'Unknown')
                metadata["chunk_index"] = doc_data.get('chunk_index', 0)
                
                context_chunks.append(
                    Document(
                        page_content=doc_data.get('content', ''),
                        metadata=metadata
                    )
                )
            
            metadata = {
                "reranking_enabled": True,
                "reranking_model": rerank_result.model_used,
                "initial_vector_search_count": len(initial_docs),
                "final_chunk_count": len(rerank_result.documents),
                "relevance_scores": rerank_result.scores,
                "best_relevance_score": rerank_result.best_score
            }
            
            logger.info(f"Re-ranking completed: {len(initial_docs)} → {len(context_chunks)} chunks")
            return context_chunks, metadata
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}, falling back to standard retrieval")
            return self._retrieve_standard(question, doc_id)
        
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