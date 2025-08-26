from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional
import re

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue, MatchAny

from newsensor_streamlit.config import settings

if TYPE_CHECKING:
    from typing import List

    from langchain.schema import Document


class QdrantService:
    """Service for interacting with Qdrant vector database with enhanced metadata support."""
    
    def __init__(self) -> None:
        # Initialize client based on environment and API key
        try:
            if settings.qdrant_api_key:
                # Check if host looks like a URL (for cloud deployments)
                if settings.qdrant_host.startswith(('http://', 'https://')):
                    self.client = QdrantClient(
                        url=settings.qdrant_host,
                        api_key=settings.qdrant_api_key
                    )
                    logger.info(f"Connected to Qdrant Cloud with API key authentication at {settings.qdrant_host}")
                else:
                    # Construct URL based on environment
                    if settings.env.lower() == "production":
                        url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
                    else:
                        url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
                    
                    self.client = QdrantClient(
                        url=url,
                        api_key=settings.qdrant_api_key
                    )
                    logger.info(f"Connected to Qdrant with API key authentication at {url} (env: {settings.env})")
            else:
                # No API key - use basic connection
                if settings.qdrant_host.startswith(('http://', 'https://')):
                    self.client = QdrantClient(url=settings.qdrant_host)
                    logger.info(f"Connected to Qdrant without authentication at {settings.qdrant_host}")
                else:
                    # Construct URL based on environment
                    if settings.env.lower() == "production":
                        url = f"https://{settings.qdrant_host}:{settings.qdrant_port}"
                    else:
                        url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
                    
                    self.client = QdrantClient(url=url)
                    logger.info(f"Connected to Qdrant without authentication at {url} (env: {settings.env})")
            
            self.collection_name = settings.qdrant_collection
            self._ensure_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            logger.error(f"Connection details - Host: {settings.qdrant_host}, Port: {settings.qdrant_port}, Env: {settings.env}")
            raise
        
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
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection {self.collection_name}: {e}")
            logger.error("This might indicate a connection issue with Qdrant. Please check:")
            logger.error("1. Qdrant server is running")
            logger.error("2. Host and port are correct")
            logger.error("3. API key is valid (if using authentication)")
            logger.error(f"4. Network connectivity to {settings.qdrant_host}:{settings.qdrant_port}")
            raise
    
    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Ensure a specific collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=768,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name}: {e}")
            raise
    
    def store_enhanced_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], source_path: str, collection_name: str = None) -> str:
        """Store enhanced metadata-aware chunks and embeddings."""
        doc_id = f"doc_{hash(source_path)}"
        
        # Use provided collection name or default
        target_collection = collection_name or self.collection_name
        
        # Ensure target collection exists
        if collection_name and collection_name != self.collection_name:
            self._ensure_collection_exists(collection_name)
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            content = chunk["content"]
            metadata = chunk["metadata"]
            
            # Create unique ID based on content hash + index
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            point_id = abs(int(content_hash, 16)) + idx
            
            # Flatten metadata for better filtering
            payload = {
                "content": content,
                "source_path": str(source_path),
                "chunk_index": idx,
                "doc_id": doc_id,
                "timestamp": str(Path(source_path).stat().st_mtime),
                
                # Enhanced metadata fields for filtering
                "sensor_model": metadata.get("sensor_model", "unknown"),
                "manufacturer": metadata.get("manufacturer", "unknown"),
                "sensor_type": metadata.get("sensor_type", "unknown"),
                "chunk_context": metadata.get("chunk_context", "general"),
                "filename": metadata.get("filename", "unknown"),
                "file_stem": metadata.get("file_stem", "unknown"),
                "processing_backend": metadata.get("processing_backend", "unknown"),
                
                # Searchable model variations
                "searchable_models": metadata.get("searchable_models", []),
                "alternate_models": metadata.get("alternate_models", []),
                
                # Nested metadata as JSON strings for complex queries
                "specifications": metadata.get("specifications", {}),
                "features": metadata.get("features", []),
                "applications": metadata.get("applications", []),
                
                # Original metadata for backward compatibility
                "metadata": metadata
            }
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=target_collection,
            points=points
        )
        
        return doc_id
            
    def store_document(self, chunks: List[Document], embeddings: List[List[float]], source_path: str, collection_name: str = None) -> str:
        """Store document chunks and embeddings (legacy method for backward compatibility)."""
        doc_id = f"doc_{hash(source_path)}"
        
        # Use provided collection name or default
        target_collection = collection_name or self.collection_name
        
        # Ensure target collection exists
        if collection_name and collection_name != self.collection_name:
            self._ensure_collection_exists(collection_name)
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create unique ID based on content hash + index
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
            point_id = abs(int(content_hash, 16)) + idx
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "source_path": str(source_path),
                    "chunk_index": idx,
                    "doc_id": doc_id,
                    "timestamp": str(Path(source_path).stat().st_mtime)
                }
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=target_collection,
            points=points
        )
        
        return doc_id
    
    def search_with_multi_sensor_filter(self, query: str, k: int = 5, collection_name: str = None) -> List[Document]:
        """
        Enhanced search that detects multiple sensor models from query and searches for them.
        Simplified to focus only on sensor models, removing manufacturer and sensor_type requirements.
        """
        try:
            from newsensor_streamlit.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            # Extract sensor models from the query
            detected_sensors = self._extract_sensor_models_from_query(query)
            
            # Create context hint for better query embedding
            context_hint = None
            if detected_sensors:
                context_hint = f"sensors: {', '.join(detected_sensors)}"
                logger.info(f"Detected sensor models in query: {detected_sensors}")
            
            query_embedding = embedding_service.generate_query_embedding(query, context_hint)
            target_collection = collection_name or self.collection_name
            
            # Build metadata filter for multiple sensors
            search_filter = None
            if detected_sensors:
                # Create OR conditions for multiple sensor models
                sensor_conditions = []
                for sensor in detected_sensors:
                    # Try multiple variations of the sensor name
                    sensor_variations = [sensor, sensor.upper(), sensor.lower()]
                    
                    # Add conditions for exact match and searchable_models array
                    sensor_conditions.extend([
                        FieldCondition(key="sensor_model", match=MatchValue(value=sensor)),
                        FieldCondition(key="searchable_models", match=MatchAny(any=sensor_variations)),
                        FieldCondition(key="alternate_models", match=MatchAny(any=sensor_variations))
                    ])
                
                # Use 'should' for OR logic between different sensor models
                search_filter = Filter(should=sensor_conditions) if sensor_conditions else None
            
            # Search with metadata filtering
            search_result = self.client.search(
                collection_name=target_collection,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                query_filter=search_filter
            )
            
            result_count = len(search_result)
            if detected_sensors:
                logger.info(f"Multi-sensor filtered search returned {result_count} results for sensors: {detected_sensors}")
            else:
                logger.info(f"Semantic search (no sensors detected) returned {result_count} results")
            
            return self._convert_to_documents(search_result)
            
        except Exception as e:
            logger.error(f"Error in multi-sensor search: {e}")
            return []
    
    def _extract_sensor_models_from_query(self, query: str) -> List[str]:
        """
        Extract sensor model names from the user query using pattern matching.
        Supports common sensor naming patterns like EVA-2311, S15S, LS219, etc.
        """
        detected_sensors = []
        
        # Common sensor model patterns
        patterns = [
            r'\b[A-Z]{2,4}-\d{3,4}\b',  # Pattern like EVA-2311, S15S-1234
            r'\b[A-Z]{2,4}\d{3,4}\b',   # Pattern like EVA2311, S15S1234
            r'\b[A-Z]+\d+[A-Z]*\b',     # Pattern like LS219, PJ85775
            r'\b[A-Z]{1,2}\d{2,3}[A-Z]?\b',  # Pattern like S15S, L219A
        ]
        
        query_upper = query.upper()
        
        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            for match in matches:
                if match not in detected_sensors:
                    detected_sensors.append(match)
        
        # Also check for known sensor models from our database
        known_sensors = self._get_known_sensor_models()
        for sensor in known_sensors:
            if sensor.upper() in query_upper and sensor not in detected_sensors:
                detected_sensors.append(sensor)
        
        return detected_sensors
    
    def _get_known_sensor_models(self) -> List[str]:
        """Get a list of known sensor models from the collection for reference matching."""
        try:
            # Quick scroll to get sample of sensor models
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,  # Sample size
                with_payload=["sensor_model", "searchable_models"],
                with_vectors=False
            )
            
            known_models = set()
            for point in scroll_result[0]:
                if point.payload:
                    # Add direct sensor_model
                    sensor_model = point.payload.get("sensor_model")
                    if sensor_model and sensor_model != "unknown":
                        known_models.add(sensor_model)
                    
                    # Add searchable_models
                    searchable_models = point.payload.get("searchable_models", [])
                    for model in searchable_models:
                        if model and model != "unknown":
                            known_models.add(model)
            
            return list(known_models)
            
        except Exception as e:
            logger.warning(f"Could not retrieve known sensor models: {e}")
            return []
    
    def search_with_metadata_filter(self, query: str, sensor_model: str = None, manufacturer: str = None, 
                                   sensor_type: str = None, k: int = 5, collection_name: str = None) -> List[Document]:
        """
        Legacy method for backward compatibility. Now simplified to focus only on sensor models.
        For multi-sensor detection, use search_with_multi_sensor_filter instead.
        """
        try:
            from newsensor_streamlit.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            # Create context hint for better query embedding
            context_hint = f"sensor model {sensor_model}" if sensor_model else None
            
            query_embedding = embedding_service.generate_query_embedding(query, context_hint)
            
            # Build simplified metadata filter - only sensor model
            search_filter = None
            if sensor_model and sensor_model.lower() != "unknown":
                # Try multiple variations and fields for flexibility
                sensor_variations = [sensor_model, sensor_model.upper(), sensor_model.lower()]
                
                sensor_conditions = [
                    FieldCondition(key="sensor_model", match=MatchValue(value=sensor_model)),
                    FieldCondition(key="searchable_models", match=MatchAny(any=sensor_variations)),
                    FieldCondition(key="alternate_models", match=MatchAny(any=sensor_variations))
                ]
                
                search_filter = Filter(should=sensor_conditions)
            
            # Use provided collection name or default
            target_collection = collection_name or self.collection_name
            
            # Search with metadata filtering
            search_result = self.client.search(
                collection_name=target_collection,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                query_filter=search_filter
            )
            
            logger.info(f"Simplified metadata-filtered search returned {len(search_result)} results")
            return self._convert_to_documents(search_result)
            
        except Exception as e:
            logger.error(f"Error in metadata-filtered search: {e}")
            return []
    
    def search_similar(self, query: str, doc_id: str, k: int = 5, collection_name: str = None) -> List[Document]:
        """Search for similar context chunks (legacy method)."""
        try:
            from newsensor_streamlit.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            query_embedding = embedding_service.generate_embedding(
                query, task_type="RETRIEVAL_QUERY"
            )
            
            # Use provided collection name or default
            target_collection = collection_name or self.collection_name
            
            # Search for similar documents
            search_result = self.client.search(
                collection_name=target_collection,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                query_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ) if doc_id else None
            )
            
            return self._convert_to_documents(search_result)
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def search_for_reranking(self, query: str, doc_id: str = None, k: int = 12, 
                           sensor_model: str = None, manufacturer: str = None) -> List[dict]:
        """
        Search for documents optimized for re-ranking. Simplified to focus only on sensor models.
        Returns raw document data with scores for re-ranking service.
        """
        try:
            from newsensor_streamlit.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            # Create context hint for better query embedding
            context_hint = f"sensor model {sensor_model}" if sensor_model else None
            
            query_embedding = embedding_service.generate_query_embedding(query, context_hint)
            
            # Build simplified filter conditions - only sensor model and doc_id
            filter_conditions = []
            
            if doc_id:
                filter_conditions.append(
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                )
            
            if sensor_model and sensor_model.lower() != "unknown":
                # Use flexible sensor model matching
                sensor_variations = [sensor_model, sensor_model.upper(), sensor_model.lower()]
                
                sensor_conditions = [
                    FieldCondition(key="sensor_model", match=MatchValue(value=sensor_model)),
                    FieldCondition(key="searchable_models", match=MatchAny(any=sensor_variations)),
                    FieldCondition(key="alternate_models", match=MatchAny(any=sensor_variations))
                ]
                
                # If we have other must conditions, add sensor conditions as should
                if filter_conditions:
                    search_filter = Filter(must=filter_conditions, should=sensor_conditions)
                else:
                    search_filter = Filter(should=sensor_conditions)
            else:
                search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Search for similar documents with larger k for re-ranking
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True,
                query_filter=search_filter
            )
            
            # Convert to dict format for re-ranking service
            documents = []
            for hit in search_result:
                if hit.payload:
                    doc_data = {
                        'content': hit.payload.get("content", ""),
                        'metadata': hit.payload.get("metadata", {}),
                        'source_path': hit.payload.get("source_path", "Unknown"),
                        'chunk_index': hit.payload.get("chunk_index", 0),
                        'vector_score': hit.score,  # Original vector similarity score
                        'doc_id': hit.payload.get("doc_id", doc_id),
                        
                        # Enhanced metadata fields
                        'sensor_model': hit.payload.get("sensor_model", "unknown"),
                        'manufacturer': hit.payload.get("manufacturer", "unknown"),
                        'sensor_type': hit.payload.get("sensor_type", "unknown"),
                        'chunk_context': hit.payload.get("chunk_context", "general")
                    }
                    documents.append(doc_data)
            
            logger.info(f"Retrieved {len(documents)} documents for re-ranking (simplified sensor-based filtering)")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents for re-ranking: {e}")
            return []
    
    def _convert_to_enhanced_documents(self, search_result) -> List[Dict[str, Any]]:
        """Convert Qdrant search results to enhanced document format."""
        documents = []
        for hit in search_result:
            if hit.payload:
                doc_data = {
                    'content': hit.payload.get("content", ""),
                    'source_path': hit.payload.get("source_path", "Unknown"),
                    'chunk_index': hit.payload.get("chunk_index", 0),
                    'vector_score': hit.score,
                    'doc_id': hit.payload.get("doc_id", ""),
                    
                    # Enhanced metadata
                    'sensor_model': hit.payload.get("sensor_model", "unknown"),
                    'manufacturer': hit.payload.get("manufacturer", "unknown"),
                    'sensor_type': hit.payload.get("sensor_type", "unknown"),
                    'chunk_context': hit.payload.get("chunk_context", "general"),
                    'filename': hit.payload.get("filename", "unknown"),
                    'specifications': hit.payload.get("specifications", {}),
                    'features': hit.payload.get("features", []),
                    'applications': hit.payload.get("applications", []),
                    
                    # Original metadata for compatibility
                    'metadata': hit.payload.get("metadata", {})
                }
                documents.append(doc_data)
        return documents
    
    def _convert_to_documents(self, search_result) -> List[Document]:
        """Convert Qdrant search results to LangChain Documents."""
        documents = []
        for hit in search_result:
            if hit.payload:
                from langchain.schema import Document
                # Merge the original metadata with the source information
                metadata = hit.payload.get("metadata", {}).copy()
                metadata["source"] = hit.payload.get("source_path", "Unknown")
                metadata["chunk_index"] = hit.payload.get("chunk_index", 0)
                metadata["vector_score"] = hit.score
                
                documents.append(
                    Document(
                        page_content=hit.payload.get("content", ""),
                        metadata=metadata
                    )
                )
        return documents
        
    def list_collections(self) -> List[dict[str, str]]:
        """List all documents in the collection."""
        try:
            collections = self.client.get_collections()
            return [{"id": c.name, "name": c.name} for c in collections.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "exists": True,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.warning(f"Collection {self.collection_name} not found or error: {e}")
            return {
                "exists": False,
                "points_count": 0,
                "vectors_count": 0,
                "status": "not_found"
            }
    
    def get_available_sensors(self) -> List[Dict[str, str]]:
        """Get list of available sensors in the collection."""
        try:
            # Use scroll to get all points with metadata
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Get up to 1000 points
                with_payload=True,
                with_vectors=False
            )
            
            # Extract unique sensor models and manufacturers
            sensors = {}
            for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                if point.payload:
                    sensor_model = point.payload.get("sensor_model", "unknown")
                    manufacturer = point.payload.get("manufacturer", "unknown")
                    filename = point.payload.get("filename", "unknown")
                    
                    if sensor_model != "unknown":
                        key = f"{sensor_model}_{manufacturer}"
                        if key not in sensors:
                            sensors[key] = {
                                "sensor_model": sensor_model,
                                "manufacturer": manufacturer,
                                "filename": filename,
                                "chunk_count": 0
                            }
                        sensors[key]["chunk_count"] += 1
            
            return list(sensors.values())
            
        except Exception as e:
            logger.error(f"Error getting available sensors: {e}")
            return []
    
    def has_documents(self) -> bool:
        """Check if the collection has any documents."""
        try:
            collection_info = self.get_collection_info()
            return collection_info["exists"] and collection_info["points_count"] > 0
        except Exception as e:
            logger.error(f"Error checking if collection has documents: {e}")
            return False
            
    def get_conversation_history(self, doc_id: str, limit: int = 50) -> List[dict[str, str]]:
        """Get conversation history for a document."""
        return []