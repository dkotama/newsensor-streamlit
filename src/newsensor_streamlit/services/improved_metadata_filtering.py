from typing import List, Optional, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
import logging

logger = logging.getLogger(__name__)

class ImprovedMetadataFilter:
    """Enhanced metadata filtering with intelligent fallback strategies."""
    
    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection_name = collection_name
    
    def search_with_intelligent_metadata_filter(
        self, 
        query: str, 
        sensor_model: str = None, 
        manufacturer: str = None,
        sensor_type: str = None, 
        k: int = 5, 
        collection_name: str = None,
        fallback_enabled: bool = True
    ) -> List[Document]:
        """Enhanced search with intelligent metadata filtering and fallback strategies."""
        try:
            from newsensor_streamlit.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            # Create context hint for better query embedding
            context_parts = []
            if sensor_model:
                context_parts.append(f"sensor model {sensor_model}")
            if manufacturer:
                context_parts.append(f"from {manufacturer}")
            context_hint = " ".join(context_parts) if context_parts else None
            
            query_embedding = embedding_service.generate_query_embedding(query, context_hint)
            target_collection = collection_name or self.collection_name
            
            # Strategy 1: Try strict metadata filtering first
            strict_results = self._search_with_strict_filter(
                query_embedding, target_collection, sensor_model, manufacturer, sensor_type, k
            )
            
            if len(strict_results) >= k * 0.6:  # If we get at least 60% of requested results
                logger.info(f"Strict filtering successful: {len(strict_results)} results")
                return self._convert_to_documents(strict_results)
            
            if not fallback_enabled:
                return self._convert_to_documents(strict_results)
            
            # Strategy 2: Try relaxed filtering with searchable_models
            logger.info("Strict filtering insufficient, trying relaxed approach")
            relaxed_results = self._search_with_relaxed_filter(
                query_embedding, target_collection, sensor_model, manufacturer, sensor_type, k
            )
            
            if len(relaxed_results) >= k * 0.6:
                logger.info(f"Relaxed filtering successful: {len(relaxed_results)} results")
                return self._convert_to_documents(relaxed_results)
            
            # Strategy 3: Try hybrid approach (combine filtered + unfiltered)
            logger.info("Relaxed filtering insufficient, trying hybrid approach")
            hybrid_results = self._search_with_hybrid_approach(
                query_embedding, target_collection, sensor_model, manufacturer, sensor_type, k
            )
            
            logger.info(f"Hybrid approach returned {len(hybrid_results)} results")
            return self._convert_to_documents(hybrid_results)
            
        except Exception as e:
            logger.error(f"Error in intelligent metadata search: {e}")
            # Final fallback: standard semantic search
            return self._fallback_semantic_search(query_embedding, target_collection, k)
    
    def _search_with_strict_filter(self, query_embedding, collection_name, sensor_model, manufacturer, sensor_type, k):
        """Strategy 1: Strict metadata filtering using exact matches."""
        filter_conditions = []
        
        if sensor_model and sensor_model.lower() != "unknown":
            filter_conditions.append(
                FieldCondition(key="sensor_model", match=MatchValue(value=sensor_model))
            )
        
        if manufacturer and manufacturer.lower() != "unknown":
            filter_conditions.append(
                FieldCondition(key="manufacturer", match=MatchValue(value=manufacturer))
            )
            
        if sensor_type and sensor_type.lower() != "unknown":
            filter_conditions.append(
                FieldCondition(key="sensor_type", match=MatchValue(value=sensor_type))
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
            query_filter=search_filter
        )
    
    def _search_with_relaxed_filter(self, query_embedding, collection_name, sensor_model, manufacturer, sensor_type, k):
        """Strategy 2: Relaxed filtering using searchable_models and broader matching."""
        filter_conditions = []
        should_conditions = []  # Use 'should' for OR logic
        
        # Use searchable_models for more flexible sensor model matching
        if sensor_model and sensor_model.lower() != "unknown":
            # Try both exact match and searchable_models array
            should_conditions.extend([
                FieldCondition(key="sensor_model", match=MatchValue(value=sensor_model)),
                FieldCondition(key="searchable_models", match=MatchAny(any=[sensor_model, sensor_model.upper(), sensor_model.lower()]))
            ])
        
        # Use alternate_models for additional flexibility
        if sensor_model and sensor_model.lower() != "unknown":
            should_conditions.append(
                FieldCondition(key="alternate_models", match=MatchAny(any=[sensor_model]))
            )
        
        # Keep strict filtering for other fields
        if manufacturer and manufacturer.lower() not in ["unknown", ""]:
            filter_conditions.append(
                FieldCondition(key="manufacturer", match=MatchValue(value=manufacturer))
            )
            
        if sensor_type and sensor_type.lower() not in ["unknown", ""]:
            filter_conditions.append(
                FieldCondition(key="sensor_type", match=MatchValue(value=sensor_type))
            )
        
        # Combine must and should conditions
        search_filter = None
        if filter_conditions and should_conditions:
            search_filter = Filter(must=filter_conditions, should=should_conditions)
        elif filter_conditions:
            search_filter = Filter(must=filter_conditions)
        elif should_conditions:
            search_filter = Filter(should=should_conditions)
        
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
            query_filter=search_filter
        )
    
    def _search_with_hybrid_approach(self, query_embedding, collection_name, sensor_model, manufacturer, sensor_type, k):
        """Strategy 3: Hybrid approach combining filtered and unfiltered results."""
        # Get fewer filtered results
        filtered_results = self._search_with_relaxed_filter(
            query_embedding, collection_name, sensor_model, manufacturer, sensor_type, k//2
        )
        
        # Get unfiltered results
        unfiltered_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        # Combine and deduplicate results
        combined_results = []
        seen_ids = set()
        
        # Add filtered results first (higher priority)
        for result in filtered_results:
            if result.id not in seen_ids:
                combined_results.append(result)
                seen_ids.add(result.id)
        
        # Add unfiltered results to fill remaining slots
        for result in unfiltered_results:
            if result.id not in seen_ids and len(combined_results) < k:
                combined_results.append(result)
                seen_ids.add(result.id)
        
        return combined_results
    
    def _fallback_semantic_search(self, query_embedding, collection_name, k):
        """Final fallback: pure semantic search without metadata filtering."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=k,
                with_payload=True
            )
            logger.info(f"Fallback semantic search returned {len(results)} results")
            return self._convert_to_documents(results)
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    def search_with_content_aware_filter(
        self, 
        query: str, 
        sensor_model: str = None, 
        manufacturer: str = None,
        sensor_type: str = None, 
        k: int = 5, 
        collection_name: str = None
    ) -> List[Document]:
        """Content-aware filtering that adapts based on query complexity."""
        
        # Analyze query complexity
        query_complexity = self._analyze_query_complexity(query)
        
        if query_complexity == "simple":
            # Use strict filtering for simple queries
            return self.search_with_intelligent_metadata_filter(
                query, sensor_model, manufacturer, sensor_type, k, collection_name, fallback_enabled=False
            )
        elif query_complexity == "multi_component":
            # Use hybrid approach for multi-component questions
            return self._search_for_multi_component_query(
                query, sensor_model, manufacturer, sensor_type, k, collection_name
            )
        else:
            # Use intelligent filtering with fallback for complex queries
            return self.search_with_intelligent_metadata_filter(
                query, sensor_model, manufacturer, sensor_type, k, collection_name, fallback_enabled=True
            )
    
    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query to determine appropriate filtering strategy."""
        query_lower = query.lower()
        
        # Multi-component indicators
        multi_component_keywords = [
            "probe", "main body", "unit", "sensor vs", "probe vs", 
            "different", "separate", "each", "both", "respectively"
        ]
        
        if any(keyword in query_lower for keyword in multi_component_keywords):
            return "multi_component"
        
        # Simple factual queries
        simple_keywords = [
            "what is the", "temperature range", "battery life", 
            "wireless technology", "accuracy", "operating temperature"
        ]
        
        if any(keyword in query_lower for keyword in simple_keywords):
            return "simple"
        
        return "complex"
    
    def _search_for_multi_component_query(
        self, query: str, sensor_model: str, manufacturer: str, sensor_type: str, k: int, collection_name: str
    ) -> List[Document]:
        """Special handling for multi-component queries like probe vs main body."""
        try:
            from newsensor_streamlit.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            # Create multiple query variations for different components
            query_variations = [
                query,  # Original query
                f"{query} probe specifications",
                f"{query} main unit specifications", 
                f"{query} housing specifications"
            ]
            
            all_results = []
            seen_ids = set()
            
            for query_variant in query_variations:
                query_embedding = embedding_service.generate_query_embedding(query_variant)
                
                # Use relaxed filtering
                results = self._search_with_relaxed_filter(
                    query_embedding, collection_name or self.collection_name, 
                    sensor_model, manufacturer, sensor_type, k//2
                )
                
                # Add unique results
                for result in results:
                    if result.id not in seen_ids and len(all_results) < k:
                        all_results.append(result)
                        seen_ids.add(result.id)
            
            logger.info(f"Multi-component search returned {len(all_results)} results")
            return self._convert_to_documents(all_results)
            
        except Exception as e:
            logger.error(f"Multi-component search failed: {e}")
            return []
    
    def _convert_to_documents(self, search_results):
        """Convert Qdrant search results to Document objects."""
        documents = []
        for result in search_results:
            try:
                doc = Document(
                    page_content=result.payload.get('content', ''),
                    metadata=result.payload
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to convert result to document: {e}")
                continue
        return documents