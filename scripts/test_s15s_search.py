#!/usr/bin/env python3
"""
Test script to verify the S15S search issue is resolved.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.newsensor_streamlit.services.document_processor import DocumentProcessor
from src.newsensor_streamlit.services.embedding_service import EmbeddingService
from src.newsensor_streamlit.services.qdrant_service import QdrantService
from loguru import logger


def test_s15s_search():
    """Test that S15S search now works correctly."""
    
    # Process the S15S document to see searchable models
    processor = DocumentProcessor("data/cache")
    test_file = Path("data/uploads/S15S-Temperature-and-Humidity-Sensor.pdf")
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    document = processor.process_pdf(test_file)
    chunks = processor.create_metadata_aware_chunks(document, chunk_size=512)
    
    if chunks:
        first_chunk = chunks[0]
        searchable_models = first_chunk["metadata"].get("searchable_models", [])
        logger.info(f"Searchable models created: {searchable_models}")
        
        # Check if 'S15S' is in the searchable models
        if "S15S" in searchable_models:
            logger.info("✅ 'S15S' found in searchable models - search should work!")
        else:
            logger.warning("❌ 'S15S' not found in searchable models")
    
    # Store the enhanced chunk
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    
    # Generate one embedding to test
    embedding = embedding_service.generate_metadata_aware_embedding(chunks[0])
    qdrant_service.store_enhanced_chunks([chunks[0]], [embedding], str(test_file))
    
    # Now test the search
    logger.info("\nTesting searches:")
    
    test_searches = [
        ("S15S", "S15S"),          # Should work now
        ("S15S-TH-MQ", "S15S-TH-MQ"),  # Should work
        ("s15s", "S15S"),          # Lower case
        ("S15", "S15S"),           # Partial match
    ]
    
    for search_term, expected_model in test_searches:
        logger.info(f"\nSearching for '{search_term}' expecting model '{expected_model}':")
        
        results = qdrant_service.search_with_metadata_filter(
            query="specifications",
            sensor_model=search_term,
            k=2
        )
        
        if results:
            actual_model = results[0].get("sensor_model", "unknown")
            score = results[0].get("vector_score", 0)
            logger.info(f"  ✅ Found: {actual_model} (score: {score:.4f})")
            
            if expected_model.lower() in actual_model.lower():
                logger.info(f"  ✅ Match confirmed!")
            else:
                logger.warning(f"  ⚠️  Expected {expected_model} but got {actual_model}")
        else:
            logger.warning(f"  ❌ No results found for '{search_term}'")


if __name__ == "__main__":
    test_s15s_search()
