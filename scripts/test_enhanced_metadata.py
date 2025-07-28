#!/usr/bin/env python3
"""
Test script for enhanced metadata system.
Tests the new metadata extraction and filtering capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.newsensor_streamlit.services.llama_parse_service import LlamaParseService
from src.newsensor_streamlit.services.document_processor import DocumentProcessor
from src.newsensor_streamlit.services.embedding_service import EmbeddingService
from src.newsensor_streamlit.services.qdrant_service import QdrantService
from src.newsensor_streamlit.config import settings
from loguru import logger
import json


def test_llamaparse_metadata_extraction():
    """Test LlamaParse enhanced metadata extraction."""
    logger.info("Testing LlamaParse metadata extraction...")
    
    if not settings.llama_parse_api_key:
        logger.warning("LlamaParse API key not found, skipping metadata extraction test")
        return
    
    parser = LlamaParseService(settings.llama_parse_api_key)
    
    # Test with S15S sensor document
    test_file = Path("data/uploads/S15S-Temperature-and-Humidity-Sensor.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"Processing {test_file.name}...")
    result = parser.process_pdf(test_file)
    
    # Print extracted metadata
    metadata = result["metadata"]
    logger.info("Extracted metadata:")
    for key, value in metadata.items():
        if key not in ["content"]:
            logger.info(f"  {key}: {value}")
    
    return result


def test_metadata_aware_chunking():
    """Test metadata-aware chunking."""
    logger.info("\nTesting metadata-aware chunking...")
    
    processor = DocumentProcessor("data/cache")
    
    # Process document
    test_file = Path("data/uploads/S15S-Temperature-and-Humidity-Sensor.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    document = processor.process_pdf(test_file)
    chunks = processor.create_metadata_aware_chunks(document, chunk_size=512)
    
    logger.info(f"Created {len(chunks)} metadata-aware chunks")
    
    # Show first chunk metadata
    if chunks:
        first_chunk = chunks[0]
        logger.info("First chunk metadata:")
        for key, value in first_chunk["metadata"].items():
            if isinstance(value, (str, int, float)):
                logger.info(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) < 5:
                logger.info(f"  {key}: {value}")
    
    return chunks


def test_enhanced_embeddings():
    """Test metadata-aware embeddings."""
    logger.info("\nTesting enhanced embeddings...")
    
    processor = DocumentProcessor("data/cache")
    embedding_service = EmbeddingService()
    
    # Create test chunk
    test_chunk = {
        "content": "The S15S sensor operates at 3.3V and provides I2C communication.",
        "metadata": {
            "sensor_model": "S15S",
            "manufacturer": "Sensirion",
            "sensor_type": "temperature_humidity",
            "chunk_context": "specifications"
        }
    }
    
    # Generate regular embedding
    regular_embedding = embedding_service.generate_embedding(test_chunk["content"])
    
    # Generate metadata-aware embedding
    enhanced_embedding = embedding_service.generate_metadata_aware_embedding(test_chunk)
    
    logger.info(f"Regular embedding dimension: {len(regular_embedding)}")
    logger.info(f"Enhanced embedding dimension: {len(enhanced_embedding)}")
    
    # Show difference in first few dimensions
    logger.info("First 5 dimensions comparison:")
    for i in range(5):
        logger.info(f"  Dim {i}: Regular={regular_embedding[i]:.4f}, Enhanced={enhanced_embedding[i]:.4f}")
    
    return regular_embedding, enhanced_embedding


def test_metadata_filtering():
    """Test metadata-based filtering in Qdrant."""
    logger.info("\nTesting metadata filtering...")
    
    qdrant_service = QdrantService()
    
    # Test metadata-filtered search
    test_queries = [
        ("S15S temperature range", "S15S", "Sensirion"),
        ("LS219 specifications", "LS219", None),
        ("humidity sensor accuracy", None, "Sensirion")
    ]
    
    for query, sensor_model, manufacturer in test_queries:
        logger.info(f"\nTesting query: '{query}' (model: {sensor_model}, manufacturer: {manufacturer})")
        
        results = qdrant_service.search_with_metadata_filter(
            query=query,
            sensor_model=sensor_model,
            manufacturer=manufacturer,
            k=3
        )
        
        logger.info(f"Found {len(results)} filtered results")
        for i, result in enumerate(results[:2]):  # Show first 2 results
            logger.info(f"  Result {i+1}:")
            logger.info(f"    Model: {result.get('sensor_model', 'unknown')}")
            logger.info(f"    Manufacturer: {result.get('manufacturer', 'unknown')}")
            logger.info(f"    Context: {result.get('chunk_context', 'unknown')}")
            logger.info(f"    Score: {result.get('vector_score', 0):.4f}")


def test_complete_pipeline():
    """Test the complete enhanced pipeline."""
    logger.info("\n" + "="*50)
    logger.info("TESTING COMPLETE ENHANCED METADATA PIPELINE")
    logger.info("="*50)
    
    # 1. Extract metadata
    logger.info("\n1. Testing metadata extraction...")
    document = test_llamaparse_metadata_extraction()
    if not document:
        logger.error("Metadata extraction failed, skipping pipeline test")
        return
    
    # 2. Create metadata-aware chunks
    logger.info("\n2. Testing metadata-aware chunking...")
    processor = DocumentProcessor("data/cache")
    chunks = processor.create_metadata_aware_chunks(document, chunk_size=512)
    
    # 3. Generate enhanced embeddings
    logger.info("\n3. Testing enhanced embeddings...")
    embedding_service = EmbeddingService()
    embeddings = []
    for chunk in chunks[:3]:  # Test first 3 chunks
        embedding = embedding_service.generate_metadata_aware_embedding(chunk)
        embeddings.append(embedding)
    
    logger.info(f"Generated {len(embeddings)} enhanced embeddings")
    
    # 4. Store in Qdrant with enhanced metadata
    logger.info("\n4. Testing enhanced storage...")
    qdrant_service = QdrantService()
    doc_id = qdrant_service.store_enhanced_chunks(
        chunks[:3], embeddings, document["source"]
    )
    logger.info(f"Stored chunks with doc_id: {doc_id}")
    
    # 5. Test metadata filtering
    logger.info("\n5. Testing metadata filtering...")
    sensor_model = document["metadata"].get("sensor_model", "unknown")
    manufacturer = document["metadata"].get("manufacturer", "unknown")
    
    results = qdrant_service.search_with_metadata_filter(
        query="temperature specifications",
        sensor_model=sensor_model,
        manufacturer=manufacturer,
        k=2
    )
    
    logger.info(f"Filtered search returned {len(results)} results")
    if results:
        result = results[0]
        logger.info(f"Best match: {result.get('sensor_model')} from {result.get('manufacturer')}")
        logger.info(f"Content preview: {result.get('content', '')[:100]}...")
    
    logger.info("\n✅ Enhanced metadata pipeline test completed!")


if __name__ == "__main__":
    # Run all tests
    try:
        test_complete_pipeline()
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
