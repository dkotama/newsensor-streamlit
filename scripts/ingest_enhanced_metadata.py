#!/usr/bin/env python3
"""
Script to ingest documents using the enhanced metadata system.
Replaces the existing collection with metadata-aware chunks.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.newsensor_streamlit.services.document_processor import DocumentProcessor
from src.newsensor_streamlit.services.embedding_service import EmbeddingService
from src.newsensor_streamlit.services.qdrant_service import QdrantService
from src.newsensor_streamlit.config import settings
from loguru import logger
import time


def ingest_with_enhanced_metadata():
    """Ingest documents using enhanced metadata system."""
    logger.info("Starting enhanced metadata ingestion...")
    
    # Initialize services
    processor = DocumentProcessor("data/cache")
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    
    # Find PDF files
    upload_dir = Path("data/uploads")
    pdf_files = list(upload_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {upload_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each document
    for pdf_file in pdf_files:
        logger.info(f"\nProcessing {pdf_file.name}...")
        start_time = time.time()
        
        try:
            # 1. Extract content and metadata
            document = processor.process_pdf(pdf_file)
            sensor_model = document["metadata"].get("sensor_model", "unknown")
            manufacturer = document["metadata"].get("manufacturer", "unknown")
            
            logger.info(f"Extracted metadata - Model: {sensor_model}, Manufacturer: {manufacturer}")
            
            # 2. Create metadata-aware chunks
            chunks = processor.create_metadata_aware_chunks(document, chunk_size=settings.chunk_size)
            logger.info(f"Created {len(chunks)} metadata-aware chunks")
            
            # 3. Generate enhanced embeddings
            logger.info("Generating enhanced embeddings...")
            embeddings = []
            for i, chunk in enumerate(chunks):
                if i % 10 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(chunks)} chunks")
                
                embedding = embedding_service.generate_metadata_aware_embedding(chunk)
                embeddings.append(embedding)
            
            # 4. Store in Qdrant
            logger.info("Storing enhanced chunks in Qdrant...")
            doc_id = qdrant_service.store_enhanced_chunks(chunks, embeddings, str(pdf_file))
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Successfully processed {pdf_file.name} in {processing_time:.2f}s")
            logger.info(f"   Doc ID: {doc_id}")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Model: {sensor_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n🎉 Enhanced metadata ingestion completed!")


def test_ingested_data():
    """Test the ingested data with metadata filtering."""
    logger.info("\nTesting ingested data...")
    
    qdrant_service = QdrantService()
    
    # Test queries that previously caused confusion
    test_cases = [
        {
            "query": "S15S model number specifications",
            "sensor_model": "S15S",
            "expected_model": "S15S"
        },
        {
            "query": "LS219 temperature range",
            "sensor_model": "LS219", 
            "expected_model": "LS219"
        },
        {
            "query": "humidity sensor accuracy",
            "sensor_model": None,
            "expected_model": None  # Should return multiple models
        }
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        sensor_model = test_case["sensor_model"]
        expected = test_case["expected_model"]
        
        logger.info(f"\n🔍 Testing: '{query}' (expecting model: {expected})")
        
        results = qdrant_service.search_with_metadata_filter(
            query=query,
            sensor_model=sensor_model,
            k=3
        )
        
        if results:
            top_result = results[0]
            actual_model = top_result.get("sensor_model", "unknown")
            score = top_result.get("vector_score", 0)
            
            logger.info(f"  ✅ Top result: {actual_model} (score: {score:.4f})")
            
            if expected and actual_model != expected:
                logger.warning(f"  ⚠️  Expected {expected} but got {actual_model}")
            
            # Show content preview
            content = top_result.get("content", "")[:150]
            logger.info(f"  📝 Content: {content}...")
            
        else:
            logger.warning(f"  ❌ No results found for '{query}'")


if __name__ == "__main__":
    try:
        # Ingest documents with enhanced metadata
        ingest_with_enhanced_metadata()
        
        # Test the results
        test_ingested_data()
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
