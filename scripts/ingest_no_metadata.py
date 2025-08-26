#!/usr/bin/env python3
"""
Script to ingest documents without metadata processing.
Saves to collection "sensors_no_metadata" with simple chunking.
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


def ingest_without_metadata():
    """Ingest documents using simple chunking without metadata."""
    logger.info("Starting no-metadata ingestion...")
    
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
            # 1. Extract content only (no metadata processing)
            document = processor.process_pdf(pdf_file)
            content = document["content"]
            
            logger.info(f"Extracted {len(content)} characters of content")
            
            # 2. Create simple chunks using the standard chunking method
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.schema import Document
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            # Create chunks with basic source metadata only
            chunks = text_splitter.create_documents([content])
            
            # Add minimal metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata = {
                    "source": str(pdf_file),
                    "chunk_index": i,
                    "filename": pdf_file.name
                }
            
            logger.info(f"Created {len(chunks)} simple chunks")
            
            # 3. Generate standard embeddings
            logger.info("Generating standard embeddings...")
            embeddings = embedding_service.generate_embeddings(
                [chunk.page_content for chunk in chunks],
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            # 4. Store in Qdrant with collection name "sensors_no_metadata"
            logger.info("Storing chunks in 'sensors_no_metadata' collection...")
            doc_id = qdrant_service.store_document(
                chunks=chunks,
                embeddings=embeddings,
                source_path=str(pdf_file),
                collection_name="sensors_no_metadata"  # Explicit collection name
            )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Successfully processed {pdf_file.name} in {processing_time:.2f}s")
            logger.info(f"   Doc ID: {doc_id}")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Collection: sensors_no_metadata")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n🎉 No-metadata ingestion completed!")


def test_no_metadata_retrieval():
    """Test the ingested data with standard retrieval."""
    logger.info("\nTesting no-metadata retrieval...")
    
    qdrant_service = QdrantService()
    
    # Test queries with standard search
    test_queries = [
        "S15S model specifications",
        "LS219 temperature range", 
        "humidity sensor accuracy",
        "I2C interface",
        "operating voltage range"
    ]
    
    for query in test_queries:
        logger.info(f"\n🔍 Testing: '{query}'")
        
        # Use standard search on the no-metadata collection
        try:
            results = qdrant_service.search_similar(
                query=query,
                doc_id="",  # Search across all documents
                k=3,
                collection_name="sensors_no_metadata"
            )
            
            if results:
                for i, result in enumerate(results[:2]):  # Show top 2
                    score = getattr(result, 'metadata', {}).get('score', 'N/A')
                    filename = getattr(result, 'metadata', {}).get('filename', 'Unknown')
                    content_preview = result.page_content[:100].replace('\n', ' ')
                    
                    logger.info(f"  {i+1}. {filename} - {content_preview}...")
                    
            else:
                logger.warning(f"  ❌ No results found for '{query}'")
                
        except Exception as e:
            logger.error(f"  ❌ Search failed: {e}")


if __name__ == "__main__":
    try:
        # Ingest documents without metadata
        ingest_without_metadata()
        
        # Test the results
        test_no_metadata_retrieval()
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
