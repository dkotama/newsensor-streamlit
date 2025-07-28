#!/usr/bin/env python3
"""
Test script for page number metadata extraction.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.newsensor_streamlit.services.llama_parse_service import LlamaParseService
from src.newsensor_streamlit.services.document_processor import DocumentProcessor
from src.newsensor_streamlit.config import settings
from loguru import logger
import json


def test_page_metadata_extraction():
    """Test page number metadata extraction."""
    logger.info("Testing page number metadata extraction...")
    
    if not settings.llama_parse_api_key:
        logger.warning("LlamaParse API key not found, testing PyPDF2 fallback only")
        parser = LlamaParseService("")
    else:
        parser = LlamaParseService(settings.llama_parse_api_key)
    
    # Test with S15S sensor document
    test_file = Path("data/uploads/S15S-Temperature-and-Humidity-Sensor.pdf")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info(f"Processing {test_file.name} for page metadata...")
    result = parser.process_pdf(test_file)
    
    # Print page-related metadata
    metadata = result["metadata"]
    logger.info("Page-related metadata extracted:")
    
    page_info = metadata.get("page_info", {})
    logger.info(f"  Page Info: {page_info}")
    
    if page_info:
        for key, value in page_info.items():
            logger.info(f"    {key}: {value}")
    
    # Test document processor with page-aware chunks
    logger.info("\nTesting page-aware chunking...")
    processor = DocumentProcessor("data/cache")
    
    document = processor.process_pdf(test_file)
    chunks = processor.create_metadata_aware_chunks(document, chunk_size=512)
    
    logger.info(f"Created {len(chunks)} chunks with page metadata")
    
    # Show page information for first few chunks
    for i, chunk in enumerate(chunks[:3]):
        chunk_meta = chunk["metadata"]
        logger.info(f"\nChunk {i+1}:")
        logger.info(f"  Total Pages: {chunk_meta.get('total_pages', 'unknown')}")
        logger.info(f"  Estimated Page: {chunk_meta.get('estimated_page', 'unknown')}")
        logger.info(f"  Chunk Context: {chunk_meta.get('chunk_context', 'unknown')}")
        logger.info(f"  Content Preview: {chunk['content'][:100]}...")
    
    return result


def test_all_documents():
    """Test page metadata for all documents."""
    logger.info("\n" + "="*50)
    logger.info("TESTING PAGE METADATA FOR ALL DOCUMENTS")
    logger.info("="*50)
    
    processor = DocumentProcessor("data/cache")
    
    # Find all PDF files
    upload_dir = Path("data/uploads")
    pdf_files = list(upload_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        logger.info(f"\n📄 Processing {pdf_file.name}...")
        
        try:
            document = processor.process_pdf(pdf_file)
            metadata = document["metadata"]
            
            # Extract page information
            page_info = metadata.get("page_info", {})
            sensor_model = metadata.get("sensor_model", "unknown")
            
            logger.info(f"  Sensor Model: {sensor_model}")
            if page_info:
                logger.info(f"  Page Info: {page_info}")
            else:
                logger.info("  No page information extracted")
            
            # Show total pages if available
            total_pages = page_info.get("total_pages")
            if total_pages:
                logger.info(f"  📊 Total Pages: {total_pages}")
                
                # Show section pages if available
                specs_page = page_info.get("specifications_page")
                features_page = page_info.get("features_page")
                wiring_page = page_info.get("wiring_page")
                
                if specs_page:
                    logger.info(f"  📋 Specifications on page: {specs_page}")
                if features_page:
                    logger.info(f"  ⚡ Features on page: {features_page}")
                if wiring_page:
                    logger.info(f"  🔌 Wiring on page: {wiring_page}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to process {pdf_file.name}: {e}")


if __name__ == "__main__":
    try:
        # Test single document in detail
        test_page_metadata_extraction()
        
        # Test all documents
        test_all_documents()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
