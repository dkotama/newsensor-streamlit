#!/usr/bin/env python3
"""Test script to verify the new dynamic PDF parser system."""

import sys
from pathlib import Path

# Add the src directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from newsensor_streamlit.config import settings
from newsensor_streamlit.services.llama_parse_service import LlamaParseService

def test_parser_system():
    """Test the dynamic parser system."""
    print("=== Testing Dynamic PDF Parser System ===\n")
    
    # Initialize the service
    parser_service = LlamaParseService(api_key=settings.llama_parse_api_key)
    
    # Check health status
    print("1. Health Check:")
    health_status = parser_service.get_parser_status()
    print(f"   Backend Health: {health_status['backend_health']}")
    print(f"   Available Backends: {health_status['available_backends']}")
    print(f"   Configured Backend: {health_status['configured_backend']}")
    print(f"   Fallback Chain: {health_status['fallback_chain']}")
    print()
    
    # Test if any backend is working
    print("2. Overall Health:")
    is_healthy = parser_service.health_check()
    print(f"   At least one backend available: {is_healthy}")
    print()
    
    # Show current settings
    print("3. Current Configuration:")
    print(f"   PDF_PARSER_BACKEND: {settings.pdf_parser_backend}")
    print(f"   PYMUPDF_WRITE_IMAGES: {settings.pymupdf_write_images}")
    print(f"   PYMUPDF_EMBED_IMAGES: {settings.pymupdf_embed_images}")
    print(f"   PYMUPDF_PAGE_CHUNKS: {settings.pymupdf_page_chunks}")
    print(f"   PYMUPDF_SHOW_PROGRESS: {settings.pymupdf_show_progress}")
    print(f"   PDF_PARSER_FALLBACK_CHAIN: {settings.pdf_parser_fallback_chain}")
    print()
    
    return is_healthy

if __name__ == "__main__":
    try:
        success = test_parser_system()
        if success:
            print("✅ Dynamic PDF parser system is working!")
            sys.exit(0)
        else:
            print("❌ No PDF parser backends are available!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error testing parser system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
