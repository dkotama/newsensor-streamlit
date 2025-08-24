#!/usr/bin/env python3
"""Test switching between different PDF parser backends."""

import sys
import os
from pathlib import Path

# Add the src directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_backend_switching():
    """Test switching between different parser backends."""
    print("=== Testing Backend Switching ===\n")
    
    # Find a test PDF
    uploads_dir = Path("data/uploads")
    pdf_files = list(uploads_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/uploads/")
        return False
    
    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf.name}")
    print()
    
    # Test different backends
    backends_to_test = [
        ("pymupdf4llm", "PyMuPDF4LLM Parser"),
        ("pypdf2", "PyPDF2 Parser"),
        ("llamaparse", "LlamaParse Parser")
    ]
    
    results = []
    
    for backend_name, backend_desc in backends_to_test:
        print(f"🔄 Testing {backend_desc}...")
        print("-" * 40)
        
        # Set environment variable
        os.environ['PDF_PARSER_BACKEND'] = backend_name
        
        try:
            # Reimport to get fresh settings
            if 'newsensor_streamlit.config' in sys.modules:
                del sys.modules['newsensor_streamlit.config']
            if 'newsensor_streamlit.services.pdf_parser_service' in sys.modules:
                del sys.modules['newsensor_streamlit.services.pdf_parser_service']
            if 'newsensor_streamlit.services.llama_parse_service' in sys.modules:
                del sys.modules['newsensor_streamlit.services.llama_parse_service']
            
            from newsensor_streamlit.config import settings
            from newsensor_streamlit.services.llama_parse_service import LlamaParseService
            
            # Initialize service
            parser_service = LlamaParseService(api_key=settings.llama_parse_api_key)
            
            # Test processing
            result = parser_service.process_pdf(test_pdf)
            
            backend_used = result['metadata']['processing_backend']
            content_length = len(result['content'])
            
            print(f"   ✅ Success: {backend_used}")
            print(f"   📊 Content Length: {content_length} characters")
            print(f"   🔧 Sensor Model: {result['metadata'].get('sensor_model', 'unknown')}")
            print(f"   🏭 Manufacturer: {result['metadata'].get('manufacturer', 'unknown')}")
            print()
            
            results.append({
                'requested': backend_name,
                'actual': backend_used,
                'success': True,
                'content_length': content_length
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print()
            
            results.append({
                'requested': backend_name,
                'actual': 'failed',
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("=" * 50)
    print("📊 BACKEND SWITCHING TEST SUMMARY")
    print("=" * 50)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} Requested: {result['requested']} → Actual: {result['actual']}")
        if result['success']:
            print(f"   Content Length: {result['content_length']} chars")
        else:
            print(f"   Error: {result['error']}")
        print()
    
    successful_backends = [r for r in results if r['success']]
    print(f"🎯 Successfully tested {len(successful_backends)}/{len(results)} backends")
    
    return len(successful_backends) > 0

if __name__ == "__main__":
    try:
        success = test_backend_switching()
        if success:
            print("✅ Backend switching test completed!")
            sys.exit(0)
        else:
            print("❌ Backend switching test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during backend switching test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
