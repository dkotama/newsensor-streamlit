#!/usr/bin/env python3
"""Test PDF processing with the new dynamic parser system."""

import sys
from pathlib import Path
import json

# Add the src directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from newsensor_streamlit.config import settings
from newsensor_streamlit.services.llama_parse_service import LlamaParseService

def test_pdf_processing():
    """Test PDF processing with available files."""
    print("=== Testing PDF Processing with Dynamic Parser ===\n")
    
    # Initialize the service
    parser_service = LlamaParseService(api_key=settings.llama_parse_api_key)
    
    # Find available PDF files
    uploads_dir = Path("data/uploads")
    pdf_files = list(uploads_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/uploads/")
        return False
    
    print(f"Found {len(pdf_files)} PDF files:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"   {i}. {pdf_file.name}")
    print()
    
    # Test with the first PDF
    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf.name}")
    print("=" * 50)
    
    try:
        # Process the PDF
        result = parser_service.process_pdf(test_pdf)
        
        print("✅ PDF Processing Successful!")
        print(f"Processing Backend: {result['metadata']['processing_backend']}")
        print(f"Content Length: {len(result['content'])} characters")
        print()
        
        # Show metadata
        print("📊 Extracted Metadata:")
        metadata = result['metadata']
        important_fields = [
            'sensor_model', 'manufacturer', 'sensor_type', 
            'specifications', 'features', 'applications'
        ]
        
        for field in important_fields:
            if field in metadata:
                value = metadata[field]
                if field == 'specifications' and isinstance(value, dict):
                    print(f"   {field.replace('_', ' ').title()}:")
                    for spec_key, spec_value in value.items():
                        print(f"      {spec_key.replace('_', ' ').title()}: {spec_value}")
                elif isinstance(value, list):
                    if value:  # Only show non-empty lists
                        print(f"   {field.replace('_', ' ').title()}: {', '.join(map(str, value))}")
                else:
                    print(f"   {field.replace('_', ' ').title()}: {value}")
        print()
        
        # Show content preview
        print("📄 Content Preview (first 500 chars):")
        print("-" * 50)
        print(result['content'][:500])
        if len(result['content']) > 500:
            print("...")
        print("-" * 50)
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ PDF Processing Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_pdf_processing()
        if success:
            print("✅ PDF processing test completed successfully!")
            sys.exit(0)
        else:
            print("❌ PDF processing test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
