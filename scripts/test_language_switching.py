#!/usr/bin/env python3
"""Test language switching for RAG responses."""

import os
import sys
from pathlib import Path

# Add the src directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_language_switching():
    """Test switching between Indonesian and English responses."""
    print("=== Testing Language Switching ===\n")
    
    test_question = "What is the operating voltage range?"
    
    # Test both languages
    languages_to_test = [
        ("id", "Indonesian"),
        ("en", "English")
    ]
    
    for lang_code, lang_name in languages_to_test:
        print(f"🔄 Testing {lang_name} responses...")
        print("-" * 40)
        
        # Set environment variable
        os.environ['response_language'] = lang_code
        
        try:
            # Clear imports to get fresh settings
            modules_to_clear = [
                'newsensor_streamlit.config',
                'newsensor_streamlit.services.rag_service'
            ]
            
            for module in modules_to_clear:
                if module in sys.modules:
                    del sys.modules[module]
            
            from newsensor_streamlit.config import settings
            from newsensor_streamlit.services.rag_service import RagService
            
            # Initialize service
            rag_service = RagService()
            
            # Test no context scenario
            result = rag_service.generate_answer(test_question, [])
            
            print(f"   ✅ Language Setting: {settings.response_language_enum.value}")
            print(f"   📝 No Context Response: {result['answer']}")
            print()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("✅ Language switching test completed!")

if __name__ == "__main__":
    test_language_switching()
