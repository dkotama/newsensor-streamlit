#!/usr/bin/env python3
"""Test script for RAG provider switching functionality."""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from newsensor_streamlit.config import settings
from newsensor_streamlit.services.rag_service import RagService
from newsensor_streamlit.services.ragas_evaluator import RagasEvaluator
from langchain.schema import Document


def test_rag_provider_switching():
    """Test switching between OpenAI and OpenRouter providers."""
    print("=" * 60)
    print("RAG PROVIDER SWITCHING TEST")
    print("=" * 60)
    
    # Test data
    test_question = "What is the operating voltage range?"
    test_documents = [
        Document(
            page_content="The sensor operates at 3.3V to 5V DC with typical power consumption of 15mA.",
            metadata={"source": "test_datasheet.pdf", "page": 1}
        ),
        Document(
            page_content="Operating temperature range: -20°C to +70°C. Supply voltage: 3.3V ± 0.3V.",
            metadata={"source": "test_datasheet.pdf", "page": 2}
        )
    ]
    
    # Store original values
    original_provider = os.environ.get("rag_provider", "openai")
    
    # Test both providers
    providers_to_test = ["openai", "openrouter"]
    
    for provider in providers_to_test:
        print(f"\n{'='*20} Testing Provider: {provider.upper()} {'='*20}")
        
        # Set provider in environment
        os.environ["rag_provider"] = provider
        
        # Reload settings by creating a new instance
        from newsensor_streamlit.config import Settings
        settings = Settings()
        
        print(f"✓ Provider set to: {settings.rag_provider}")
        print(f"✓ Provider enum: {settings.rag_provider_enum.value}")
        print(f"✓ RAG model: {settings.rag_chat_model}")
        print(f"✓ Temperature: {settings.rag_temperature}")
        
        # Test RAG Service
        try:
            print(f"\n--- Testing RAG Service with {provider} ---")
            rag_service = RagService()
            
            # Check LLM configuration
            llm_config = {
                "model": rag_service.llm.model_name,
                "temperature": rag_service.llm.temperature,
            }
            
            if hasattr(rag_service.llm, 'base_url') and rag_service.llm.base_url:
                llm_config["base_url"] = rag_service.llm.base_url
            
            print(f"✓ LLM Config: {llm_config}")
            
            # Generate a test answer
            result = rag_service.generate_answer(test_question, test_documents)
            print(f"✓ Answer generated successfully (length: {len(result['answer'])} chars)")
            print(f"✓ Context used: {len(result['context'])} chars")
            print(f"✓ Sources: {result['sources']}")
            
            # Preview answer
            answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            print(f"✓ Answer preview: {answer_preview}")
            
        except Exception as e:
            print(f"✗ RAG Service failed with {provider}: {e}")
        
        # Test RAGAS Evaluator
        try:
            print(f"\n--- Testing RAGAS Evaluator with {provider} ---")
            evaluator = RagasEvaluator()
            
            print(f"✓ RAGAS available: {evaluator.is_available()}")
            
            if evaluator.is_available():
                # Test evaluation (this might take a while)
                print("  Running RAGAS evaluation (this may take 30-60 seconds)...")
                
                contexts = [doc.page_content for doc in test_documents]
                metrics = evaluator.evaluate_answer(
                    question=test_question,
                    answer=result['answer'] if 'result' in locals() else "Test answer for evaluation",
                    contexts=contexts
                )
                
                if metrics:
                    print(f"✓ RAGAS metrics: {metrics}")
                else:
                    print("⚠ RAGAS evaluation returned empty metrics")
            
        except Exception as e:
            print(f"✗ RAGAS Evaluator failed with {provider}: {e}")
        
        print("-" * 60)
    
    # Restore original provider
    if original_provider:
        os.environ["rag_provider"] = original_provider
    elif "rag_provider" in os.environ:
        del os.environ["rag_provider"]
    
    print(f"\n✓ Restored original provider: {original_provider}")
    print("\n" + "=" * 60)
    print("RAG PROVIDER SWITCHING TEST COMPLETED")
    print("=" * 60)


def test_provider_configurations():
    """Test provider configuration validation."""
    print("\n" + "=" * 60)
    print("PROVIDER CONFIGURATION VALIDATION")
    print("=" * 60)
    
    # Check API keys availability
    openai_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY") or settings.openrouter_api_key
    
    print(f"OpenAI API Key configured: {'Yes' if openai_key else 'No'}")
    print(f"OpenRouter API Key configured: {'Yes' if openrouter_key else 'No'}")
    
    if openai_key:
        print(f"OpenAI Key preview: {openai_key[:8]}...{openai_key[-4:]}")
    
    if openrouter_key:
        print(f"OpenRouter Key preview: {openrouter_key[:8]}...{openrouter_key[-4:]}")
    
    # Test both provider configurations
    for provider in ["openai", "openrouter"]:
        print(f"\n--- Testing {provider.upper()} Configuration ---")
        
        os.environ["rag_provider"] = provider
        from newsensor_streamlit.config import Settings
        test_settings = Settings()
        
        try:
            # Test RAG service initialization with new settings instance
            from newsensor_streamlit.services.rag_service import RagService
            rag_service = RagService()
            print(f"✓ {provider} RAG service initialized successfully")
            
            # Test RAGAS evaluator initialization
            from newsensor_streamlit.services.ragas_evaluator import RagasEvaluator
            evaluator = RagasEvaluator()
            print(f"✓ {provider} RAGAS evaluator initialized successfully")
            print(f"✓ {provider} RAGAS available: {evaluator.is_available()}")
            
        except Exception as e:
            print(f"✗ {provider} configuration failed: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("Starting RAG provider switching tests...")
    
    try:
        # Test provider configurations first
        test_provider_configurations()
        
        # Test actual provider switching
        test_rag_provider_switching()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
