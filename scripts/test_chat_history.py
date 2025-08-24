#!/usr/bin/env python3
"""Test chat history context integration with configurable RAG settings."""

import sys
import os
from pathlib import Path
from langchain.schema import Document

# Add the src directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_chat_history_context():
    """Test that chat history is included in context."""
    print("=== Testing Chat History Context with Configurable RAG ===\n")
    
    try:
        from newsensor_streamlit.config import settings
        from newsensor_streamlit.services.rag_service import RagService
        
        # Display current settings
        print("📋 Current RAG Configuration:")
        print(f"   Model: {settings.rag_chat_model}")
        print(f"   Temperature: {settings.rag_temperature}")
        print(f"   Language: {settings.response_language_enum.value}")
        print()
        
        # Initialize RAG service
        rag_service = RagService()
        
        # Create mock context documents
        mock_context = [
            Document(
                page_content="OULTX125R Temperature Sensor\nOperating Voltage: 12-24V DC\nAccuracy: ±0.5°C\nRange: -40°C to 85°C\nOutput: 4-20mA",
                metadata={"source": "OULTX125R_datasheet.pdf"}
            )
        ]
        
        print("🔄 Step 1: Ask first question (no chat history)")
        print("-" * 50)
        
        # First question
        first_question = "What is the operating voltage of OULTX125R?"
        result1 = rag_service.generate_answer(first_question, mock_context)
        
        print(f"Question: {first_question}")
        print(f"Answer: {result1['answer'][:200]}...")
        print(f"Chat History Used: {result1.get('chat_history_used', False)}")
        print(f"Chat History Length: {result1.get('chat_history_length', 0)}")
        print()
        
        print("🔄 Step 2: Ask follow-up question (should use chat history)")
        print("-" * 50)
        
        # Follow-up question that refers to previous context
        followup_question = "What about the accuracy of that sensor?"
        result2 = rag_service.generate_answer(followup_question, mock_context)
        
        print(f"Question: {followup_question}")
        print(f"Answer: {result2['answer'][:200]}...")
        print(f"Chat History Used: {result2.get('chat_history_used', False)}")
        print(f"Chat History Length: {result2.get('chat_history_length', 0)}")
        print()
        
        print("🔄 Step 3: Ask contextual question (tests chat history understanding)")
        print("-" * 50)
        
        # Contextual question
        contextual_question = "Can I use it with 15V supply?"
        result3 = rag_service.generate_answer(contextual_question, mock_context)
        
        print(f"Question: {contextual_question}")
        print(f"Answer: {result3['answer'][:200]}...")
        print(f"Chat History Used: {result3.get('chat_history_used', False)}")
        print(f"Chat History Length: {result3.get('chat_history_length', 0)}")
        print()
        
        print("🔄 Step 4: Check conversation history")
        print("-" * 50)
        
        # Get conversation history
        history = rag_service.get_conversation_history()
        print(f"Total messages in history: {len(history)}")
        
        for i, message in enumerate(history):
            message_type = "Human" if i % 2 == 0 else "AI"
            content_preview = message.content[:80] + "..." if len(message.content) > 80 else message.content
            print(f"  {i+1}. {message_type}: {content_preview}")
        print()
        
        print("🔄 Step 5: Test conversation persistence")
        print("-" * 50)
        
        # Save conversation
        rag_service.save_conversation_to_file("test_chat_history")
        print("✅ Conversation saved to file")
        
        # Clear memory
        original_history_length = len(history)
        rag_service.clear_conversation_history()
        cleared_history = rag_service.get_conversation_history()
        print(f"✅ Memory cleared (was {original_history_length}, now {len(cleared_history)})")
        
        # Load conversation
        success = rag_service.load_conversation_from_file("test_chat_history")
        print(f"✅ Conversation loaded: {success}")
        
        # Check if history is restored
        restored_history = rag_service.get_conversation_history()
        print(f"✅ Restored messages: {len(restored_history)}")
        
        print("\n" + "=" * 60)
        print("📊 CHAT HISTORY WITH RAG SETTINGS TEST SUMMARY")
        print("=" * 60)
        
        test_results = {
            "RAG settings loaded": settings.rag_chat_model is not None,
            "Temperature configured": settings.rag_temperature is not None,
            "First question processed": result1 is not None,
            "Follow-up question processed": result2 is not None,
            "Contextual question processed": result3 is not None,
            "Chat history accumulated": len(history) >= 6,  # 3 Q&A pairs
            "History used in follow-up": result2.get('chat_history_used', False),
            "History used in contextual": result3.get('chat_history_used', False),
            "Conversation persistence": success,
            "History restoration": len(restored_history) == original_history_length
        }
        
        for test, passed in test_results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {test}")
        
        all_passed = all(test_results.values())
        print(f"\n🎯 Overall: {'PASS' if all_passed else 'FAIL'}")
        
        if all_passed:
            print("\n🌟 Chat history context integration is working perfectly!")
            print("   - Questions can reference previous conversations")
            print("   - Settings are configurable via environment variables")
            print("   - Conversations can be saved and restored")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error during chat history test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_settings():
    """Test with different RAG settings."""
    print("\n🔧 Testing Different RAG Settings")
    print("=" * 40)
    
    # Test with different temperature
    os.environ['RAG_TEMPERATURE'] = '0.7'
    os.environ['RAG_CHAT_MODEL'] = 'gpt-3.5-turbo'
    
    # Clear modules to reload settings
    modules_to_clear = [
        'newsensor_streamlit.config',
        'newsensor_streamlit.services.rag_service'
    ]
    
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    try:
        from newsensor_streamlit.config import settings
        print(f"📋 Updated Settings:")
        print(f"   Model: {settings.rag_chat_model}")
        print(f"   Temperature: {settings.rag_temperature}")
        return True
    except Exception as e:
        print(f"❌ Error testing different settings: {e}")
        return False

if __name__ == "__main__":
    success1 = test_chat_history_context()
    success2 = test_different_settings()
    
    overall_success = success1 and success2
    print(f"\n🏁 Final Result: {'SUCCESS' if overall_success else 'FAILURE'}")
    sys.exit(0 if overall_success else 1)
