#!/usr/bin/env python3
"""
Test script to verify that new conversations are being saved to MongoDB.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newsensor_streamlit.services.conversation_service import ConversationService

def test_new_conversation_creation():
    """Test creating a new conversation and adding a message."""
    
    print("🧪 Testing new conversation creation with MongoDB...")
    
    # Get initial count
    service = ConversationService()
    initial_conversations = service.list_conversations()
    initial_count = len(initial_conversations)
    print(f"Initial conversation count: {initial_count}")
    
    # Create a new conversation
    print("\n📝 Creating new conversation...")
    conversation_id = service.create_conversation(
        doc_id="test_sensor_model",
        document_name="Test Sensor Integration Guide"
    )
    print(f"Created conversation: {conversation_id}")
    
    # Add a message to the conversation
    print("\n💬 Adding message...")
    service.add_message(
        conversation_id=conversation_id,
        question="How do I test this integration?",
        answer="To test the sensor integration, you should follow these steps...",
        sources=["Test Guide.pdf"],
        metrics={"faithfulness": 0.92, "answer_relevancy": 0.88},
        context="Based on the test integration guide...",
        ragas_metrics={"faithfulness": 0.92, "answer_relevancy": 0.88, "context_precision": 0.85},
        retrieval_metadata={"reranking_enabled": False, "processing_time": 2.1}
    )
    print("Message added successfully!")
    
    # Verify the conversation exists and has the message
    print("\n🔍 Verifying conversation was saved...")
    saved_conversation = service.get_conversation(conversation_id)
    
    if saved_conversation:
        print(f"✅ Conversation found in MongoDB!")
        print(f"   Document: {saved_conversation['document_name']}")
        print(f"   Messages: {len(saved_conversation['messages'])}")
        print(f"   Created: {saved_conversation['created_at']}")
        
        if saved_conversation['messages']:
            first_msg = saved_conversation['messages'][0]
            print(f"   First question: {first_msg['question']}")
            print(f"   Answer length: {len(first_msg['answer'])}")
    else:
        print("❌ Conversation not found!")
        return False
    
    # Check the updated count
    updated_conversations = service.list_conversations()
    updated_count = len(updated_conversations)
    print(f"\nFinal conversation count: {updated_count}")
    print(f"New conversations added: {updated_count - initial_count}")
    
    # Show recent conversations
    print("\n📋 Recent conversations:")
    for conv in updated_conversations[:3]:
        print(f"   - {conv['conversation_id'][:8]}... : {conv['document_name']} ({conv.get('message_count', 0)} msgs)")
    
    return True

if __name__ == "__main__":
    success = test_new_conversation_creation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: New conversations are being saved to MongoDB!")
        print("   Your application is working correctly.")
    else:
        print("❌ FAILED: There might be an issue with conversation saving.")
    print("=" * 60)
