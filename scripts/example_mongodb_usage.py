#!/usr/bin/env python3
"""
Example usage of the MongoDB conversation storage system.
This shows how to use the new MongoDB backend while maintaining compatibility.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newsensor_streamlit import conversation_storage

def example_save_new_conversation():
    """Example: Save a new conversation."""
    print("📝 Example: Saving a new conversation...")
    
    # Create a new conversation in the same format as your existing JSON files
    conversation_data = {
        "conversation_id": "example-conversation-123",
        "doc_id": "OULTX125R_datasheet", 
        "document_name": "OULTX125R - Example Datasheet",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": [
            {
                "timestamp": datetime.now().isoformat(),
                "question": "How do I connect this sensor to Arduino?",
                "answer": "To connect the OULTX125R sensor to Arduino, you need to use the GPIO pins...",
                "context": "Based on the datasheet specifications...",
                "sources": ["OULTX125R - Datasheet.pdf"],
                "ragas_metrics": {},
                "legacy_metrics": {
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.92,
                    "context_precision": 0.88,
                    "context_recall": 0.90
                },
                "retrieval_metadata": {
                    "reranking_enabled": True,
                    "reranking_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                    "initial_vector_search_count": 10,
                    "final_chunk_count": 5,
                    "relevance_scores": [0.85, 0.78, 0.72, 0.68, 0.61],
                    "best_relevance_score": 0.85,
                    "metadata_filtering_used": True,
                    "detected_models": ["OULTX125R"],
                    "processing_time": 2.34,
                    "metadata_filters": {
                        "sensor_model": "OULTX125R",
                        "manufacturer": "NewsSnsor",
                        "doc_id": "OULTX125R_datasheet"
                    }
                }
            }
        ],
        "ragas_summary": {
            "total_questions": 1,
            "avg_faithfulness": 0.95,
            "avg_answer_relevancy": 0.92,
            "avg_context_precision": 0.88,
            "avg_context_recall": 0.90
        }
    }
    
    # Save using the same API as before - but now it goes to MongoDB!
    try:
        file_path = conversation_storage.save_conversation_to_file(
            conversation_data, 
            "example-conversation-123"
        )
        print(f"✅ Saved conversation to: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
        return False

def example_load_conversation():
    """Example: Load an existing conversation."""
    print("\n📖 Example: Loading a conversation...")
    
    # Get all conversations (same API as before)
    all_conversations = conversation_storage.get_all_conversation_files()
    print(f"Found {len(all_conversations)} conversations")
    
    if all_conversations:
        # Load the first conversation
        first_conversation_path = all_conversations[0]
        conversation = conversation_storage.load_conversation_from_file(first_conversation_path)
        
        if conversation:
            print(f"✅ Loaded: {conversation['document_name']}")
            print(f"   Messages: {len(conversation['messages'])}")
            print(f"   Created: {conversation['created_at'][:10]}")  # Just the date
            
            if conversation['messages']:
                first_msg = conversation['messages'][0]
                print(f"   First question: {first_msg['question'][:50]}...")
            
            return True
        else:
            print("❌ Failed to load conversation")
            return False
    else:
        print("ℹ️  No conversations found")
        return True

def example_search_conversations():
    """Example: Search conversations (new MongoDB-specific feature)."""
    print("\n🔍 Example: Searching conversations...")
    
    # Search for conversations containing specific terms
    search_terms = ["Raspberry Pi", "Arduino", "GPIO", "sensor"]
    
    for term in search_terms:
        results = conversation_storage.search_conversations(term, limit=3)
        print(f"   '{term}': {len(results)} results")
        
        for result in results[:2]:  # Show first 2 results
            print(f"     - {result['document_name']} ({result['conversation_id'][:8]}...)")

def example_add_message_to_existing():
    """Example: Add message to existing conversation (new feature)."""
    print("\n➕ Example: Adding message to existing conversation...")
    
    # Check if our example conversation exists
    if conversation_storage.conversation_exists("example-conversation-123"):
        new_message = {
            "timestamp": datetime.now().isoformat(),
            "question": "What's the power consumption of this sensor?",
            "answer": "The OULTX125R sensor has low power consumption, typically drawing less than 100mA...",
            "context": "According to the specifications section...",
            "sources": ["OULTX125R - Datasheet.pdf"],
            "ragas_metrics": {},
            "legacy_metrics": {
                "faithfulness": 0.93,
                "answer_relevancy": 0.89,
                "context_precision": 0.85,
                "context_recall": 0.91
            },
            "retrieval_metadata": {
                "reranking_enabled": True,
                "processing_time": 1.87
            }
        }
        
        success = conversation_storage.add_message_to_conversation(
            "example-conversation-123", 
            new_message
        )
        
        if success:
            print("✅ Added new message to conversation")
            
            # Load and verify
            file_path = "data/conversations/example-conversation-123.json"  # Virtual path
            updated_conversation = conversation_storage.load_conversation_from_file(file_path)
            print(f"   Updated conversation now has {len(updated_conversation['messages'])} messages")
        else:
            print("❌ Failed to add message")
    else:
        print("ℹ️  Example conversation not found, skipping message addition")

def main():
    """Run all examples."""
    print("=" * 60)
    print("📚 MongoDB Conversation Storage Examples")
    print("   Same API as file-based storage, but powered by MongoDB!")
    print("=" * 60)
    
    # Example 1: Save new conversation
    example_save_new_conversation()
    
    # Example 2: Load existing conversation
    example_load_conversation()
    
    # Example 3: Search conversations (new feature)
    example_search_conversations()
    
    # Example 4: Add message to existing conversation (new feature)
    example_add_message_to_existing()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("   Your existing code will work exactly the same way,")
    print("   but now it's using MongoDB for better performance and features.")
    print("=" * 60)

if __name__ == "__main__":
    main()
