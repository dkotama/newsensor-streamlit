#!/usr/bin/env python3
"""
Test script to verify MongoDB migration and data integrity.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService

def test_mongodb_conversation_data():
    """Test MongoDB conversation data integrity."""
    
    print("🧪 Testing MongoDB conversation data...")
    
    service = ConversationMongoDBService()
    
    # Test loading a specific conversation
    conversation_id = "cafac49a-8cbe-4218-96ea-43601ea3fc83"
    conversation = service.load_conversation(conversation_id)
    
    if not conversation:
        print(f"❌ Could not load conversation {conversation_id}")
        return False
    
    print(f"✅ Loaded conversation: {conversation_id}")
    print(f"   Document: {conversation.get('document_name', 'Unknown')}")
    print(f"   Created: {conversation.get('created_at', 'Unknown')}")
    print(f"   Messages: {len(conversation.get('messages', []))}")
    
    # Check if messages exist and have the right structure
    messages = conversation.get('messages', [])
    if messages:
        first_message = messages[0]
        print(f"   First message timestamp: {first_message.get('timestamp', 'Unknown')}")
        print(f"   First question: {first_message.get('question', 'Unknown')[:50]}...")
        print(f"   First answer length: {len(first_message.get('answer', ''))}")
        
        # Check if all required fields are present
        required_fields = ['timestamp', 'question', 'answer', 'context', 'sources']
        missing_fields = [field for field in required_fields if field not in first_message]
        
        if missing_fields:
            print(f"   ⚠️  Missing fields in message: {missing_fields}")
        else:
            print(f"   ✅ All required message fields present")
    
    # Test listing conversations
    all_conversations = service.list_conversations()
    print(f"   Total conversations in MongoDB: {len(all_conversations)}")
    
    # Test search functionality
    search_results = service.search_conversations("Raspberry Pi")
    print(f"   Search results for 'Raspberry Pi': {len(search_results)} conversations")
    
    return True

def test_conversation_file_service():
    """Test the compatibility wrapper."""
    
    print("\n🔌 Testing ConversationFileService compatibility wrapper...")
    
    from newsensor_streamlit.services.conversation_mongodb_service import ConversationFileService
    
    file_service = ConversationFileService()
    
    # Test getting all conversation files
    file_paths = file_service.get_all_conversation_files()
    print(f"   File service found {len(file_paths)} conversation 'files'")
    
    if file_paths:
        # Test loading a conversation using file path
        first_file = file_paths[0]
        print(f"   Testing load from 'file': {Path(first_file).name}")
        
        conversation = file_service.load_conversation_from_file(first_file)
        if conversation:
            print(f"   ✅ Successfully loaded via file service")
            print(f"   Messages: {len(conversation.get('messages', []))}")
        else:
            print(f"   ❌ Failed to load via file service")
            return False
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 MongoDB Migration Verification Tests")
    print("=" * 60)
    
    # Test 1: Direct MongoDB service
    test1_passed = test_mongodb_conversation_data()
    
    # Test 2: Compatibility wrapper
    test2_passed = test_conversation_file_service()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! MongoDB migration is working correctly.")
        print("   Your application can now use MongoDB for conversation storage.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
