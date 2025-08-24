"""
Drop-in replacement for existing conversation file handling.
This module maintains the exact same API as your existing code but uses MongoDB instead.
"""

from .services.conversation_mongodb_service import ConversationFileService

# Export the service with the same interface as file-based operations
conversation_service = ConversationFileService()

# Backward compatibility functions that match your existing code
def save_conversation_to_file(conversation_data, conversation_id):
    """Save conversation to MongoDB (maintains file API compatibility)."""
    return conversation_service.save_conversation_to_file(conversation_data, conversation_id)

def load_conversation_from_file(file_path):
    """Load conversation from MongoDB (maintains file API compatibility)."""
    return conversation_service.load_conversation_from_file(file_path)

def get_all_conversation_files():
    """Get all conversation 'files' from MongoDB (maintains file API compatibility)."""
    return conversation_service.get_all_conversation_files()

def delete_conversation_file(file_path):
    """Delete conversation from MongoDB (maintains file API compatibility)."""
    return conversation_service.delete_conversation_file(file_path)

def conversation_exists(conversation_id):
    """Check if conversation exists in MongoDB."""
    return conversation_service.conversation_exists(conversation_id)

# Additional MongoDB-specific functions for enhanced functionality
def search_conversations(query, doc_id=None, limit=10):
    """Search conversations by content."""
    return conversation_service.mongodb_service.search_conversations(query, doc_id, limit)

def list_conversations(doc_id=None):
    """List conversations with optional filtering."""
    return conversation_service.mongodb_service.list_conversations(doc_id)

def add_message_to_conversation(conversation_id, message_data):
    """Add a new message to existing conversation."""
    return conversation_service.mongodb_service.add_message_to_conversation(conversation_id, message_data)

def update_ragas_summary(conversation_id, ragas_data):
    """Update RAGAS summary for conversation."""
    return conversation_service.mongodb_service.update_ragas_summary(conversation_id, ragas_data)
