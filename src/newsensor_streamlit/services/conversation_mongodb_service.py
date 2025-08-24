import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..database.mongodb_client import get_mongodb_client
from ..models.conversation_models import ConversationDocument, ConversationMessage, RagasSummary


class ConversationMongoDBService:
    """MongoDB service for conversation management."""
    
    def __init__(self):
        try:
            self.mongodb = get_mongodb_client()
            self.collection = self.mongodb.conversations
            
            # Test connection on initialization
            connection_ok = self.mongodb.test_connection()
            if not connection_ok:
                print("WARNING: MongoDB connection test failed during initialization")
                print(f"URI: {str(self.mongodb.uri)[:30]}...")
                print(f"Database: {self.mongodb.db_name}")
                print(f"Collection: {self.mongodb.conversations_collection_name}")
        except Exception as e:
            print(f"ERROR: Failed to initialize MongoDB service: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise
        
    def save_conversation(self, conversation_data: Dict[str, Any]) -> bool:
        """Save conversation in the exact same format as JSON files."""
        try:
            # Convert to ConversationDocument for validation
            conversation = ConversationDocument(**conversation_data)
            
            # Use upsert to replace existing conversation
            result = self.collection.replace_one(
                {"conversation_id": conversation.conversation_id},
                conversation.dict(),
                upsert=True
            )
            
            return result.acknowledged
        except Exception as e:
            print(f"Error saving conversation: {e}")
            print(f"Error type: {type(e).__name__}")
            # Log more details for debugging
            print(f"MongoDB URI (masked): {str(self.mongodb.uri)[:30]}...")
            print(f"Database: {self.mongodb.db_name}")
            print(f"Collection: {self.mongodb.conversations_collection_name}")
            print(f"Conversation ID: {conversation_data.get('conversation_id', 'unknown')}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation in the exact same format as JSON files."""
        try:
            doc = self.collection.find_one(
                {"conversation_id": conversation_id},
                {"_id": 0}  # Exclude MongoDB ObjectId
            )
            return doc
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None
    
    def list_conversations(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List conversations, optionally filtered by doc_id."""
        try:
            query = {}
            if doc_id:
                query["doc_id"] = doc_id
            
            cursor = self.collection.find(
                query,
                {
                    "_id": 0,
                    "conversation_id": 1,
                    "doc_id": 1,
                    "document_name": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "ragas_summary": 1
                }
            ).sort("updated_at", -1)
            
            return list(cursor)
        except Exception as e:
            print(f"Error listing conversations: {e}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation."""
        try:
            result = self.collection.delete_one({"conversation_id": conversation_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False
    
    def add_message_to_conversation(self, conversation_id: str, message_data: Dict[str, Any]) -> bool:
        """Add a new message to existing conversation."""
        try:
            # Create message object for validation
            message = ConversationMessage(**message_data)
            
            # Update conversation with new message and updated timestamp
            result = self.collection.update_one(
                {"conversation_id": conversation_id},
                {
                    "$push": {"messages": message.dict()},
                    "$set": {"updated_at": datetime.now().isoformat()}
                }
            )
            
            return result.modified_count > 0
        except Exception as e:
            print(f"Error adding message: {e}")
            return False
    
    def update_ragas_summary(self, conversation_id: str, ragas_data: Dict[str, Any]) -> bool:
        """Update RAGAS summary for conversation."""
        try:
            ragas_summary = RagasSummary(**ragas_data)
            
            result = self.collection.update_one(
                {"conversation_id": conversation_id},
                {"$set": {"ragas_summary": ragas_summary.dict()}}
            )
            
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating RAGAS summary: {e}")
            return False
    
    def search_conversations(self, query: str, doc_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations by content using text search."""
        try:
            search_query = {"$text": {"$search": query}}
            if doc_id:
                search_query["doc_id"] = doc_id
            
            cursor = self.collection.find(
                search_query,
                {
                    "_id": 0,
                    "conversation_id": 1,
                    "doc_id": 1,
                    "document_name": 1,
                    "updated_at": 1,
                    "messages": 1,
                    "score": {"$meta": "textScore"}
                }
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            return list(cursor)
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []


class ConversationFileService:
    """Wrapper that maintains the same interface as your existing file-based functions."""
    
    def __init__(self):
        self.mongodb_service = ConversationMongoDBService()
        # Keep the file path structure for backward compatibility
        self.conversations_dir = Path("data/conversations")
    
    def save_conversation_to_file(self, conversation_data: Dict[str, Any], conversation_id: str) -> str:
        """Mimics saving to file but actually saves to MongoDB."""
        # Save to MongoDB instead of file
        success = self.mongodb_service.save_conversation(conversation_data)
        
        if success:
            # Return the "file path" format your existing code expects
            return str(self.conversations_dir / f"{conversation_id}.json")
        else:
            raise Exception("Failed to save conversation to MongoDB")
    
    def load_conversation_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Mimics loading from file but actually loads from MongoDB."""
        # Extract conversation_id from file path
        conversation_id = Path(file_path).stem
        
        return self.mongodb_service.load_conversation(conversation_id)
    
    def get_all_conversation_files(self) -> List[str]:
        """Mimics getting file list but actually gets from MongoDB."""
        conversations = self.mongodb_service.list_conversations()
        
        # Return file path format your existing code expects
        file_paths = []
        for conv in conversations:
            file_path = str(self.conversations_dir / f"{conv['conversation_id']}.json")
            file_paths.append(file_path)
        
        return file_paths
    
    def delete_conversation_file(self, file_path: str) -> bool:
        """Mimics deleting file but actually deletes from MongoDB."""
        conversation_id = Path(file_path).stem
        return self.mongodb_service.delete_conversation(conversation_id)
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if conversation exists."""
        conversation = self.mongodb_service.load_conversation(conversation_id)
        return conversation is not None
