#!/usr/bin/env python3
"""
Migration script to move existing JSON conversation files to MongoDB.
Preserves the exact same data structure for backward compatibility.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService
from newsensor_streamlit.database.mongodb_client import get_mongodb_client


def migrate_conversations_to_mongodb():
    """Migrate existing JSON conversation files to MongoDB."""
    
    print("🚀 Starting MongoDB conversation migration...")
    
    # Test MongoDB connection first
    mongodb_client = get_mongodb_client()
    if not mongodb_client.test_connection():
        print("❌ Cannot connect to MongoDB. Please check your connection and try again.")
        return False
    
    print("✅ MongoDB connection successful")
    
    # Initialize services
    mongodb_service = ConversationMongoDBService()
    conversations_dir = Path("data/conversations")
    
    if not conversations_dir.exists():
        print("ℹ️  No conversations directory found. Nothing to migrate.")
        return True
    
    # Get all JSON files (exclude exports and test files)
    json_files = [
        f for f in conversations_dir.glob("*.json") 
        if not f.name.startswith("test_") and f.name != "test_chat_history.json"
    ]
    
    if not json_files:
        print("ℹ️  No conversation files found to migrate.")
        return True
    
    print(f"📁 Found {len(json_files)} conversation files to migrate...")
    
    migrated_count = 0
    failed_count = 0
    
    for json_file in json_files:
        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Validate required fields and set defaults for missing ones
            if 'conversation_id' not in conversation_data:
                print(f"⚠️  Skipping {json_file.name}: Missing conversation_id")
                failed_count += 1
                continue
            
            # Set default values for missing fields
            if 'doc_id' not in conversation_data or conversation_data['doc_id'] is None:
                conversation_data['doc_id'] = "collection_wide"
            
            if 'document_name' not in conversation_data:
                conversation_data['document_name'] = "Collection"
            
            if 'created_at' not in conversation_data:
                conversation_data['created_at'] = datetime.now().isoformat()
            
            if 'updated_at' not in conversation_data:
                conversation_data['updated_at'] = datetime.now().isoformat()
            
            if 'messages' not in conversation_data:
                conversation_data['messages'] = []
            
            if 'ragas_summary' not in conversation_data:
                conversation_data['ragas_summary'] = {
                    "total_questions": 0,
                    "avg_faithfulness": 0.0,
                    "avg_answer_relevancy": 0.0,
                    "avg_context_precision": 0.0,
                    "avg_context_recall": 0.0
                }
            
            # Save to MongoDB
            success = mongodb_service.save_conversation(conversation_data)
            
            if success:
                migrated_count += 1
                print(f"✅ Migrated: {json_file.name}")
                
                # Create backup and move original file
                backup_dir = conversations_dir / "migrated_backup"
                backup_dir.mkdir(exist_ok=True)
                
                # Move original file to backup
                backup_file = backup_dir / json_file.name
                json_file.rename(backup_file)
                
            else:
                failed_count += 1
                print(f"❌ Failed to migrate: {json_file.name}")
                
        except json.JSONDecodeError as e:
            failed_count += 1
            print(f"❌ Invalid JSON in {json_file.name}: {e}")
        except Exception as e:
            failed_count += 1
            print(f"❌ Error migrating {json_file.name}: {e}")
    
    print(f"\n📊 Migration Summary:")
    print(f"✅ Successfully migrated: {migrated_count}")
    print(f"❌ Failed to migrate: {failed_count}")
    print(f"📁 Total processed: {migrated_count + failed_count}")
    
    if migrated_count > 0:
        print(f"📁 Original files backed up to: {conversations_dir / 'migrated_backup'}")
        print("ℹ️  You can safely delete the backup folder after verifying the migration.")
    
    return failed_count == 0


def verify_migration():
    """Verify that all conversations were migrated correctly."""
    print("\n🔍 Verifying migration...")
    
    try:
        mongodb_service = ConversationMongoDBService()
        conversations = mongodb_service.list_conversations()
        
        print(f"✅ Found {len(conversations)} conversations in MongoDB:")
        
        for conv in conversations:
            message_count = len(conv.get('messages', []))
            created_date = conv.get('created_at', 'Unknown')[:10]  # Just the date part
            print(f"  📄 {conv['conversation_id'][:8]}... - {conv['document_name']} ({message_count} messages) - {created_date}")
        
        # Show some statistics
        total_messages = sum(len(conv.get('messages', [])) for conv in conversations)
        print(f"\n📊 Total messages across all conversations: {total_messages}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False


def setup_indexes():
    """Create MongoDB indexes for better performance."""
    print("\n🔧 Setting up MongoDB indexes...")
    
    try:
        mongodb_client = get_mongodb_client()
        collection = mongodb_client.conversations
        
        # Create indexes
        collection.create_index("conversation_id", unique=True)
        collection.create_index("doc_id")
        collection.create_index("updated_at")
        collection.create_index([("doc_id", 1), ("updated_at", -1)])
        
        # Text search index for questions and answers
        collection.create_index([
            ("messages.question", "text"),
            ("messages.answer", "text"),
            ("messages.context", "text")
        ])
        
        print("✅ MongoDB indexes created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        return False


def main():
    """Main migration function."""
    print("=" * 60)
    print("📚 NewsSnsor Conversation Migration Tool")
    print("   From JSON Files → MongoDB")
    print("=" * 60)
    
    # Step 1: Migrate conversations
    migration_success = migrate_conversations_to_mongodb()
    
    if not migration_success:
        print("\n❌ Migration failed. Please check the errors above and try again.")
        sys.exit(1)
    
    # Step 2: Verify migration
    verification_success = verify_migration()
    
    if not verification_success:
        print("\n❌ Migration verification failed. Please check your MongoDB connection.")
        sys.exit(1)
    
    # Step 3: Setup indexes
    index_success = setup_indexes()
    
    if not index_success:
        print("\n⚠️  Index creation failed, but migration was successful.")
        print("   You can run the index setup manually later.")
    
    print("\n" + "=" * 60)
    print("🎉 Migration completed successfully!")
    print("   Your conversations are now stored in MongoDB.")
    print("   Your application will now use MongoDB instead of JSON files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
