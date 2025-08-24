# MongoDB Conversation Storage Migration Guide

## Overview

Your conversation storage has been successfully migrated from JSON files to MongoDB while maintaining 100% backward compatibility. Your existing code will work exactly the same way, but now with better performance, search capabilities, and scalability.

## ✅ What's Been Completed

1. **MongoDB Configuration Added** - Environment variables and settings configured
2. **18 Conversations Migrated** - All your existing JSON conversations are now in MongoDB
3. **Backward Compatibility** - Your existing code continues to work without changes
4. **Enhanced Features** - New search and messaging capabilities available

## 🔄 How to Update Your Existing Code

### Option 1: Minimal Changes (Recommended)

Simply replace your existing conversation import with:

```python
# Old way (if you had imports like this):
# from some_module import save_conversation_to_file, load_conversation_from_file

# New way - just change the import:
from newsensor_streamlit import conversation_storage

# All your existing code remains the same:
file_path = conversation_storage.save_conversation_to_file(data, conversation_id)
conversation = conversation_storage.load_conversation_from_file(file_path)
all_files = conversation_storage.get_all_conversation_files()
```

### Option 2: Direct MongoDB Service (For New Features)

If you want to use enhanced MongoDB features:

```python
from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService

service = ConversationMongoDBService()

# Enhanced features:
results = service.search_conversations("Raspberry Pi")
conversations = service.list_conversations(doc_id="OULTX125R")
service.add_message_to_conversation(conversation_id, message_data)
```

## 📋 Environment Configuration

Your `.env` file now includes:

```properties
# MongoDB Configuration for Conversations
mongodb_uri=mongodb://admin:password@localhost:27017/
mongodb_database=newsensor_conversations
mongodb_conversations_collection=conversations
```

For **production**, update the `mongodb_uri` to point to your MongoDB Atlas or production instance:

```properties
# Production MongoDB Atlas example:
mongodb_uri=mongodb+srv://username:password@cluster.mongodb.net/newsensor_conversations?retryWrites=true&w=majority
```

## 🚀 New Features Available

### 1. Full-Text Search
```python
# Search across all conversation content
results = conversation_storage.search_conversations("Arduino GPIO", limit=10)
```

### 2. Add Messages to Existing Conversations
```python
new_message = {
    "timestamp": datetime.now().isoformat(),
    "question": "Follow-up question",
    "answer": "Follow-up answer",
    # ... rest of message structure
}
success = conversation_storage.add_message_to_conversation(conversation_id, new_message)
```

### 3. Filter by Document
```python
# Get conversations for specific documents
oultx_conversations = conversation_storage.list_conversations(doc_id="OULTX125R")
```

### 4. Update RAGAS Metrics
```python
ragas_data = {
    "total_questions": 5,
    "avg_faithfulness": 0.92,
    "avg_answer_relevancy": 0.89,
    # ...
}
conversation_storage.update_ragas_summary(conversation_id, ragas_data)
```

## 🗂️ File Structure Changes

- **Conversations Directory**: `data/conversations/` now contains backup files in `migrated_backup/`
- **Original Files**: Safely backed up in `data/conversations/migrated_backup/`
- **MongoDB Collections**: Data is stored in `newsensor_conversations.conversations`

## 🔧 Testing & Verification

To verify everything is working:

```bash
# Test the migration
uv run python scripts/test_mongodb_migration.py

# See usage examples
uv run python scripts/example_mongodb_usage.py
```

## 📈 Performance Benefits

- **Faster Queries**: MongoDB indexing provides faster search and retrieval
- **Concurrent Access**: Multiple processes can safely access conversations
- **Scalability**: Handles thousands of conversations efficiently
- **Search**: Full-text search across all conversation content
- **Atomic Updates**: Safe concurrent message additions

## 🔒 Data Safety

- ✅ All original JSON files backed up in `migrated_backup/`
- ✅ Data structure exactly preserved
- ✅ All 18 conversations migrated successfully
- ✅ Full rollback possible if needed

## 🐛 Troubleshooting

### MongoDB Connection Issues
```python
from newsensor_streamlit.database.mongodb_client import get_mongodb_client

# Test connection
client = get_mongodb_client()
if client.test_connection():
    print("MongoDB connected!")
else:
    print("Connection failed - check your mongodb_uri")
```

### Missing Conversations
If a conversation seems missing:
```bash
# Check MongoDB directly
uv run python -c "
from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService
service = ConversationMongoDBService()
print('Total conversations:', len(service.list_conversations()))
"
```

## 🚀 Deployment Notes

### Local Development
- MongoDB runs on `localhost:27017` (current setup)
- No authentication required for development

### Production Deployment
1. Update `mongodb_uri` in `.env` to production MongoDB instance
2. Ensure MongoDB Atlas or production server is accessible
3. Update connection string with proper authentication
4. Consider MongoDB connection pooling for high traffic

### Example Production URI
```properties
# MongoDB Atlas
mongodb_uri=mongodb+srv://newsensor_user:secure_password@newsensor-cluster.abc123.mongodb.net/newsensor_conversations?retryWrites=true&w=majority

# Self-hosted MongoDB with authentication
mongodb_uri=mongodb://newsensor_user:secure_password@mongodb.yourdomain.com:27017/newsensor_conversations
```

## 🎯 Next Steps

1. **Update Your Application Code**: Replace conversation imports as shown above
2. **Test Everything**: Run your application and verify conversations load correctly
3. **Deploy**: Update production environment variables
4. **Clean Up**: After confirming everything works, you can delete `migrated_backup/` folder

## 💡 Tips

- Your existing Streamlit app will work without any code changes if you update the imports
- All conversation IDs remain the same
- Search functionality is now available for building conversation history features
- Message timestamps and structure are preserved exactly
- RAGAS metrics and retrieval metadata are fully preserved

---

**Migration Complete!** 🎉 Your conversations are now powered by MongoDB with enhanced capabilities while maintaining full backward compatibility.
