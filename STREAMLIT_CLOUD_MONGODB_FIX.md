# Streamlit Cloud MongoDB Configuration Guide

## Problem
Your Streamlit app works locally with MongoDB Atlas but fails on Streamlit Cloud with the error:
```
Exception: Failed to create conversation in MongoDB
```

## Root Cause
Streamlit Cloud requires a specific format for secrets configuration. The TOML format you used:

```toml
[mongodb]
mongodb_uri="mongodb+srv://darmakotama:feVqmFX6d3QYgdea@cluster0.b17tubo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongodb_database="newsensor_conversations"
mongodb_conversations_collection="conversations"
```

May not be loading correctly due to nested structure or environment variable conversion issues.

## Solutions

### Option 1: Flat Structure (Recommended)
Use a flat structure in your `.streamlit/secrets.toml` file on Streamlit Cloud:

```toml
# .streamlit/secrets.toml
mongodb_uri = "mongodb+srv://darmakotama:feVqmFX6d3QYgdea@cluster0.b17tubo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongodb_database = "newsensor_conversations"
mongodb_conversations_collection = "conversations"

# Other required API keys
google_api_key = "your_google_key"
openai_api_key = "your_openai_key"
# ... other keys
```

### Option 2: Environment Variables
Alternatively, set these as individual environment variables in Streamlit Cloud:
- `MONGODB_URI`
- `MONGODB_DATABASE` 
- `MONGODB_CONVERSATIONS_COLLECTION`

### Option 3: Mixed Approach
Keep the nested structure but ensure all required fields are present:

```toml
[mongodb]
mongodb_uri = "mongodb+srv://darmakotama:feVqmFX6d3QYgdea@cluster0.b17tubo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongodb_database = "newsensor_conversations"
mongodb_conversations_collection = "conversations"

# Also include flat versions as fallback
mongodb_uri = "mongodb+srv://darmakotama:feVqmFX6d3QYgdea@cluster0.b17tubo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongodb_database = "newsensor_conversations"
mongodb_conversations_collection = "conversations"
```

## Code Changes Applied

I've updated the MongoDB client (`src/newsensor_streamlit/database/mongodb_client.py`) to handle multiple configuration sources in this priority order:

1. Streamlit secrets (nested): `st.secrets.mongodb.mongodb_uri`
2. Streamlit secrets (flat): `st.secrets.mongodb_uri`
3. Environment variables (uppercase): `os.environ['MONGODB_URI']`
4. Environment variables (lowercase): `os.environ['mongodb_uri']`
5. Configuration file fallback

## Testing

### Debug Script for Local Testing
Run the debug script locally to verify everything works:
```bash
cd /path/to/project
uv run python debug_mongodb_connection.py
```

### Debug Page for Streamlit Cloud
1. Temporarily add the debug page to your app
2. Navigate to the debug page to see configuration details
3. Check which configuration source is being used
4. Verify connection status

### Manual Testing Steps
1. Deploy the updated code to Streamlit Cloud
2. Check the logs for detailed error messages (now included)
3. Try creating a conversation - should work now
4. Remove debug code once verified

## Common Issues & Solutions

### Issue: "No module named 'streamlit'" in MongoDB client
**Solution**: The enhanced code handles this gracefully with try/except blocks

### Issue: Secrets not loading at all
**Solution**: 
- Check Streamlit Cloud secrets panel configuration
- Verify TOML syntax is correct
- Try flat structure instead of nested

### Issue: Connection timeout
**Solution**:
- Verify MongoDB Atlas IP whitelist includes Streamlit Cloud IPs (or use 0.0.0.0/0)  
- Check if cluster is active and not paused
- Verify connection string format

### Issue: Authentication failed
**Solution**:
- Double-check username/password in connection string
- Verify user has read/write permissions on the database
- Check if password contains special characters that need URL encoding

## Verification Steps
1. Check that conversations are being created (look in MongoDB Atlas interface)
2. Try creating multiple conversations
3. Verify that conversation history persists
4. Test message adding functionality

## Rollback Plan
If issues persist, you can temporarily revert to file-based storage by:
1. Changing `ConversationService` to use file-based backend
2. Setting a flag in environment to disable MongoDB
3. Using the existing file system until MongoDB is fully resolved

The code is designed to be backward compatible, so this rollback should be seamless.
