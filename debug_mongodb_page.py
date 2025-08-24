"""
MongoDB Configuration Verification Page
Add this to your Streamlit app temporarily to debug configuration issues.
"""

import streamlit as st
import os
from pathlib import Path
import sys

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def show_mongodb_debug():
    """Show MongoDB configuration debug information."""
    st.title("🔍 MongoDB Configuration Debug")
    
    st.header("1. Streamlit Secrets")
    try:
        if hasattr(st, 'secrets'):
            st.success("✓ st.secrets is available")
            
            # Check for mongodb section
            if hasattr(st.secrets, 'mongodb'):
                st.success("✓ st.secrets.mongodb section found")
                mongodb_secrets = st.secrets.mongodb
                
                for key in ['mongodb_uri', 'mongodb_database', 'mongodb_conversations_collection']:
                    if hasattr(mongodb_secrets, key):
                        value = getattr(mongodb_secrets, key)
                        if 'uri' in key and '@' in str(value):
                            # Mask sensitive info
                            masked = str(value).split('@')[0] + '@***' + str(value).split('@')[1][-20:]
                            st.write(f"✓ {key}: {masked}")
                        else:
                            st.write(f"✓ {key}: {value}")
                    else:
                        st.error(f"✗ {key} not found in secrets.mongodb")
            else:
                st.warning("⚠️ st.secrets.mongodb section not found")
                
                # Check for flat structure
                flat_keys = ['mongodb_uri', 'mongodb_database', 'mongodb_conversations_collection']
                found_any = False
                for key in flat_keys:
                    if key in st.secrets:
                        found_any = True
                        value = st.secrets[key]
                        if 'uri' in key and '@' in str(value):
                            masked = str(value).split('@')[0] + '@***' + str(value).split('@')[1][-20:]
                            st.write(f"✓ {key} (flat): {masked}")
                        else:
                            st.write(f"✓ {key} (flat): {value}")
                
                if not found_any:
                    st.error("✗ No MongoDB keys found in secrets at all")
                    st.write("Available secret keys:", list(st.secrets.keys()))
        else:
            st.error("✗ st.secrets is not available")
    except Exception as e:
        st.error(f"Error checking secrets: {e}")
    
    st.header("2. Environment Variables")
    mongo_env_vars = [k for k in os.environ if 'mongo' in k.lower()]
    if mongo_env_vars:
        st.success(f"✓ Found {len(mongo_env_vars)} MongoDB environment variables")
        for var in sorted(mongo_env_vars):
            value = os.environ[var]
            if 'uri' in var.lower() and '@' in value:
                masked = value.split('@')[0] + '@***' + value.split('@')[1][-20:]
                st.write(f"✓ {var}: {masked}")
            else:
                st.write(f"✓ {var}: {value}")
    else:
        st.warning("⚠️ No MongoDB environment variables found")
    
    st.header("3. Configuration Loading Test")
    try:
        from newsensor_streamlit.config import settings
        st.success("✓ Configuration loaded successfully")
        
        st.write(f"✓ MongoDB URI: {settings.mongodb_uri[:30]}...")
        st.write(f"✓ MongoDB Database: {settings.mongodb_database}")  
        st.write(f"✓ MongoDB Collection: {settings.mongodb_conversations_collection}")
        
    except Exception as e:
        st.error(f"✗ Configuration loading failed: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    st.header("4. MongoDB Connection Test")
    try:
        from newsensor_streamlit.database.mongodb_client import get_mongodb_client
        
        with st.spinner("Testing MongoDB connection..."):
            mongodb_client = get_mongodb_client()
            st.success("✓ MongoDB client created")
            
            # Test connection
            if mongodb_client.test_connection():
                st.success("✓ MongoDB connection successful!")
                
                # Test basic operations
                try:
                    collection = mongodb_client.conversations
                    
                    # Test write
                    test_doc = {"test_id": "streamlit_debug", "message": "connection test"}
                    result = collection.insert_one(test_doc)
                    
                    if result.acknowledged:
                        st.success(f"✓ Write test successful: {result.inserted_id}")
                        
                        # Clean up
                        collection.delete_one({"_id": result.inserted_id})
                        st.success("✓ Cleanup successful")
                    else:
                        st.error("✗ Write test failed")
                        
                except Exception as e:
                    st.error(f"✗ Database operation failed: {e}")
            else:
                st.error("✗ MongoDB connection failed")
                
    except Exception as e:
        st.error(f"✗ MongoDB client creation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    st.header("5. Conversation Service Test")
    try:
        from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService
        
        with st.spinner("Testing conversation service..."):
            service = ConversationMongoDBService()
            st.success("✓ ConversationMongoDBService created")
            
            # Test conversation creation
            test_conversation = {
                "conversation_id": "debug_test_conversation_streamlit",
                "doc_id": "debug_doc",
                "document_name": "Debug Test",
                "created_at": "2025-08-25T00:00:00",
                "updated_at": "2025-08-25T00:00:00", 
                "messages": [],
                "ragas_summary": {
                    "total_questions": 0,
                    "avg_faithfulness": 0.0,
                    "avg_answer_relevancy": 0.0,
                    "avg_context_precision": 0.0,
                    "avg_context_recall": 0.0
                }
            }
            
            if service.save_conversation(test_conversation):
                st.success("✓ Conversation save test successful")
                
                loaded = service.load_conversation("debug_test_conversation_streamlit")
                if loaded:
                    st.success("✓ Conversation load test successful")
                    
                    if service.delete_conversation("debug_test_conversation_streamlit"):
                        st.success("✓ Conversation delete test successful")
                    else:
                        st.warning("⚠️ Cleanup failed (conversation may still exist)")
                else:
                    st.error("✗ Conversation load test failed")
            else:
                st.error("✗ Conversation save test failed")
                
    except Exception as e:
        st.error(f"✗ Conversation service test failed: {e}")
        import traceback  
        st.code(traceback.format_exc())

if __name__ == "__main__":
    show_mongodb_debug()
