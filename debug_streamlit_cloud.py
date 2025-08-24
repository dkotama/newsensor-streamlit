#!/usr/bin/env python3
"""
Streamlit Cloud specific debug script.
This should be run on Streamlit Cloud to diagnose the MongoDB issue.
"""

import streamlit as st
import os
import traceback
from pathlib import Path
import sys

st.title("MongoDB Connection Debug - Streamlit Cloud")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

st.header("Environment Debug")
st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")

st.header("Streamlit Secrets")
st.write("Available secrets:")
try:
    if hasattr(st, 'secrets'):
        for key in st.secrets:
            if 'mongodb' in key.lower():
                value = str(st.secrets[key])
                # Mask password
                if '@' in value and 'mongodb' in key:
                    masked_value = value.split('@')[0].split('//')[0] + '//' + '*****@' + value.split('@')[1]
                    st.write(f"{key}: {masked_value}")
                else:
                    st.write(f"{key}: {value}")
    else:
        st.write("No st.secrets available")
except Exception as e:
    st.error(f"Error accessing secrets: {e}")

st.header("Environment Variables")
mongo_vars = [key for key in os.environ if 'mongo' in key.lower()]
if mongo_vars:
    for var in sorted(mongo_vars):
        value = os.environ[var]
        # Mask password in URI
        if 'mongodb_uri' in var.lower() and '@' in value:
            masked_value = value.split('@')[0].split('//')[0] + '//' + '*****@' + value.split('@')[1]
            st.write(f"{var}: {masked_value}")
        else:
            st.write(f"{var}: {value}")
else:
    st.write("No MongoDB-related environment variables found")

st.header("Direct Configuration Test")
try:
    from newsensor_streamlit.config import settings
    st.success("Configuration loaded successfully")
    st.write(f"MongoDB URI: {settings.mongodb_uri[:30]}...")
    st.write(f"MongoDB Database: {settings.mongodb_database}")
    st.write(f"MongoDB Collection: {settings.mongodb_conversations_collection}")
except Exception as e:
    st.error(f"Configuration loading failed: {e}")
    st.text(traceback.format_exc())

st.header("PyMongo Connection Test")
try:
    from pymongo import MongoClient
    
    # Try to get URI from different sources
    mongodb_uri = None
    
    # Method 1: Try from st.secrets
    if hasattr(st, 'secrets') and hasattr(st.secrets, 'mongodb') and hasattr(st.secrets.mongodb, 'mongodb_uri'):
        mongodb_uri = st.secrets.mongodb.mongodb_uri
        st.info("Using MongoDB URI from st.secrets.mongodb.mongodb_uri")
    
    # Method 2: Try from st.secrets direct
    elif hasattr(st, 'secrets') and 'mongodb_uri' in st.secrets:
        mongodb_uri = st.secrets.mongodb_uri
        st.info("Using MongoDB URI from st.secrets.mongodb_uri")
    
    # Method 3: Try from environment
    elif 'MONGODB_URI' in os.environ:
        mongodb_uri = os.environ['MONGODB_URI']
        st.info("Using MongoDB URI from environment variable")
    
    # Method 4: Try from config
    else:
        try:
            from newsensor_streamlit.config import settings
            mongodb_uri = settings.mongodb_uri
            st.info("Using MongoDB URI from settings configuration")
        except:
            pass
    
    if not mongodb_uri:
        st.error("No MongoDB URI found in any configuration source")
        st.stop()
    
    st.write(f"Testing with URI: {mongodb_uri[:30]}...")
    
    client = MongoClient(mongodb_uri)
    client.admin.command('ping')
    st.success("✓ MongoDB connection successful!")
    
    # Test database access
    db_name = getattr(st.secrets.mongodb, 'mongodb_database', 'newsensor_conversations')
    db = client[db_name]
    st.success(f"✓ Database '{db_name}' accessible")
    
    # Test collection access
    collection_name = getattr(st.secrets.mongodb, 'mongodb_conversations_collection', 'conversations')
    collection = db[collection_name]
    st.success(f"✓ Collection '{collection_name}' accessible")
    
    # Test write
    test_doc = {"test": "streamlit_cloud_test", "timestamp": "2025-08-25"}
    result = collection.insert_one(test_doc)
    if result.acknowledged:
        st.success(f"✓ Write test successful: {result.inserted_id}")
        collection.delete_one({"_id": result.inserted_id})
        st.success("✓ Cleanup successful")
    else:
        st.error("✗ Write test failed")
    
    client.close()

except Exception as e:
    st.error(f"MongoDB connection failed: {e}")
    st.text(traceback.format_exc())

st.header("Conversation Service Test")
try:
    from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService
    
    service = ConversationMongoDBService()
    st.success("✓ ConversationMongoDBService created")
    
    test_conversation = {
        "conversation_id": "streamlit_cloud_test",
        "doc_id": "test_doc",
        "document_name": "Test Document",
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
    
    save_result = service.save_conversation(test_conversation)
    if save_result:
        st.success("✓ Conversation save test successful")
        
        load_result = service.load_conversation("streamlit_cloud_test")
        if load_result:
            st.success("✓ Conversation load test successful")
            
            delete_result = service.delete_conversation("streamlit_cloud_test")
            st.success(f"✓ Cleanup successful: {delete_result}")
        else:
            st.error("✗ Conversation load test failed")
    else:
        st.error("✗ Conversation save test failed")

except Exception as e:
    st.error(f"Conversation service test failed: {e}")
    st.text(traceback.format_exc())
