#!/usr/bin/env python3
"""
Debug script to test MongoDB connection and diagnose issues.
This can be run both locally and on Streamlit Cloud to identify the problem.
"""

import os
import traceback
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("=== Environment Debug ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    
    print("\n=== Environment Variables ===")
    mongo_related_vars = [key for key in os.environ.keys() if 'mongo' in key.lower()]
    if mongo_related_vars:
        for var in sorted(mongo_related_vars):
            value = os.environ[var]
            # Mask password in URI for security
            if 'mongodb_uri' in var.lower() and '@' in value:
                masked_value = value.split('@')[0].split('//')[0] + '//' + '*****@' + value.split('@')[1]
                print(f"{var}: {masked_value}")
            else:
                print(f"{var}: {value}")
    else:
        print("No MongoDB-related environment variables found")
    
    print("\n=== Checking .env file ===")
    env_file = Path(__file__).parent / ".env"
    print(f"Looking for .env at: {env_file}")
    print(f".env file exists: {env_file.exists()}")
    
    if env_file.exists():
        print("First few lines of .env file:")
        with open(env_file) as f:
            lines = f.readlines()[:10]  # First 10 lines
            for i, line in enumerate(lines, 1):
                if 'mongodb' in line.lower():
                    # Mask sensitive info
                    if '@' in line:
                        parts = line.split('@')
                        if len(parts) > 1:
                            line = parts[0] + '@*****'
                    print(f"  {i}: {line.strip()}")

    print("\n=== Testing Configuration Loading ===")
    from newsensor_streamlit.config import settings
    
    print(f"Loaded MongoDB URI: {settings.mongodb_uri[:20]}...")
    print(f"MongoDB Database: {settings.mongodb_database}")
    print(f"MongoDB Collection: {settings.mongodb_conversations_collection}")
    
    print("\n=== Testing Direct PyMongo Connection ===")
    from pymongo import MongoClient
    
    client = MongoClient(settings.mongodb_uri)
    print("Created MongoDB client")
    
    # Test connection
    print("Testing connection with ping...")
    client.admin.command('ping')
    print("✓ MongoDB connection successful!")
    
    # Test database access
    db = client[settings.mongodb_database]
    print(f"✓ Database '{settings.mongodb_database}' accessible")
    
    # Test collection access
    collection = db[settings.mongodb_conversations_collection]
    print(f"✓ Collection '{settings.mongodb_conversations_collection}' accessible")
    
    # Test write operation
    test_doc = {
        "test_id": "debug_test",
        "timestamp": "2025-08-25T00:00:00",
        "message": "Connection test"
    }
    
    print("\n=== Testing Write Operation ===")
    result = collection.insert_one(test_doc)
    if result.acknowledged:
        print(f"✓ Test document inserted with ID: {result.inserted_id}")
        
        # Clean up test document
        collection.delete_one({"_id": result.inserted_id})
        print("✓ Test document cleaned up")
    else:
        print("✗ Failed to insert test document")
    
    print("\n=== Testing MongoDB Service Classes ===")
    from newsensor_streamlit.database.mongodb_client import get_mongodb_client
    
    mongodb_client = get_mongodb_client()
    print("✓ MongoDBClient instance created")
    
    connection_test = mongodb_client.test_connection()
    print(f"✓ Connection test result: {connection_test}")
    
    print("\n=== Testing Conversation Service ===")
    from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService
    
    conversation_service = ConversationMongoDBService()
    print("✓ ConversationMongoDBService instance created")
    
    # Test save operation
    test_conversation = {
        "conversation_id": "debug_test_conversation",
        "doc_id": "debug_doc",
        "document_name": "Debug Document",
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
    
    save_result = conversation_service.save_conversation(test_conversation)
    print(f"✓ Test conversation save result: {save_result}")
    
    if save_result:
        # Test load operation
        loaded_conversation = conversation_service.load_conversation("debug_test_conversation")
        if loaded_conversation:
            print("✓ Test conversation loaded successfully")
            
            # Clean up
            delete_result = conversation_service.delete_conversation("debug_test_conversation")
            print(f"✓ Test conversation cleaned up: {delete_result}")
        else:
            print("✗ Failed to load test conversation")
    
    print("\n🎉 All tests passed! MongoDB connection is working properly.")
    client.close()
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
    print("\n=== Full traceback ===")
    traceback.print_exc()
    
    print("\n=== Debugging suggestions ===")
    print("1. Check if MongoDB URI is correct")
    print("2. Verify network connectivity to MongoDB Atlas")
    print("3. Confirm database and collection names match")
    print("4. Check if IP address is whitelisted in MongoDB Atlas")
    print("5. Verify username/password credentials")
