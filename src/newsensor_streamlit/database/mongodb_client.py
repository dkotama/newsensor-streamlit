import os
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import streamlit as st


class MongoDBClient:
    """MongoDB client for handling conversations storage."""
    
    def __init__(self, config=None):
        # Import here to avoid circular imports
        if config is None:
            from ..config import settings
            config = settings
            
        self.uri = config.mongodb_uri
        self.db_name = config.mongodb_database
        self.conversations_collection_name = config.mongodb_conversations_collection
        
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        
    @property
    def client(self) -> MongoClient:
        """Get MongoDB client instance."""
        if self._client is None:
            self._client = MongoClient(self.uri)
        return self._client
    
    @property
    def db(self) -> Database:
        """Get database instance."""
        if self._db is None:
            self._db = self.client[self.db_name]
        return self._db
    
    @property
    def conversations(self) -> Collection:
        """Get conversations collection."""
        return self.db[self.conversations_collection_name]
    
    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
    
    def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return False


# Singleton instance using Streamlit cache
@st.cache_resource
def get_mongodb_client() -> MongoDBClient:
    """Get cached MongoDB client instance."""
    return MongoDBClient()
