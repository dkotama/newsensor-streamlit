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
        
        # Try to get MongoDB configuration from multiple sources
        # This handles both local .env files and Streamlit Cloud secrets
        self.uri = self._get_mongodb_uri(config)
        self.db_name = self._get_mongodb_database(config)
        self.conversations_collection_name = self._get_mongodb_collection(config)
        
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
    
    def _get_mongodb_uri(self, config) -> str:
        """Get MongoDB URI from various sources."""
        # Method 1: Try Streamlit secrets (nested structure)
        try:
            if hasattr(st, 'secrets') and hasattr(st.secrets, 'mongodb') and hasattr(st.secrets.mongodb, 'mongodb_uri'):
                return st.secrets.mongodb.mongodb_uri
        except:
            pass
        
        # Method 2: Try Streamlit secrets (flat structure)
        try:
            if hasattr(st, 'secrets') and 'mongodb_uri' in st.secrets:
                return st.secrets.mongodb_uri
        except:
            pass
        
        # Method 3: Try environment variable
        if 'MONGODB_URI' in os.environ:
            return os.environ['MONGODB_URI']
        
        # Method 4: Try lowercase environment variable
        if 'mongodb_uri' in os.environ:
            return os.environ['mongodb_uri']
        
        # Method 5: Try from config/settings
        return getattr(config, 'mongodb_uri', 'mongodb://localhost:27017')
    
    def _get_mongodb_database(self, config) -> str:
        """Get MongoDB database name from various sources."""
        # Method 1: Try Streamlit secrets (nested)
        try:
            if hasattr(st, 'secrets') and hasattr(st.secrets, 'mongodb') and hasattr(st.secrets.mongodb, 'mongodb_database'):
                return st.secrets.mongodb.mongodb_database
        except:
            pass
        
        # Method 2: Try Streamlit secrets (flat)
        try:
            if hasattr(st, 'secrets') and 'mongodb_database' in st.secrets:
                return st.secrets.mongodb_database
        except:
            pass
        
        # Method 3: Try environment
        if 'MONGODB_DATABASE' in os.environ:
            return os.environ['MONGODB_DATABASE']
        
        if 'mongodb_database' in os.environ:
            return os.environ['mongodb_database']
        
        # Method 4: From config
        return getattr(config, 'mongodb_database', 'newsensor_conversations')
    
    def _get_mongodb_collection(self, config) -> str:
        """Get MongoDB collection name from various sources."""
        # Method 1: Try Streamlit secrets (nested)
        try:
            if hasattr(st, 'secrets') and hasattr(st.secrets, 'mongodb') and hasattr(st.secrets.mongodb, 'mongodb_conversations_collection'):
                return st.secrets.mongodb.mongodb_conversations_collection
        except:
            pass
        
        # Method 2: Try Streamlit secrets (flat)
        try:
            if hasattr(st, 'secrets') and 'mongodb_conversations_collection' in st.secrets:
                return st.secrets.mongodb_conversations_collection
        except:
            pass
        
        # Method 3: Try environment
        if 'MONGODB_CONVERSATIONS_COLLECTION' in os.environ:
            return os.environ['MONGODB_CONVERSATIONS_COLLECTION']
        
        if 'mongodb_conversations_collection' in os.environ:
            return os.environ['mongodb_conversations_collection']
        
        # Method 4: From config
        return getattr(config, 'mongodb_conversations_collection', 'conversations')
        
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
