#!/usr/bin/env python3
"""
Test script to verify conversation saving is working.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.newsensor_streamlit.core.chat_engine import ChatEngine
from src.newsensor_streamlit.services.conversation_service import ConversationService
from loguru import logger


def test_conversation_saving():
    """Test that conversations are being saved properly."""
    logger.info("Testing conversation saving...")
    
    # Initialize services
    conversation_service = ConversationService()
    chat_engine = ChatEngine()
    
    # Check if we have any existing conversations
    existing_conversations = conversation_service.list_conversations()
    logger.info(f"Found {len(existing_conversations)} existing conversations")
    
    for conv in existing_conversations[:3]:  # Show first 3
        logger.info(f"  - {conv['conversation_id'][:8]}: {conv['document_name']} ({conv['message_count']} messages)")
    
    # Test creating a new conversation
    logger.info("\nCreating test conversation...")
    test_conversation_id = conversation_service.create_conversation(
        doc_id="test_doc_123", 
        document_name="Test Sensor Collection"
    )
    
    logger.info(f"Created conversation: {test_conversation_id}")
    
    # Test adding a message
    logger.info("Adding test message...")
    conversation_service.add_message(
        conversation_id=test_conversation_id,
        question="What is the operating temperature?",
        answer="The operating temperature range varies by sensor model.",
        sources=["test_source.pdf"],
        metrics={"faithfulness": 0.85, "answer_relevancy": 0.78},
        context="Test context for temperature sensor",
        ragas_metrics={"faithfulness": 0.85, "answer_relevancy": 0.78, "context_precision": 0.82, "context_recall": 0.80},
        retrieval_metadata={"reranking_enabled": False, "processing_time": 1.5}
    )
    
    # Verify the conversation was saved
    logger.info("Verifying conversation was saved...")
    saved_conversation = conversation_service.get_conversation(test_conversation_id)
    
    if saved_conversation:
        logger.info(f"✅ Conversation saved successfully!")
        logger.info(f"   Messages: {len(saved_conversation['messages'])}")
        logger.info(f"   RAGAS Summary: {saved_conversation['ragas_summary']}")
        
        # Check if the JSON file exists
        json_file = Path(f"data/conversations/{test_conversation_id}.json")
        if json_file.exists():
            logger.info(f"✅ JSON file exists: {json_file}")
            logger.info(f"   File size: {json_file.stat().st_size} bytes")
        else:
            logger.error(f"❌ JSON file not found: {json_file}")
    else:
        logger.error("❌ Conversation not found after saving!")
    
    # Test collection info
    logger.info("\nTesting collection info...")
    collection_info = chat_engine.qdrant_service.get_collection_info()
    logger.info(f"Collection exists: {collection_info['exists']}")
    logger.info(f"Points count: {collection_info['points_count']}")
    
    if collection_info["exists"] and collection_info["points_count"] > 0:
        available_sensors = chat_engine.qdrant_service.get_available_sensors()
        logger.info(f"Available sensors: {len(available_sensors)}")
        for sensor in available_sensors:
            logger.info(f"  - {sensor['sensor_model']} ({sensor['manufacturer']})")


if __name__ == "__main__":
    test_conversation_saving()
