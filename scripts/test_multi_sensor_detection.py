#!/usr/bin/env python3
"""
Test script for the improved multi-sensor detection functionality.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from newsensor_streamlit.services.qdrant_service import QdrantService
from loguru import logger

def test_multi_sensor_detection():
    """Test the multi-sensor detection and filtering capability."""
    
    print("🔍 Testing Multi-Sensor Detection & Filtering")
    print("=" * 60)
    
    # Initialize Qdrant service
    qdrant_service = QdrantService()
    
    # Test queries with multiple sensors
    test_queries = [
        "Compare the battery life of EVA-2311 and S15S temperature sensors",
        "What is the temperature range for EVA-2311 vs LS219 sensors?",
        "Show me specifications for PJ85775 and EVA-2311 sensors",
        "EVA-2311 sensor accuracy compared to S15S-1234",
        "Battery life of Advantech EVA-2311 sensor only",
        "What wireless technology does the temperature sensor use?"  # No specific sensor mentioned
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        
        # Test sensor extraction
        detected_sensors = qdrant_service._extract_sensor_models_from_query(query)
        print(f"Detected Sensors: {detected_sensors if detected_sensors else 'None'}")
        
        try:
            # Test multi-sensor search on sensors_with_metadata collection
            results = qdrant_service.search_with_multi_sensor_filter(
                query, k=3, collection_name="sensors_with_metadata"
            )
            print(f"Results Found: {len(results)}")
            
            if results:
                print("Sample Results:")
                for j, result in enumerate(results[:2], 1):  # Show first 2 results
                    sensor_model = result.metadata.get('sensor_model', 'unknown')
                    content_preview = result.page_content[:100] + "..." if len(result.page_content) > 100 else result.page_content
                    print(f"  {j}. Sensor: {sensor_model}")
                    print(f"     Content: {content_preview}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)

def test_known_sensors():
    """Test retrieval of known sensor models from the database."""
    print("\n🗄️  Testing Known Sensor Models Retrieval")
    print("=" * 60)
    
    qdrant_service = QdrantService()
    # Override collection name to use the one with data
    qdrant_service.collection_name = "sensors_with_metadata"
    
    try:
        known_sensors = qdrant_service._get_known_sensor_models()
        print(f"Found {len(known_sensors)} known sensor models:")
        
        for sensor in sorted(known_sensors)[:20]:  # Show first 20
            print(f"  - {sensor}")
        
        if len(known_sensors) > 20:
            print(f"  ... and {len(known_sensors) - 20} more")
            
    except Exception as e:
        print(f"Error retrieving known sensors: {e}")

def test_pattern_matching():
    """Test the regex patterns for sensor detection."""
    print("\n🔤 Testing Sensor Pattern Matching")
    print("=" * 60)
    
    qdrant_service = QdrantService()
    
    test_strings = [
        "EVA-2311 sensor specifications",
        "Compare S15S with LS219 sensors", 
        "PJ85775 datasheet information",
        "EVA2311 without hyphen",
        "Temperature sensor L219A specs",
        "Show me ABC-1234 and XYZ5678 details",
        "Generic temperature sensor info"  # Should detect nothing
    ]
    
    
    for test_string in test_strings:
        detected = qdrant_service._extract_sensor_models_from_query(test_string)
        print(f"'{test_string}'")
        print(f"  → Detected: {detected if detected else 'None'}")
        print()

if __name__ == "__main__":
    logger.info("🚀 Starting Multi-Sensor Detection Tests")
    
    test_pattern_matching()
    test_known_sensors()
    test_multi_sensor_detection()
    
    logger.info("✅ Multi-Sensor Detection Tests Completed")
