#!/usr/bin/env python3
"""Test script to verify RAGAS evaluation pipeline works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from newsensor_streamlit.services.ragas_evaluator import RagasEvaluator
from newsensor_streamlit.config import settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ragas_evaluator():
    """Test RAGAS evaluator with sample data."""
    
    print("🔧 Testing RAGAS Evaluator Pipeline")
    print(f"OpenAI API Key configured: {'Yes' if settings.openai_api_key else 'No'}")
    print(f"RAGAS Model: {settings.ragas_evaluator_model}")
    print("-" * 50)
    
    # Create evaluator instance
    evaluator = RagasEvaluator()
    
    # Check availability
    available = evaluator.is_available()
    print(f"RAGAS Available: {available}")
    
    if not available:
        print("❌ RAGAS not available - check dependencies and API key")
        return False
    
    # Sample evaluation data
    test_question = "What is the operating temperature range of the S15S sensor?"
    test_answer = "The S15S Temperature and Humidity Sensor operates in the temperature range of -40°C to +85°C for measurement and -10°C to +60°C for optimal performance."
    test_contexts = [
        "The S15S sensor specification shows operating temperature: -40°C to +85°C measurement range",
        "Environmental conditions: Operating temperature -10°C to +60°C for best accuracy", 
        "Temperature sensor range: -40 to +85 degrees Celsius"
    ]
    
    print("📝 Test Data:")
    print(f"Question: {test_question}")
    print(f"Answer: {test_answer[:100]}...")
    print(f"Contexts: {len(test_contexts)} context chunks")
    print("-" * 50)
    
    try:
        print("🚀 Running RAGAS evaluation...")
        result = evaluator.evaluate_answer(
            question=test_question,
            answer=test_answer,
            contexts=test_contexts
        )
        
        if result:
            print("✅ RAGAS Evaluation Results:")
            for metric, score in result.items():
                print(f"  {metric}: {score}")
            
            # Verify these are real scores (not the old heuristics)
            if (result.get("faithfulness") != 0.85 or 
                result.get("answer_relevancy") != 0.78):
                print("✅ Real RAGAS evaluation confirmed (scores differ from heuristics)")
            else:
                print("⚠️  Warning: Scores match old heuristics - may not be real evaluation")
                
            return True
        else:
            print("❌ RAGAS evaluation returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return False

def test_batch_evaluation():
    """Test batch evaluation functionality."""
    
    print("\n🔧 Testing Batch RAGAS Evaluation")
    print("-" * 50)
    
    evaluator = RagasEvaluator()
    
    if not evaluator.is_available():
        print("❌ RAGAS not available for batch testing")
        return False
    
    # Sample batch data
    batch_data = [
        {
            "question": "What is the humidity range of the sensor?",
            "answer": "The sensor measures humidity from 0% to 100% RH.",
            "contexts": ["Humidity measurement range: 0-100% relative humidity"]
        },
        {
            "question": "What is the supply voltage?",
            "answer": "The supply voltage is 3.3V to 5V DC.",
            "contexts": ["Power supply: 3.3V to 5.0V DC input voltage"]
        }
    ]
    
    try:
        print(f"📊 Evaluating {len(batch_data)} questions...")
        results = evaluator.batch_evaluate(batch_data)
        
        if results and len(results) == len(batch_data):
            print("✅ Batch evaluation completed:")
            for i, result in enumerate(results):
                print(f"  Question {i+1}: {len(result)} metrics")
                if result:
                    avg_score = sum(result.values()) / len(result)
                    print(f"    Average score: {avg_score:.3f}")
            return True
        else:
            print("❌ Batch evaluation failed or incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 RAGAS Pipeline Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test individual evaluation
    if not test_ragas_evaluator():
        success = False
    
    # Test batch evaluation  
    if not test_batch_evaluation():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All RAGAS tests passed!")
        sys.exit(0)
    else:
        print("💥 Some RAGAS tests failed!")
        sys.exit(1)
