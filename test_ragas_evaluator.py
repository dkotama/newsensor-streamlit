#!/usr/bin/env python3
"""Quick test of RAGAS evaluator."""

import sys
import os
sys.path.append('src')

from newsensor_streamlit.services.ragas_evaluator import RagasEvaluator

# Test the fixed evaluator
evaluator = RagasEvaluator()

if not evaluator.is_available():
    print("RAGAS not available - check API keys")
    exit(1)

print("RAGAS evaluator is available!")

# Test data
question = "What is the battery life of the EVA-2311?"
answer = "The battery life is up to 5 years at 25°C with 15-minute intervals."
contexts = [
    "Battery life up till 5 years @25°C 15mins interval setting",
    "EVA-2311 is powered by 2 AA 3.6v Li-ion batteries"
]
reference = "The EVA-2311 has a battery life of up to 5 years at 25°C."

print("Testing RAGAS evaluation...")
try:
    result = evaluator.evaluate_answer(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=reference
    )
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
