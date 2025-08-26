#!/usr/bin/env python3
"""Test using the exact pattern from the working sample script."""

import os
import sys
sys.path.append('src')

from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from newsensor_streamlit.config import settings

# Initialize the LLM for evaluation exactly like the sample
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=settings.openai_api_key,  # Use from settings
    temperature=0.0
)
evaluator_llm = LangchainLLMWrapper(llm)

# Test data exactly like the sample would have
test_data = [{
    "user_input": "What is the battery life of the EVA-2311?",
    "response": "The battery life is up to 5 years at 25°C with 15-minute intervals.",
    "retrieved_contexts": [
        "Battery life up till 5 years @25°C 15mins interval setting",
        "EVA-2311 is powered by 2 AA 3.6v Li-ion batteries"
    ],
    "reference": "The EVA-2311 has a battery life of up to 5 years at 25°C."
}]

# Create an EvaluationDataset using raw dict data (not pre-converted samples)
dataset = EvaluationDataset.from_list(test_data)
print("Dataset created successfully")

# Define metrics exactly like the sample
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall()
]

print("Running evaluation...")
try:
    # Run evaluation exactly like the sample
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm
    )
    
    print("Evaluation successful!")
    print(results.to_pandas())
    
except Exception as e:
    print(f"Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
