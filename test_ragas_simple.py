#!/usr/bin/env python3
"""Simple RAGAS test to debug the issue."""

import os
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

# Set up the evaluator
chat_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)
evaluator_llm = LangchainLLMWrapper(chat_llm)

# Test data
question = "What is the battery life of the EVA-2311?"
answer = "The battery life is up to 5 years at 25°C with 15-minute intervals."
contexts = [
    "Battery life up till 5 years @25°C 15mins interval setting",
    "EVA-2311 is powered by 2 AA 3.6v Li-ion batteries"
]
reference = "The EVA-2311 has a battery life of up to 5 years at 25°C."

# Create sample
sample = SingleTurnSample(
    user_input=question,
    response=answer,
    retrieved_contexts=contexts,
    reference=reference
)

print("Sample created successfully!")
print(f"Sample: {sample}")

# Create dataset
dataset = EvaluationDataset.from_list([sample])
print("Dataset created successfully!")

# Define metrics
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall()
]

print("Running evaluation...")
try:
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm
    )
    print("Evaluation successful!")
    print(result.to_pandas())
except Exception as e:
    print(f"Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
