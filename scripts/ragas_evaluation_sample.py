from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
import json
import os

# Load the JSON dataset
dataset_file = "sensors_ragas_dataset_topk5.json"  # Change to "merged_sensors_ragas_dataset_topk5.json" for merged dataset
with open(dataset_file, "r") as f:
    data = json.load(f)

# Convert JSON entries to SingleTurnSample objects
samples = [
    SingleTurnSample(
        user_input=item["user_input"],
        response=item["response"],
        retrieved_contexts=item["retrieved_contexts"],
        reference=item["reference"]
    )
    for item in data
]

# Create an EvaluationDataset
dataset = EvaluationDataset.from_list(samples)

# Initialize the LLM for evaluation with ChatGPT-4o
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),  # Ensure your API key is set in environment variables
    temperature=0.0  # Set to 0 for deterministic evaluation
)
evaluator_llm = LangchainLLMWrapper(llm)

# Define metrics to evaluate
metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
]

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm
)

# Output results
print("Evaluation Results:")
print(results.to_pandas())

# Save results to CSV
output_file = "sensors_ragas_results_gpt4o.csv"
results.to_pandas().to_csv(output_file, index=False)
print(f"Results saved to {output_file}")