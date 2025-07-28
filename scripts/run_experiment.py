#!/usr/bin/env python3
"""
Experiment runner for comparing standard vs re-ranking retrieval methods.
Uses existing documents from Qdrant collection without re-ingesting.

Usage with UV package manager:
    uv run python scripts/run_experiment.py
    uv run python scripts/run_experiment.py --list-docs  
    uv run python scripts/run_experiment.py --doc-id your_doc_id
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from newsensor_streamlit.core.chat_engine import ChatEngine
    from newsensor_streamlit.services.qdrant_service import QdrantService
    from newsensor_streamlit.config import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to run with UV: uv run python scripts/run_experiment.py")
    sys.exit(1)

def print_header():
    """Print experiment header."""
    print("=" * 80)
    print("🧪 NEWSENSOR RE-RANKING EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Collection: {settings.qdrant_collection}")
    print()

def print_separator():
    """Print section separator."""
    print("-" * 80)

def format_metrics(metrics: Dict[str, float], title: str) -> str:
    """Format metrics for display with colors."""
    if not metrics:
        return f"{title}: No metrics available"
    
    formatted = f"{title}:\n"
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            # Color coding
            if value >= 0.7:
                color = "🟢"
            elif value >= 0.5:
                color = "🟡"
            else:
                color = "🔴"
            formatted += f"  {color} {metric.replace('_', ' ').title()}: {value:.3f}\n"
        else:
            formatted += f"  • {metric.replace('_', ' ').title()}: {value}\n"
    
    return formatted.rstrip()

def get_available_documents(qdrant_service: QdrantService) -> List[str]:
    """Get list of available document IDs from Qdrant collection."""
    try:
        # Use the client to get points with unique doc_ids
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        result = qdrant_service.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=1000,  # Get many points to find unique doc_ids
            with_payload=["doc_id"]
        )
        
        doc_ids = set()
        for point in result[0]:
            if point.payload and "doc_id" in point.payload:
                doc_ids.add(point.payload["doc_id"])
        
        return sorted(list(doc_ids))
        
    except Exception as e:
        print(f"❌ Error getting documents: {e}")
        return []

def select_document_interactive(available_docs: List[str]) -> str:
    """Interactive document selection."""
    print("📄 Available Documents:")
    print()
    
    for i, doc_id in enumerate(available_docs, 1):
        print(f"  {i}. {doc_id}")
    
    print()
    while True:
        try:
            choice = input(f"Select document (1-{len(available_docs)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_docs):
                return available_docs[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(available_docs)}")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            return None

def load_test_questions() -> List[str]:
    """Load questions from test dataset."""
    try:
        dataset_path = Path("data/evaluation/test_dataset.csv")
        if dataset_path.exists():
            import pandas as pd
            df = pd.read_csv(dataset_path)
            return df['question'].tolist()
    except Exception as e:
        print(f"⚠️  Could not load test dataset: {e}")
    
    # Fallback questions
    return [
        "What is the operating voltage range for this sensor?",
        "What is the I2C address of the sensor?",
        "How do I configure the sensor for continuous measurement mode?",
        "What are the pin connections needed for Arduino?",
        "What is the measurement accuracy of this sensor?",
        "How do I read sensor data via I2C?",
        "What is the power consumption in sleep mode?",
        "How do I calibrate the sensor?"
    ]

async def run_single_comparison(chat_engine: ChatEngine, doc_id: str, question: str, question_num: int, total_questions: int):
    """Run a single question comparison between standard and re-ranking."""
    
    print(f"\n📝 Question {question_num}/{total_questions}")
    print(f"❓ {question}")
    print()
    
    # Standard retrieval
    print("🔍 Testing STANDARD retrieval...")
    start_time = time.time()
    try:
        standard_response = chat_engine.ask_question(
            question=question,
            doc_id=doc_id,
            use_reranking=False
        )
        standard_time = time.time() - start_time
        standard_success = True
    except Exception as e:
        print(f"❌ Standard retrieval failed: {e}")
        standard_response = {"answer": "Error", "ragas_metrics": {}, "retrieval_metadata": {}}
        standard_time = 0
        standard_success = False
    
    # Re-ranking retrieval
    print("🎯 Testing RE-RANKING retrieval...")
    start_time = time.time()
    try:
        reranked_response = chat_engine.ask_question(
            question=question,
            doc_id=doc_id,
            use_reranking=True
        )
        reranked_time = time.time() - start_time
        reranked_success = True
    except Exception as e:
        print(f"❌ Re-ranking retrieval failed: {e}")
        reranked_response = {"answer": "Error", "ragas_metrics": {}, "retrieval_metadata": {}}
        reranked_time = 0
        reranked_success = False
    
    print_separator()
    
    # Display results
    if standard_success:
        print("📊 STANDARD RETRIEVAL RESULTS:")
        print(f"⏱️  Time: {standard_time:.2f}s")
        print(f"📄 Answer: {standard_response['answer'][:200]}{'...' if len(standard_response['answer']) > 200 else ''}")
        if standard_response.get("ragas_metrics"):
            print(format_metrics(standard_response["ragas_metrics"], "🎯 RAGAS Metrics"))
        print()
    
    if reranked_success:
        print("🎯 RE-RANKING RETRIEVAL RESULTS:")
        print(f"⏱️  Time: {reranked_time:.2f}s")
        print(f"📄 Answer: {reranked_response['answer'][:200]}{'...' if len(reranked_response['answer']) > 200 else ''}")
        if reranked_response.get("ragas_metrics"):
            print(format_metrics(reranked_response["ragas_metrics"], "🎯 RAGAS Metrics"))
        if reranked_response.get("retrieval_metadata"):
            metadata = reranked_response["retrieval_metadata"]
            if metadata.get("reranking_enabled"):
                print(f"🔄 Retrieval: {metadata.get('initial_vector_search_count', 0)} → {metadata.get('final_chunk_count', 0)} docs")
                print(f"📈 Best Score: {metadata.get('best_relevance_score', 0):.3f}")
        print()
    
    # Performance comparison
    if standard_success and reranked_success:
        time_diff = reranked_time - standard_time
        if time_diff > 0:
            print(f"⏱️  Re-ranking was {time_diff:.2f}s slower")
        else:
            print(f"⏱️  Re-ranking was {abs(time_diff):.2f}s faster")
        
        # Compare RAGAS metrics if available
        if (standard_response.get("ragas_metrics") and reranked_response.get("ragas_metrics")):
            print("\n📈 RAGAS METRIC COMPARISON:")
            std_metrics = standard_response["ragas_metrics"]
            re_metrics = reranked_response["ragas_metrics"]
            
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric in std_metrics and metric in re_metrics:
                    std_val = std_metrics[metric]
                    re_val = re_metrics[metric]
                    diff = re_val - std_val
                    
                    if diff > 0.05:
                        trend = "🟢 ↗️"
                    elif diff < -0.05:
                        trend = "🔴 ↘️"
                    else:
                        trend = "🟡 ≈"
                    
                    print(f"  {trend} {metric.replace('_', ' ').title()}: {std_val:.3f} → {re_val:.3f} ({diff:+.3f})")
    
    print_separator()
    
    return {
        "question": question,
        "standard_success": standard_success,
        "reranked_success": reranked_success,
        "standard_time": standard_time if standard_success else None,
        "reranked_time": reranked_time if reranked_success else None,
        "standard_metrics": standard_response.get("ragas_metrics", {}) if standard_success else {},
        "reranked_metrics": reranked_response.get("ragas_metrics", {}) if reranked_success else {},
        "time_difference": reranked_time - standard_time if (standard_success and reranked_success) else None
    }

def generate_experiment_summary(results: List[Dict]) -> Dict:
    """Generate experiment summary statistics."""
    if not results:
        return {}
    
    successful_results = [r for r in results if r["standard_success"] and r["reranked_success"]]
    
    if not successful_results:
        return {"error": "No successful comparisons"}
    
    # Time analysis
    avg_standard_time = sum(r["standard_time"] for r in successful_results) / len(successful_results)
    avg_reranked_time = sum(r["reranked_time"] for r in successful_results) / len(successful_results)
    
    # Metric improvements
    metric_improvements = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        improvements = []
        for result in successful_results:
            if (metric in result["standard_metrics"] and metric in result["reranked_metrics"]):
                std_val = result["standard_metrics"][metric]
                re_val = result["reranked_metrics"][metric]
                improvements.append(re_val - std_val)
        
        if improvements:
            metric_improvements[metric] = {
                "avg_improvement": sum(improvements) / len(improvements),
                "improved_count": sum(1 for imp in improvements if imp > 0.05),
                "degraded_count": sum(1 for imp in improvements if imp < -0.05),
                "similar_count": sum(1 for imp in improvements if -0.05 <= imp <= 0.05)
            }
    
    return {
        "total_questions": len(results),
        "successful_comparisons": len(successful_results),
        "avg_standard_time": avg_standard_time,
        "avg_reranked_time": avg_reranked_time,
        "avg_time_overhead": avg_reranked_time - avg_standard_time,
        "metric_improvements": metric_improvements
    }

def print_final_summary(summary: Dict):
    """Print final experiment summary."""
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if "error" in summary:
        print(f"❌ {summary['error']}")
        return
    
    print(f"✅ Successful comparisons: {summary['successful_comparisons']}/{summary['total_questions']}")
    print(f"⏱️  Average standard time: {summary['avg_standard_time']:.2f}s")
    print(f"⏱️  Average re-ranking time: {summary['avg_reranked_time']:.2f}s")
    print(f"📈 Average time overhead: {summary['avg_time_overhead']:.2f}s")
    print()
    
    if summary.get("metric_improvements"):
        print("🎯 RAGAS METRIC IMPROVEMENTS:")
        for metric, data in summary["metric_improvements"].items():
            print(f"\n📊 {metric.replace('_', ' ').title()}:")
            print(f"  • Average improvement: {data['avg_improvement']:+.3f}")
            print(f"  • Better: {data['improved_count']} questions")
            print(f"  • Similar: {data['similar_count']} questions") 
            print(f"  • Worse: {data['degraded_count']} questions")
    
    print("\n💡 RECOMMENDATIONS:")
    if summary['avg_time_overhead'] < 2.0:
        print("  ✅ Re-ranking overhead is acceptable (< 2s)")
    else:
        print("  ⚠️  Re-ranking overhead is high (> 2s) - consider optimizing")
    
    # Check if re-ranking is generally better
    if summary.get("metric_improvements"):
        improvements = [data['avg_improvement'] for data in summary["metric_improvements"].values()]
        avg_overall_improvement = sum(improvements) / len(improvements) if improvements else 0
        
        if avg_overall_improvement > 0.05:
            print("  🎯 Re-ranking shows significant quality improvements - recommended for important queries")
        elif avg_overall_improvement > 0:
            print("  🟡 Re-ranking shows marginal improvements - use for critical queries")
        else:
            print("  ⚠️  Re-ranking doesn't show clear quality improvements")

async def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Compare standard vs re-ranking retrieval")
    parser.add_argument("--doc-id", help="Specific document ID to test")
    parser.add_argument("--list-docs", action="store_true", help="List available documents")
    parser.add_argument("--questions", nargs="+", help="Custom questions to test")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    print_header()
    
    # Initialize services
    print("🔧 Initializing services...")
    try:
        chat_engine = ChatEngine()
        qdrant_service = QdrantService()
        
        # Check service availability
        reranking_available = hasattr(chat_engine, 'reranking_service') and chat_engine.reranking_service is not None
        ragas_available = hasattr(chat_engine, 'ragas_evaluator') and chat_engine.ragas_evaluator and chat_engine.ragas_evaluator.is_available()
        
        print(f"  • Re-ranking: {'✅ Available' if reranking_available else '❌ Not available'}")
        print(f"  • RAGAS: {'✅ Available' if ragas_available else '❌ Not available'}")
        
        if not reranking_available:
            print("\n⚠️  Re-ranking not available. Install with: uv add sentence-transformers torch transformers")
            
        if not ragas_available:
            print("\n⚠️  RAGAS not available. Install with: uv add ragas datasets")
            print("     Also configure OPENAI_API_KEY in .env file")
        
        if not (reranking_available or ragas_available):
            print("\n❌ Neither re-ranking nor RAGAS available. Experiment will be limited.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return 1
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return 1
    
    print_separator()
    
    # Handle document listing
    if args.list_docs:
        print("📄 Getting available documents from Qdrant...")
        available_docs = get_available_documents(qdrant_service)
        
        if not available_docs:
            print("❌ No documents found in collection")
            return 1
        
        print(f"\n📚 Found {len(available_docs)} documents:")
        for i, doc_id in enumerate(available_docs, 1):
            print(f"  {i}. {doc_id}")
        
        return 0
    
    # Get document ID
    doc_id = args.doc_id
    if not doc_id:
        available_docs = get_available_documents(qdrant_service)
        if not available_docs:
            print("❌ No documents found in collection")
            return 1
        
        doc_id = select_document_interactive(available_docs)
        if not doc_id:
            print("❌ No document selected")
            return 1
    
    print(f"📄 Using document: {doc_id}")
    
    # Get questions
    questions = args.questions if args.questions else load_test_questions()
    print(f"❓ Testing {len(questions)} questions")
    
    print_separator()
    
    # Run experiments
    results = []
    for i, question in enumerate(questions, 1):
        result = await run_single_comparison(chat_engine, doc_id, question, i, len(questions))
        results.append(result)
        
        # Small delay between questions
        if i < len(questions):
            await asyncio.sleep(0.5)
    
    # Generate summary
    summary = generate_experiment_summary(results)
    print_final_summary(summary)
    
    # Save results if requested
    if not args.no_save:
        output_file = args.output or f"experiments/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        experiment_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "doc_id": doc_id,
                "questions_count": len(questions),
                "reranking_available": reranking_available,
                "ragas_available": ragas_available
            },
            "questions": questions,
            "results": results,
            "summary": summary
        }
        
        with open(output_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
