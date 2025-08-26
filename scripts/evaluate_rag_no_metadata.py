#!/usr/bin/env python3
"""
Script to evaluate the RAG system using RAGAS with ground truth data.
Tests the non-metadata collection (sensors_no_metadata) first.
"""

import sys
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.newsensor_streamlit.core.chat_engine import ChatEngine
from src.newsensor_streamlit.services.ragas_evaluator import RagasEvaluator
from src.newsensor_streamlit.config import settings
from loguru import logger


def load_ground_truth_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load ground truth dataset from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Direct array of questions
            dataset = data
        elif "questions" in data:
            # Structured format with questions array
            dataset = data["questions"]
        else:
            # Single question format
            dataset = [data]
        
        logger.info(f"Loaded {len(dataset)} ground truth samples from {file_path}")
        
        # Log first sample structure for debugging
        if dataset:
            sample_keys = list(dataset[0].keys())
            logger.info(f"Sample structure keys: {sample_keys}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load ground truth dataset: {e}")
        return []


def evaluate_no_metadata_rag(dataset: List[Dict[str, Any]], collection_name: str = "sensors_no_metadata") -> Dict[str, Any]:
    """Evaluate the RAG system using the no-metadata collection."""
    logger.info(f"Starting RAG evaluation on collection: {collection_name}")
    logger.info(f"Configuration:")
    logger.info(f"  - Evaluator Model: {settings.ragas_evaluator_model}")
    logger.info(f"  - Max Context Chunks (top_k): {settings.max_context_chunks}")
    logger.info(f"  - Collection: {collection_name}")
    
    # Initialize services
    chat_engine = ChatEngine()
    ragas_evaluator = RagasEvaluator()
    
    if not ragas_evaluator.is_available():
        logger.error("RAGAS evaluator is not available")
        return {}
    
    results = []
    total_processing_time = 0
    
    for i, sample in enumerate(dataset):
        logger.info(f"\n--- Evaluating Sample {i+1}/{len(dataset)} ---")
        
        # Extract fields from JSON structure
        question = sample["user_input"]  # Use correct field name
        ground_truth = sample["reference"]  # Use correct field name
        expected_response = sample.get("response", "")  # Optional field for comparison
        expected_contexts = sample.get("retrieved_contexts", "")  # Optional field for comparison
        
        logger.info(f"Question: {question}")
        logger.info(f"Ground Truth: {ground_truth}")
        
        try:
            # Get RAG response (using no-metadata collection via modified search)
            start_time = time.time()
            
            # We need to modify the search to use the specific collection
            # Since chat_engine.ask_question doesn't support collection_name directly,
            # we'll use the qdrant_service directly for this evaluation
            
            from src.newsensor_streamlit.services.embedding_service import EmbeddingService
            from src.newsensor_streamlit.services.qdrant_service import QdrantService
            from src.newsensor_streamlit.services.rag_service import RagService
            
            # Initialize services
            embedding_service = EmbeddingService()
            qdrant_service = QdrantService()
            rag_service = RagService()
            
            # Generate query embedding
            query_embedding = embedding_service.generate_embedding(
                question, task_type="RETRIEVAL_QUERY"
            )
            
            # Search in no-metadata collection
            context_chunks = qdrant_service.search_similar(
                query=question,
                doc_id="",  # Search across all documents
                k=settings.max_context_chunks,
                collection_name=collection_name
            )
            
            # Generate answer using RAG service
            answer_result = rag_service.generate_answer(
                question=question,
                context=context_chunks
            )
            generated_answer = answer_result["answer"]
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            logger.info(f"Generated Answer: {generated_answer[:100]}...")
            logger.info(f"Processing Time: {processing_time:.2f}s")
            logger.info(f"Retrieved {len(context_chunks)} context chunks")
            
            # Prepare contexts for RAGAS evaluation
            contexts = [chunk.page_content for chunk in context_chunks]
            
            # Log retrieved contexts for human review
            logger.info("Retrieved Contexts:")
            for idx, context in enumerate(contexts[:3]):  # Show first 3
                logger.info(f"  Context {idx+1}: {context[:100]}...")
            
            # Evaluate with RAGAS
            logger.info("Running RAGAS evaluation...")
            ragas_metrics = ragas_evaluator.evaluate_answer(
                question=question,
                answer=generated_answer,
                contexts=contexts,
                ground_truth=ground_truth
            )
            
            # Store comprehensive results for human review
            result = {
                "sample_id": i + 1,
                "question": question,
                "generated_answer": generated_answer,
                "ground_truth": ground_truth,
                "expected_response": expected_response,
                "expected_contexts": expected_contexts,
                "retrieved_contexts": contexts,
                "contexts_count": len(contexts),
                "processing_time": processing_time,
                "ragas_metrics": ragas_metrics,
                "context_sources": [
                    {
                        "chunk_index": idx,
                        "source": chunk.metadata.get("source", "unknown"),
                        "filename": chunk.metadata.get("filename", "unknown"),
                        "content_preview": chunk.page_content[:200]
                    }
                    for idx, chunk in enumerate(context_chunks)
                ]
            }
            results.append(result)
            
            # Log metrics
            if ragas_metrics:
                logger.info("RAGAS Metrics:")
                for metric, value in ragas_metrics.items():
                    logger.info(f"  {metric}: {value:.3f}")
            else:
                logger.warning("No RAGAS metrics generated")
                
        except Exception as e:
            logger.error(f"Failed to evaluate sample {i+1}: {e}")
            import traceback
            error_trace = traceback.format_exc()
            
            result = {
                "sample_id": i + 1,
                "question": question,
                "generated_answer": "ERROR",
                "ground_truth": ground_truth,
                "expected_response": expected_response,
                "expected_contexts": expected_contexts,
                "retrieved_contexts": [],
                "contexts_count": 0,
                "processing_time": 0,
                "ragas_metrics": {},
                "error": str(e),
                "error_trace": error_trace,
                "context_sources": []
            }
            results.append(result)
    
    # Calculate aggregate metrics
    successful_evaluations = [r for r in results if "error" not in r]
    failed_evaluations = [r for r in results if "error" in r]
    
    aggregate_metrics = {}
    if successful_evaluations:
        # Calculate averages for each RAGAS metric
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        for metric in metric_names:
            values = [r["ragas_metrics"].get(metric, 0) for r in successful_evaluations if r["ragas_metrics"]]
            aggregate_metrics[metric] = {
                "mean": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "count": len(values)
            }
    
    # Compile final evaluation report
    evaluation_summary = {
        "metadata": {
            "collection_name": collection_name,
            "evaluator_model": settings.ragas_evaluator_model,
            "top_k_chunks": settings.max_context_chunks,
            "total_samples": len(dataset),
            "successful_evaluations": len(successful_evaluations),
            "failed_evaluations": len(failed_evaluations),
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(dataset) if dataset else 0,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": "Google Gemini"
            }
        },
        "aggregate_metrics": aggregate_metrics,
        "detailed_results": results
    }
    
    logger.info(f"\n=== EVALUATION SUMMARY ===")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total Samples: {len(dataset)}")
    logger.info(f"Successful: {len(successful_evaluations)}")
    logger.info(f"Failed: {len(failed_evaluations)}")
    logger.info(f"Total Processing Time: {total_processing_time:.2f}s")
    logger.info(f"Average Processing Time: {total_processing_time / len(dataset):.2f}s per sample")
    
    if aggregate_metrics:
        logger.info(f"\n=== AGGREGATE METRICS ===")
        for metric, stats in aggregate_metrics.items():
            logger.info(f"{metric}:")
            logger.info(f"  Mean: {stats['mean']:.3f}")
            logger.info(f"  Min: {stats['min']:.3f}")
            logger.info(f"  Max: {stats['max']:.3f}")
            logger.info(f"  Count: {stats['count']}")
        
        # Overall performance assessment
        overall_scores = [stats['mean'] for stats in aggregate_metrics.values()]
        overall_score = sum(overall_scores) / len(overall_scores)
        logger.info(f"\nOverall Average Score: {overall_score:.3f}")
            
        performance_level = "Excellent" if overall_score >= 0.8 else \
                          "Good" if overall_score >= 0.6 else \
                          "Fair" if overall_score >= 0.4 else "Poor"
        logger.info(f"Performance Level: {performance_level}")
    
    return evaluation_summary


def save_evaluation_results(results: Dict[str, Any], output_file: str):
    """Save evaluation results to JSON file with comprehensive logging."""
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Detailed evaluation results saved to: {output_path}")
        logger.info(f"💾 File size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Log key statistics for human review
        metadata = results.get("metadata", {})
        logger.info(f"\n📊 HUMAN REVIEW SUMMARY:")
        logger.info(f"   • Timestamp: {metadata.get('evaluation_timestamp', 'N/A')}")
        logger.info(f"   • Total Samples: {metadata.get('total_samples', 0)}")
        logger.info(f"   • Success Rate: {metadata.get('successful_evaluations', 0)}/{metadata.get('total_samples', 0)}")
        logger.info(f"   • Processing Time: {metadata.get('total_processing_time', 0):.1f}s")
        logger.info(f"   • Collection: {metadata.get('collection_name', 'N/A')}")
        logger.info(f"   • Model: {metadata.get('evaluator_model', 'N/A')}")
        
        # Log metric overview
        aggregate_metrics = results.get("aggregate_metrics", {})
        if aggregate_metrics:
            logger.info(f"\n🎯 METRICS OVERVIEW:")
            for metric, stats in aggregate_metrics.items():
                logger.info(f"   • {metric}: {stats['mean']:.3f} (range: {stats['min']:.3f}-{stats['max']:.3f})")
        
        logger.info(f"\n🔍 For detailed analysis, review the JSON file with:")
        logger.info(f"   • Generated answers vs ground truth")
        logger.info(f"   • Retrieved contexts and sources")
        logger.info(f"   • Individual RAGAS scores")
        logger.info(f"   • Processing times per sample")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def save_evaluation_results_csv(results: Dict[str, Any], output_file: str):
    """Save evaluation results to CSV file with same structure as JSON."""
    try:
        # Create CSV filename by replacing .json with .csv
        csv_file = output_file.replace('.json', '.csv')
        csv_path = Path(csv_file)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare CSV data
        csv_data = []
        evaluation_results = results.get("detailed_results", [])
        
        for sample in evaluation_results:
            # Flatten the data structure for CSV - clean version without verbose contexts
            row = {
                'sample_id': sample.get('sample_id', ''),
                'question': sample.get('question', ''),
                'generated_answer': sample.get('generated_answer', ''),
                'ground_truth': sample.get('ground_truth', ''),
                'processing_time': sample.get('processing_time', ''),
                'contexts_count': sample.get('contexts_count', ''),
            }
            
            # Add RAGAS metrics
            ragas_metrics = sample.get('ragas_metrics', {})
            row.update({
                'faithfulness': ragas_metrics.get('faithfulness', ''),
                'answer_relevancy': ragas_metrics.get('answer_relevancy', ''),
                'context_precision': ragas_metrics.get('context_precision', ''),
                'context_recall': ragas_metrics.get('context_recall', ''),
            })
            
            csv_data.append(row)
        
        # Write CSV file
        if csv_data:
            fieldnames = csv_data[0].keys()
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"📊 CSV evaluation results saved to: {csv_path}")
            logger.info(f"💾 CSV file size: {csv_path.stat().st_size / 1024:.1f} KB")
            logger.info(f"📝 CSV contains {len(csv_data)} evaluation samples")
        else:
            logger.warning("No evaluation data to save to CSV")
            
    except Exception as e:
        logger.error(f"Failed to save CSV results: {e}")


if __name__ == "__main__":
    try:
        # Configuration
        ground_truth_file = "data/dataset/sample_single.json"  # Use existing ground truth
        collection_name = "sensors_no_metadata"
        
        # Generate timestamped output file
        timestamp = int(time.time())
        output_file = f"data/evaluation/rag_evaluation_{collection_name}_{timestamp}.json"
        
        logger.info("🚀 Starting RAG System Evaluation (No Metadata)")
        logger.info(f"📖 Ground Truth File: {ground_truth_file}")
        logger.info(f"🗂️  Collection: {collection_name}")
        logger.info(f"💾 Output File: {output_file}")
        
        # Load ground truth dataset
        dataset = load_ground_truth_dataset(ground_truth_file)
        
        # Run evaluation
        evaluation_results = evaluate_no_metadata_rag(dataset, collection_name)
        
        # Save results
        save_evaluation_results(evaluation_results, output_file)
        save_evaluation_results_csv(evaluation_results, output_file)
        
        logger.info("✅ RAG evaluation completed successfully!")
        logger.info(f"📂 Results available at: {output_file}")
        logger.info(f"📊 CSV results available at: {output_file.replace('.json', '.csv')}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
