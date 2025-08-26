#!/usr/bin/env python3
"""Manual evaluation helper - compare generated answers with ground truth."""

import json
import sys
from difflib import SequenceMatcher

def similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def manual_evaluate(results_file):
    """Manually evaluate the RAG results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("📊 MANUAL RAG EVALUATION REPORT")
    print("=" * 50)
    
    detailed_results = data.get("detailed_results", [])
    
    total_similarity = 0
    for i, result in enumerate(detailed_results, 1):
        generated = result.get("generated_answer", "")
        ground_truth = result.get("ground_truth", "")
        question = result.get("question", "")
        
        # Calculate semantic similarity
        sim_score = similarity(generated, ground_truth)
        total_similarity += sim_score
        
        print(f"\n🔍 Sample {i}")
        print(f"Question: {question[:80]}...")
        print(f"Generated: {generated[:100]}...")
        print(f"Ground Truth: {ground_truth[:100]}...")
        print(f"Similarity Score: {sim_score:.3f}")
        
        # Simple quality assessment
        if sim_score >= 0.8:
            quality = "Excellent ✅"
        elif sim_score >= 0.6:
            quality = "Good ✓"
        elif sim_score >= 0.4:
            quality = "Fair ⚠️"
        else:
            quality = "Poor ❌"
        
        print(f"Quality Assessment: {quality}")
        
        # Check for key facts
        key_facts_match = 0
        if "5 years" in generated and "5 years" in ground_truth:
            key_facts_match += 1
        if "25°C" in generated and "25°C" in ground_truth:
            key_facts_match += 1
        if "15-minute" in generated and "15-minute" in ground_truth:
            key_facts_match += 1
            
        print(f"Key Facts Matched: {key_facts_match}/3")
    
    # Overall assessment
    avg_similarity = total_similarity / len(detailed_results) if detailed_results else 0
    
    print(f"\n📈 OVERALL RESULTS")
    print(f"Average Similarity: {avg_similarity:.3f}")
    print(f"Total Samples: {len(detailed_results)}")
    
    if avg_similarity >= 0.8:
        overall_quality = "Excellent - RAG system performing very well"
    elif avg_similarity >= 0.6:
        overall_quality = "Good - RAG system performing adequately" 
    elif avg_similarity >= 0.4:
        overall_quality = "Fair - RAG system needs improvement"
    else:
        overall_quality = "Poor - RAG system needs significant work"
    
    print(f"Overall Quality: {overall_quality}")
    
    # Performance metrics
    metadata = data.get("metadata", {})
    print(f"\n⚡ PERFORMANCE METRICS")
    print(f"Average Response Time: {metadata.get('average_processing_time', 0):.2f}s")
    print(f"Success Rate: {metadata.get('successful_evaluations', 0)}/{metadata.get('total_samples', 0)}")
    print(f"Collection: {metadata.get('collection_name', 'N/A')}")

if __name__ == "__main__":
    # Use the latest results file
    results_file = "data/evaluation/rag_evaluation_sensors_no_metadata_1753727847.json"
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    try:
        manual_evaluate(results_file)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        print("Please run the evaluation script first.")
    except Exception as e:
        print(f"Error: {e}")
