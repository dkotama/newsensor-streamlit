"""
Experiment service for comparing retrieval methods and evaluating re-ranking performance.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from ..config import settings
from ..core.chat_engine import ChatEngine

logger = logging.getLogger(__name__)


class RerankingExperimentService:
    """Service for running A/B experiments comparing standard vs re-ranking retrieval."""
    
    def __init__(self, chat_engine: ChatEngine):
        self.chat_engine = chat_engine
        self.results_dir = Path("data/experiments")
        self.results_dir.mkdir(exist_ok=True)
        
    async def run_comparison_experiment(
        self, 
        doc_id: str,
        questions: List[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run A/B comparison between standard and re-ranking retrieval.
        
        Args:
            doc_id: Document ID to test against
            questions: List of questions to test (uses test dataset if None)
            save_results: Whether to save results to CSV
            
        Returns:
            Dictionary with experiment results
        """
        if questions is None:
            questions = self._load_test_questions()
        
        if not questions:
            logger.warning("No questions available for experiment")
            return {}
        
        logger.info(f"Starting comparison experiment with {len(questions)} questions")
        
        results = []
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, question in enumerate(questions):
            logger.info(f"Testing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Test standard retrieval
                start_time = time.time()
                standard_response = self.chat_engine.ask_question(
                    question=question,
                    doc_id=doc_id,
                    use_reranking=False
                )
                standard_time = time.time() - start_time
                
                # Test re-ranking retrieval
                start_time = time.time()
                reranked_response = self.chat_engine.ask_question(
                    question=question,
                    doc_id=doc_id,
                    use_reranking=True
                )
                reranked_time = time.time() - start_time
                
                # Compile results
                result = {
                    "experiment_id": experiment_id,
                    "question_id": i + 1,
                    "question": question,
                    "doc_id": doc_id,
                    
                    # Standard retrieval results
                    "standard_answer": standard_response["answer"],
                    "standard_time": round(standard_time, 3),
                    "standard_sources": len(standard_response["sources"]),
                    "standard_legacy_metrics": standard_response.get("metrics", {}),
                    "standard_ragas_metrics": standard_response.get("ragas_metrics", {}),
                    
                    # Re-ranking results
                    "reranked_answer": reranked_response["answer"],
                    "reranked_time": round(reranked_time, 3),
                    "reranked_sources": len(reranked_response["sources"]),
                    "reranked_legacy_metrics": reranked_response.get("metrics", {}),
                    "reranked_ragas_metrics": reranked_response.get("ragas_metrics", {}),
                    
                    # Re-ranking metadata
                    "reranking_metadata": reranked_response.get("retrieval_metadata", {}),
                    
                    # Performance comparison
                    "time_difference": round(reranked_time - standard_time, 3),
                    "time_improvement": round((standard_time - reranked_time) / standard_time * 100, 1) if standard_time > 0 else 0,
                    
                    "timestamp": datetime.now().isoformat()
                }
                
                # Compare RAGAS metrics if available
                if standard_response.get("ragas_metrics") and reranked_response.get("ragas_metrics"):
                    result.update(self._compare_ragas_metrics(
                        standard_response["ragas_metrics"],
                        reranked_response["ragas_metrics"]
                    ))
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error testing question {i+1}: {e}")
                results.append({
                    "experiment_id": experiment_id,
                    "question_id": i + 1,
                    "question": question,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Save results if requested
        if save_results:
            self._save_results(experiment_id, results, summary)
        
        return {
            "experiment_id": experiment_id,
            "results": results,
            "summary": summary
        }
    
    def _load_test_questions(self) -> List[str]:
        """Load test questions from the dataset."""
        try:
            if self.chat_engine.ragas_evaluator:
                self.chat_engine.ragas_evaluator.load_test_dataset()
                if self.chat_engine.ragas_evaluator.test_dataset is not None:
                    return self.chat_engine.ragas_evaluator.test_dataset['question'].tolist()
        except Exception as e:
            logger.warning(f"Could not load test questions: {e}")
        
        # Fallback questions
        return [
            "What is the operating voltage range for this sensor?",
            "What is the I2C address of the sensor?",
            "How do I configure the sensor for continuous measurement mode?",
            "What are the pin connections needed for Arduino?",
            "What is the measurement accuracy of this sensor?"
        ]
    
    def _compare_ragas_metrics(
        self, 
        standard_metrics: Dict[str, float], 
        reranked_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare RAGAS metrics between standard and re-ranking."""
        comparison = {}
        
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric_name in standard_metrics and metric_name in reranked_metrics:
                standard_val = standard_metrics[metric_name]
                reranked_val = reranked_metrics[metric_name]
                
                improvement = reranked_val - standard_val
                improvement_pct = (improvement / standard_val * 100) if standard_val > 0 else 0
                
                comparison[f"{metric_name}_improvement"] = round(improvement, 3)
                comparison[f"{metric_name}_improvement_pct"] = round(improvement_pct, 1)
                comparison[f"{metric_name}_better"] = "reranked" if improvement > 0 else "standard"
        
        return comparison
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate experiment summary statistics."""
        if not results:
            return {}
        
        successful_results = [r for r in results if "error" not in r]
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        summary = {
            "total_questions": len(results),
            "successful_questions": len(successful_results),
            "error_rate": (len(results) - len(successful_results)) / len(results),
            
            # Performance metrics
            "avg_standard_time": sum(r.get("standard_time", 0) for r in successful_results) / len(successful_results),
            "avg_reranked_time": sum(r.get("reranked_time", 0) for r in successful_results) / len(successful_results),
            
            # Count improvements
            "reranking_faster_count": sum(1 for r in successful_results if r.get("time_improvement", 0) > 0),
            "reranking_slower_count": sum(1 for r in successful_results if r.get("time_improvement", 0) < 0),
        }
        
        # RAGAS improvements summary
        ragas_improvements = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            improvements = [r.get(f"{metric}_improvement", 0) for r in successful_results if f"{metric}_improvement" in r]
            if improvements:
                ragas_improvements[f"avg_{metric}_improvement"] = sum(improvements) / len(improvements)
                ragas_improvements[f"{metric}_better_count"] = sum(1 for imp in improvements if imp > 0)
        
        summary["ragas_summary"] = ragas_improvements
        return summary
    
    def _save_results(self, experiment_id: str, results: List[Dict], summary: Dict) -> None:
        """Save experiment results to files."""
        # Save detailed results to CSV
        results_file = self.results_dir / f"{experiment_id}_results.csv"
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            logger.info(f"Saved experiment results to {results_file}")
        
        # Save summary to JSON
        summary_file = self.results_dir / f"{experiment_id}_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved experiment summary to {summary_file}")
    
    def list_experiments(self) -> List[Dict[str, str]]:
        """List all completed experiments."""
        experiments = []
        
        for results_file in self.results_dir.glob("*_results.csv"):
            experiment_id = results_file.stem.replace("_results", "")
            summary_file = self.results_dir / f"{experiment_id}_summary.json"
            
            experiments.append({
                "experiment_id": experiment_id,
                "results_file": str(results_file),
                "summary_file": str(summary_file) if summary_file.exists() else None,
                "created_at": datetime.fromtimestamp(results_file.stat().st_mtime).isoformat()
            })
        
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)
