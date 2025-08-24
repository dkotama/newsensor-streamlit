from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional

from loguru import logger
from newsensor_streamlit.config import settings

class RagasEvaluator:
    """Real RAGAS evaluation service using GPT-4o for numerical metrics."""
    
    def __init__(self):
        self.evaluator_model = settings.ragas_evaluator_model  # gpt-4o
        self.test_dataset_path = settings.ragas_test_dataset_path
        self.fallback_silent = settings.ragas_fallback_silent
        self._evaluator_llm = None
        self._embeddings = None
        
    def _get_evaluator_llm(self):
        """Lazy load the evaluator LLM with provider-specific configuration."""
        if self._evaluator_llm is None:
            try:
                from langchain_openai import ChatOpenAI
                
                if settings.rag_provider_enum.value == "openai":
                    self._evaluator_llm = ChatOpenAI(
                        model=self.evaluator_model,
                        temperature=0.0,
                        api_key=settings.openai_api_key
                    )
                else:  # openrouter
                    self._evaluator_llm = ChatOpenAI(
                        model=self.evaluator_model,
                        temperature=0.0,
                        api_key=settings.openrouter_api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                
                logger.info(f"Initialized RAGAS evaluator with {self.evaluator_model} (provider: {settings.rag_provider_enum.value})")
            except Exception as e:
                logger.error(f"Failed to initialize RAGAS evaluator: {e}")
                raise
        return self._evaluator_llm
    
    def _get_embeddings(self):
        """Lazy load embeddings for RAGAS with provider-specific configuration."""
        if self._embeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                
                if settings.rag_provider_enum.value == "openai":
                    self._embeddings = OpenAIEmbeddings(
                        api_key=settings.openai_api_key
                    )
                else:  # openrouter - Note: OpenRouter may not support embeddings API
                    # For now, fallback to OpenAI embeddings even with OpenRouter provider
                    # This is because OpenRouter might not have embeddings endpoint
                    if settings.openai_api_key:
                        logger.warning("Using OpenAI embeddings as fallback for RAGAS (OpenRouter may not support embeddings)")
                        self._embeddings = OpenAIEmbeddings(
                            api_key=settings.openai_api_key
                        )
                    else:
                        raise ValueError("OpenAI API key required for embeddings when using OpenRouter")
                
                logger.info("Initialized RAGAS embeddings")
            except Exception as e:
                logger.error(f"Failed to initialize RAGAS embeddings: {e}")
                raise
        return self._embeddings
    
    def is_available(self) -> bool:
        """Check if RAGAS evaluation is available."""
        try:
            import ragas
            import datasets
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            
            # Check if the appropriate API key is configured based on provider
            if settings.rag_provider_enum.value == "openai":
                if not settings.openai_api_key:
                    logger.warning("OpenAI API key not configured for RAGAS evaluation")
                    return False
            else:  # openrouter
                if not settings.openrouter_api_key:
                    logger.warning("OpenRouter API key not configured for RAGAS evaluation")
                    return False
                # Also need OpenAI key for embeddings fallback
                if not settings.openai_api_key:
                    logger.warning("OpenAI API key required for embeddings when using OpenRouter")
                    return False
                
            return True
            
        except ImportError as e:
            if not self.fallback_silent:
                logger.warning(f"RAGAS dependencies not available: {e}")
            return False
    
    def evaluate_answer(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str], 
        ground_truth: str = None
    ) -> Dict[str, float]:
        """Evaluate answer using real RAGAS metrics."""
        
        if not self.is_available():
            if not self.fallback_silent:
                logger.warning("RAGAS not available, returning empty metrics")
            return {}
        
        try:
            # Import RAGAS components
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            from datasets import Dataset
            
            # Prepare evaluator components
            evaluator_llm = self._get_evaluator_llm()
            embeddings = self._get_embeddings()
            
            # Configure metrics with our LLM and embeddings
            faithfulness.llm = evaluator_llm
            answer_relevancy.llm = evaluator_llm
            answer_relevancy.embeddings = embeddings
            context_precision.llm = evaluator_llm
            context_recall.llm = evaluator_llm
            context_recall.embeddings = embeddings
            
            # Prepare data for RAGAS evaluation
            # Ensure contexts is a list of lists (required format)
            if isinstance(contexts, list) and len(contexts) > 0:
                if isinstance(contexts[0], str):
                    # Convert list of strings to list of lists
                    contexts_formatted = [contexts]
                else:
                    contexts_formatted = contexts
            else:
                contexts_formatted = [[]]
            
            # Create dataset
            data = Dataset.from_dict({
                'question': [question],
                'answer': [answer],
                'contexts': contexts_formatted,
                'ground_truth': [ground_truth] if ground_truth else [answer]
            })
            
            logger.info(f"Evaluating with RAGAS: Q='{question[:50]}...', A='{answer[:50]}...'")
            
            # Run evaluation
            start_time = time.time()
            result = evaluate(
                data,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                raise_exceptions=False  # Don't crash on errors
            )
            evaluation_time = time.time() - start_time
            
            # Extract scores - handle both Series and scalar values
            def extract_score(value):
                """Safely extract numeric score from RAGAS result."""
                if hasattr(value, 'iloc'):
                    # It's a pandas Series
                    return float(value.iloc[0]) if len(value) > 0 else 0.0
                else:
                    # It's already a scalar
                    return float(value)
            
            metrics = {
                "faithfulness": round(extract_score(result["faithfulness"]), 3),
                "answer_relevancy": round(extract_score(result["answer_relevancy"]), 3),
                "context_precision": round(extract_score(result["context_precision"]), 3),
                "context_recall": round(extract_score(result["context_recall"]), 3)
            }
            
            logger.info(f"RAGAS evaluation completed in {evaluation_time:.2f}s: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            if not self.fallback_silent:
                logger.warning("Returning empty RAGAS metrics due to evaluation failure")
            return {}
    
    def get_ground_truth(self, question: str) -> Optional[str]:
        """Get ground truth for a question from test dataset (if available)."""
        try:
            import pandas as pd
            df = pd.read_csv(self.test_dataset_path)
            
            # Simple matching - in practice you'd want fuzzy matching
            matches = df[df['question'].str.contains(question[:20], case=False, na=False)]
            if not matches.empty:
                return matches.iloc[0]['ground_truth']
                
        except Exception as e:
            logger.debug(f"Could not load ground truth: {e}")
            
        return None
    
    def batch_evaluate(self, evaluation_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Evaluate multiple questions at once."""
        results = []
        
        for item in evaluation_data:
            result = self.evaluate_answer(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth")
            )
            results.append(result)
        
        return results

