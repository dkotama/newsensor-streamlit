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
        """Evaluate answer using real RAGAS metrics with new API."""
        
        if not self.is_available():
            if not self.fallback_silent:
                logger.warning("RAGAS not available, returning empty metrics")
            return {}
        
        try:
            # Set OpenAI API key in environment for RAGAS internal use
            import os
            if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            
            # Import RAGAS components - new API
            from ragas import evaluate, SingleTurnSample, EvaluationDataset
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
            
            # Prepare evaluator LLM 
            evaluator_llm = self._get_evaluator_llm()
            
            # Prepare contexts - ensure it's a list of strings
            if isinstance(contexts, list):
                context_list = contexts
            else:
                # If contexts is a single string, split it or wrap it in a list
                context_list = [contexts] if contexts else []
            
            # Create sample data as dictionary (EvaluationDataset.from_list expects dicts)
            sample_data = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": context_list,  # Must be a list
                "reference": ground_truth if ground_truth else answer
            }
            
            # Create evaluation dataset from list of dicts (not pre-converted samples)
            dataset = EvaluationDataset.from_list([sample_data])
            
            # Define metrics to evaluate
            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall()
            ]
            
            logger.info(f"Evaluating with RAGAS: Q='{question[:50]}...', A='{answer[:50]}...'")
            
            # Run evaluation
            start_time = time.time()
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm
            )
            evaluation_time = time.time() - start_time
            
            # Convert to pandas DataFrame to extract metrics
            df = result.to_pandas()
            
            # Extract scores from the first (and only) row
            extracted_metrics = {
                "faithfulness": round(float(df.iloc[0]['faithfulness']), 3),
                "answer_relevancy": round(float(df.iloc[0]['answer_relevancy']), 3),
                "context_precision": round(float(df.iloc[0]['context_precision']), 3),
                "context_recall": round(float(df.iloc[0]['context_recall']), 3)
            }
            
            logger.info(f"RAGAS evaluation completed in {evaluation_time:.2f}s: {extracted_metrics}")
            return extracted_metrics
            
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

