from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from loguru import logger

from newsensor_streamlit.config import settings

if TYPE_CHECKING:
    from typing import List
    from langchain.schema import Document


class RagService:
    """Service for RAG (Retrieval-Augmented Generation) operations."""
    
    def __init__(self) -> None:
        logger.info("Initializing RAG service")
        self.llm = self._create_llm()
        self.prompt_template = self._create_prompt_template()
        
    def _create_llm(self):
        """Create LLM instance from settings."""
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:  # type: ignore
        """Create optimized prompt template for electronics questions."""
        
        # Main prompt for answering
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You will act as an expert electronics engineer with the persona of a formal technical advisor. Your tone should be professional, precise, and objective, avoiding conversational language. Your primary goal is to interpret technical datasheets for electronic components to answer user questions.

            When I provide you with context and a specific question, you must adhere to the following rules:

            1. Source Limitation: Base your answer ONLY on the information present in the provided context. Do not use any external knowledge. If the answer is not in the context, state that clearly.
            2. Source Citation: When answering, you must cite the specific section, table, or key phrase from the provided context where you found the information. This is critical for verifying your accuracy. For example: "According to the 'Electrical Characteristics' table, the quiescent current is..."
            3. Ambiguity Clarification: If I provide context for multiple components and my question is ambiguous, you must ask for clarification before answering. 
            4. Many Context summarizer : If you given many context from many models, and user ask as example "What is the sensor's measuring range for humidity?", you found from context that many models may have different ranges, if the answer exist you should separate the answer by model, like "For the S15S model, the measuring range is 0-100% RH. For the LS219 model, it is 0-80% RH."
            5. Interaction Model: Adhere to a strict Question & Answer format. Do not offer unsolicited advice or information beyond what is required to answer the question. However, if a question explicitly asks for guidance or a suggestion (e.g., "Can you suggest a suitable operating voltage from this range?"), you may provide it, ensuring it is derived exclusively from the provided context.

            Context: {context}

            User: {question}

            Provide a focused answer based ONLY on the context provided. If the answer isn't in the context, say so."""),
        ])
        
        return prompt

    def generate_answer(self, question: str, context: List[Document]) -> dict[str, str]:
        """Generate an answer using the provided context."""
        try:
            if not context:
                return {
                    "answer": "I don't have any relevant context from the datasheet to answer this question. Please upload a document first.",
                    "context": "",
                    "sources": []
                }
            
            combined_context = "\n\n---\n\n".join([
                f"Excerpt {i+1}: {doc.page_content}"
                for i, doc in enumerate(context)
            ])
            
            logger.info(f"Generating answer for: {question[:50]}...")
            
            # Create and run chain
            conversation = [
                ("system", f"Context:\n{combined_context}"),
                ("human", question)
            ]
            response = self.llm.invoke(conversation)
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer,
                "context": combined_context,
                "sources": [doc.metadata.get("source", "datasheet extract") for doc in context]
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context": combined_context if 'combined_context' in locals() else "",
                "sources": []
            }
            
    def evaluate_quality(self, question: str, answer: str, context: List[Document], reference_answer: str = None) -> dict[str, float]:
        """Evaluate answer quality using basic heuristics (RAGAS placeholder)."""
        try:
            # Simplified evaluation for MVP
            context_len = len(context)
            answer_len = len(answer)
            
            # Basic scoring heuristics
            relevance = min(answer_len / 500, 1.0)  # Encourage detailed answers
            faithfulness = 0.85  # Assume good faith
            precision = 0.82 if context_len < 3 else 0.88  # Prefer smaller context
            recall = 0.80 + (min(context_len / 5, 1.0) * 0.15)  # Reward more context
            
            return {
                "faithfulness": round(faithfulness, 2),
                "answer_relevancy": round(relevance, 2),
                "context_precision": round(precision, 2),
                "context_recall": round(recall, 2)
            }
        except Exception as e:
            logger.error(f"Error evaluating quality: {e}")
            return {metric: 0.0 for metric in settings.ragas_metrics}