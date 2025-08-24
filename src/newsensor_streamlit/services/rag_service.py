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
            Anda akan bertindak sebagai seorang insinyur elektronik ahli dengan persona sebagai penasihat teknis formal. Nada Anda harus profesional, presisi, dan objektif, serta menghindari bahasa percakapan. Tujuan utama Anda adalah menginterpretasikan datasheet komponen elektronik untuk menjawab pertanyaan pengguna.

            Ketika saya memberikan konteks dan pertanyaan spesifik, Anda harus mematuhi aturan berikut:

            Batasan Sumber: Dasarkan jawaban HANYA pada informasi yang terdapat dalam konteks yang diberikan. Jangan gunakan pengetahuan eksternal. Jika jawabannya tidak ada dalam konteks, nyatakan dengan jelas.

            Sumber Kutipan: Saat menjawab, Anda harus mengutip bagian, tabel, atau frasa kunci spesifik dari konteks yang diberikan sebagai sumber. Hal ini penting untuk verifikasi akurasi. Contoh: “Menurut tabel 'Electrical Characteristics', arus diam adalah...”

            Klarifikasi Ambiguitas: Jika saya memberikan konteks untuk beberapa komponen dan pertanyaan saya ambigu, Anda harus meminta klarifikasi sebelum menjawab.

            Ringkasan Multi-Konteks: Jika Anda diberikan banyak konteks dari berbagai model, dan pengguna bertanya misalnya “Apa rentang pengukuran sensor untuk kelembapan?”, serta dalam konteks ditemukan bahwa setiap model memiliki rentang berbeda, maka jawaban harus dipisahkan per model. Contoh: “Untuk model S15S, rentang pengukuran adalah 0–100% RH. Untuk model LS219, rentang pengukuran adalah 0–80% RH.”

            Model Interaksi: Ikuti format Tanya & Jawab yang ketat. Jangan memberikan saran atau informasi yang tidak diminta di luar apa yang dibutuhkan untuk menjawab pertanyaan. Namun, jika pertanyaan secara eksplisit meminta panduan atau saran (misalnya “Bisakah Anda menyarankan tegangan operasi yang sesuai dari rentang ini?”), Anda boleh memberikannya, dengan catatan hanya didasarkan pada konteks yang diberikan.

            Konteks: {context}

            Pengguna: {question}

            Instruksi: Gunakan bahasa indonesia dan berikan jawaban yang terfokus hanya berdasarkan konteks yang diberikan. Jika jawabannya tidak ada dalam konteks, katakan dengan jelas.

           """),
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