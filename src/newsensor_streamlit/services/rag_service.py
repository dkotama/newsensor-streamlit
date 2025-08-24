from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from loguru import logger

from newsensor_streamlit.config import settings, ResponseLanguage

if TYPE_CHECKING:
    from typing import List
    from langchain.schema import Document


class RagService:
    """Service for RAG (Retrieval-Augmented Generation) operations with conversation memory."""
    
    def __init__(self) -> None:
        logger.info("Initializing RAG service with conversation memory")
        self.llm = self._create_llm()
        self.prompt_templates = self._create_prompt_templates()
        
        # Initialize conversation memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,  # Summarize when chat history exceeds this
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )
        
    def _create_llm(self):
        """Create LLM instance from settings with provider-specific configuration."""
        if settings.rag_provider_enum.value == "openai":
            return ChatOpenAI(
                model=settings.rag_chat_model,
                api_key=settings.openai_api_key,
                temperature=settings.rag_temperature
            )
        else:  # openrouter
            return ChatOpenAI(
                model=settings.rag_chat_model,
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=settings.rag_temperature
            )
    
    def _create_prompt_templates(self) -> dict[ResponseLanguage, ChatPromptTemplate]:
        """Create prompt templates for both languages with conversation context."""
        
        # Indonesian prompt (existing)
        indonesian_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Anda akan bertindak sebagai seorang insinyur elektronik ahli dengan persona sebagai penasihat teknis formal. Nada Anda harus profesional, presisi, dan objektif, serta menghindari bahasa percakapan. Tujuan utama Anda adalah menginterpretasikan datasheet komponen elektronik untuk menjawab pertanyaan pengguna.

            Ketika saya memberikan konteks dan pertanyaan spesifik, Anda harus mematuhi aturan berikut:

            Batasan Sumber: Dasarkan jawaban HANYA pada informasi yang terdapat dalam konteks yang diberikan. Jangan gunakan pengetahuan eksternal. Jika jawabannya tidak ada dalam konteks, nyatakan dengan jelas.

            Sumber Kutipan: Saat menjawab, Anda harus mengutip bagian, tabel, atau frasa kunci spesifik dari konteks yang diberikan sebagai sumber. Hal ini penting untuk verifikasi akurasi. Contoh: “Menurut tabel 'Electrical Characteristics', arus diam adalah...”

            Klarifikasi Ambiguitas: Jika saya memberikan konteks untuk beberapa komponen dan pertanyaan saya ambigu, Anda harus meminta klarifikasi sebelum menjawab.

            Ringkasan Multi-Konteks: Jika Anda diberikan banyak konteks dari berbagai model, dan pengguna bertanya misalnya “Apa rentang pengukuran sensor untuk kelembapan?”, serta dalam konteks ditemukan bahwa setiap model memiliki rentang berbeda, maka jawaban harus dipisahkan per model. Contoh: “Untuk model S15S, rentang pengukuran adalah 0–100% RH. Untuk model LS219, rentang pengukuran adalah 0–80% RH.”

            Model Interaksi: Ikuti format Tanya & Jawab yang ketat. Jangan memberikan saran atau informasi yang tidak diminta di luar apa yang dibutuhkan untuk menjawab pertanyaan. Namun, jika pertanyaan secara eksplisit meminta panduan atau saran (misalnya “Bisakah Anda menyarankan tegangan operasi yang sesuai dari rentang ini?”), Anda boleh memberikannya, dengan catatan hanya didasarkan pada konteks yang diberikan.

            PENTING: Anda memiliki akses ke riwayat percakapan sebelumnya. Gunakan informasi dari percakapan sebelumnya untuk memberikan jawaban yang lebih tepat dan kontekstual. Jika pertanyaan merujuk pada diskusi sebelumnya (seperti "itu", "sensor yang tadi", "model tersebut"), rujuk ke riwayat percakapan untuk memahami konteksnya.

            Context dari Datasheet: {context}

            Instruksi: Gunakan bahasa Indonesia dan berikan jawaban yang terfokus berdasarkan konteks datasheet dan riwayat percakapan. Jika jawabannya tidak ada dalam konteks, katakan dengan jelas.

           """),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        
        # English prompt (translated)
        english_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You will act as an expert electronics engineer with a persona as a formal technical advisor. Your tone should be professional, precise, and objective, avoiding conversational language. Your primary goal is to interpret electronic component datasheets to answer user questions.

            When I provide context and specific questions, you must follow these rules:

            Source Constraints: Base your answers ONLY on information contained in the provided context. Do not use external knowledge. If the answer is not in the context, state this clearly.

            Source Citation: When answering, you must cite specific sections, tables, or key phrases from the provided context as sources. This is important for accuracy verification. Example: "According to the 'Electrical Characteristics' table, the quiescent current is..."

            Ambiguity Clarification: If I provide context for multiple components and my question is ambiguous, you should ask for clarification before answering.

            Multi-Context Summary: If you are given multiple contexts from various models, and the user asks for example "What is the sensor measurement range for humidity?", and the context shows that each model has different ranges, then the answer must be separated by model. Example: "For model S15S, the measurement range is 0–100% RH. For model LS219, the measurement range is 0–80% RH."

            Interaction Model: Follow a strict Question & Answer format. Do not provide advice or information that is not requested beyond what is needed to answer the question. However, if the question explicitly requests guidance or advice (e.g., "Can you suggest an appropriate operating voltage from this range?"), you may provide it, with the caveat that it is only based on the provided context.

            IMPORTANT: You have access to previous conversation history. Use information from previous conversations to provide more accurate and contextual answers. If questions refer to previous discussions (like "that", "the sensor mentioned earlier", "that model"), refer to the conversation history to understand the context.

            Datasheet Context: {context}

            Instructions: Use English and provide answers focused on the datasheet context and conversation history. If the answer is not in the context, state this clearly.

           """),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        
        return {
            ResponseLanguage.INDONESIAN: indonesian_prompt,
            ResponseLanguage.ENGLISH: english_prompt
        }
    
    def _get_current_prompt_template(self) -> ChatPromptTemplate:
        """Get the current prompt template based on settings."""
        return self.prompt_templates[settings.response_language_enum]

    def generate_answer(self, question: str, context: List[Document], chat_history: Optional[List[BaseMessage]] = None) -> dict[str, str]:
        """Generate an answer using the provided context and chat history."""
        try:
            if not context:
                # Return language-appropriate no context message
                if settings.response_language_enum == ResponseLanguage.INDONESIAN:
                    no_context_msg = "Saya tidak memiliki konteks yang relevan dari datasheet untuk menjawab pertanyaan ini. Silakan upload dokumen terlebih dahulu."
                else:
                    no_context_msg = "I don't have any relevant context from the datasheet to answer this question. Please upload a document first."
                
                return {
                    "answer": no_context_msg,
                    "context": "",
                    "sources": [],
                    "chat_history_used": False
                }
            
            combined_context = "\n\n---\n\n".join([
                f"Excerpt {i+1}: {doc.page_content}"
                for i, doc in enumerate(context)
            ])
            
            # Get or use provided chat history
            if chat_history is None:
                chat_history = self.memory.chat_memory.messages
            
            # Log chat history usage
            chat_history_length = len(chat_history) if chat_history else 0
            logger.info(f"Generating answer for: {question[:50]}... (Language: {settings.response_language_enum.value}, Model: {settings.rag_chat_model}, Chat History: {chat_history_length} messages)")
            
            # Get current prompt template
            prompt_template = self._get_current_prompt_template()
            
            # Format the prompt with context and chat history
            formatted_prompt = prompt_template.format_messages(
                context=combined_context,
                chat_history=chat_history or [],
                question=question
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Save to memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            
            return {
                "answer": answer,
                "context": combined_context,
                "sources": [doc.metadata.get("source", "datasheet extract") for doc in context],
                "chat_history_used": chat_history_length > 0,
                "chat_history_length": chat_history_length
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            
            # Return language-appropriate error message
            if settings.response_language_enum == ResponseLanguage.INDONESIAN:
                error_msg = f"Terjadi kesalahan saat menghasilkan jawaban: {str(e)}"
            else:
                error_msg = f"Error generating answer: {str(e)}"
            
            return {
                "answer": error_msg,
                "context": combined_context if 'combined_context' in locals() else "",
                "sources": [],
                "chat_history_used": False,
                "chat_history_length": 0
            }
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history."""
        return self.memory.chat_memory.messages
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")
    
    def save_conversation_to_file(self, filename: str) -> None:
        """Save conversation history to file."""
        import json
        from pathlib import Path
        
        conversations_dir = settings.conversations_dir
        conversations_dir.mkdir(exist_ok=True)
        
        history = []
        for message in self.memory.chat_memory.messages:
            history.append({
                "type": "human" if isinstance(message, HumanMessage) else "ai",
                "content": message.content,
                "timestamp": str(message.additional_kwargs.get("timestamp", ""))
            })
        
        filepath = conversations_dir / f"{filename}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation_from_file(self, filename: str) -> bool:
        """Load conversation history from file."""
        import json
        from pathlib import Path
        
        filepath = settings.conversations_dir / f"{filename}.json"
        
        if not filepath.exists():
            logger.warning(f"Conversation file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            self.memory.clear()
            
            for msg in history:
                if msg["type"] == "human":
                    self.memory.chat_memory.add_user_message(msg["content"])
                else:
                    self.memory.chat_memory.add_ai_message(msg["content"])
            
            logger.info(f"Conversation loaded from {filepath} ({len(history)} messages)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return False
            
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