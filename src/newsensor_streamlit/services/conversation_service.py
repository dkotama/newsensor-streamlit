from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from newsensor_streamlit.config import settings
from newsensor_streamlit.services.conversation_mongodb_service import ConversationMongoDBService


class ConversationService:
    """Service for managing conversation history and RAGAS metrics."""

    def __init__(self) -> None:
        # Keep conversations_dir for backward compatibility (exports, etc.)
        self.conversations_dir = Path(settings.conversations_dir)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Use MongoDB service for actual storage
        self.mongodb_service = ConversationMongoDBService()
        logger.info("ConversationService initialized with MongoDB backend")

    def create_conversation(self, doc_id: str, document_name: str) -> str:
        """Create a new conversation session."""
        conversation_id = str(uuid.uuid4())
        
        conversation_data = {
            "conversation_id": conversation_id,
            "doc_id": doc_id,
            "document_name": document_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "ragas_summary": {
                "total_questions": 0,
                "avg_faithfulness": 0.0,
                "avg_answer_relevancy": 0.0,
                "avg_context_precision": 0.0,
                "avg_context_recall": 0.0
            }
        }
        
        # Save to MongoDB
        success = self.mongodb_service.save_conversation(conversation_data)
        if success:
            logger.info(f"Created new conversation: {conversation_id} for document: {document_name}")
            return conversation_id
        else:
            raise Exception(f"Failed to create conversation in MongoDB")
    
    def add_message(
        self, 
        conversation_id: str, 
        question: str, 
        answer: str, 
        sources: List[str], 
        metrics: Dict[str, float],
        context: str = "",
        ragas_metrics: Dict[str, float] = None,
        retrieval_metadata: Dict[str, Any] = None
    ) -> None:
        """Add a message to the conversation with enhanced metrics and metadata."""
        conversation = self.mongodb_service.load_conversation(conversation_id)
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return
            
        message = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "context": context,
            "sources": sources,
            "ragas_metrics": ragas_metrics or {},
            "legacy_metrics": metrics,  # Keep old metrics for backward compatibility
            "retrieval_metadata": retrieval_metadata or {}
        }
        
        conversation["messages"].append(message)
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Update RAGAS summary with new metrics if available
        metrics_to_update = ragas_metrics if ragas_metrics else metrics
        self._update_ragas_summary(conversation, metrics_to_update)
        
        # Save back to MongoDB
        success = self.mongodb_service.save_conversation(conversation)
        if success:
            logger.info(f"Added message to conversation {conversation_id}")
        else:
            logger.error(f"Failed to save message to conversation {conversation_id}")
        
        # Log re-ranking usage
        if retrieval_metadata and retrieval_metadata.get("reranking_enabled"):
            logger.info(
                f"Re-ranking used: {retrieval_metadata.get('initial_vector_search_count')} → "
                f"{retrieval_metadata.get('final_chunk_count')} chunks, "
                f"best score: {retrieval_metadata.get('best_relevance_score', 0):.3f}"
            )
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any] | None:
        """Retrieve a conversation by ID."""
        return self.mongodb_service.load_conversation(conversation_id)
        
    def list_conversations(self, doc_id: str = None) -> List[Dict[str, Any]]:
        """List all conversations, optionally filtered by document ID."""
        conversations = self.mongodb_service.list_conversations(doc_id)
        
        # Add message_count for backward compatibility with UI
        for conv in conversations:
            if 'messages' in conv:
                conv['message_count'] = len(conv['messages'])
            else:
                # For summary view, we need to load the full conversation to count messages
                full_conv = self.mongodb_service.load_conversation(conv['conversation_id'])
                conv['message_count'] = len(full_conv.get('messages', [])) if full_conv else 0
        
        return sorted(conversations, key=lambda x: x.get("updated_at", ""), reverse=True)

    def export_conversation(self, conversation_id: str, format: str = "json") -> str:
        """Export conversation to different formats."""
        conversation = self.mongodb_service.load_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
            
        export_dir = self.conversations_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            export_path = export_dir / f"{conversation_id}_{timestamp}.json"
            with open(export_path, "w") as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
                
        elif format == "markdown":
            export_path = export_dir / f"{conversation_id}_{timestamp}.md"
            self._export_to_markdown(conversation, export_path)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Exported conversation to {export_path}")
        return str(export_path)
    
    def _load_conversation(self, conversation_id: str) -> Dict[str, Any] | None:
        """Load conversation from MongoDB (kept for backward compatibility)."""
        return self.mongodb_service.load_conversation(conversation_id)
            
    def _save_conversation(self, conversation_id: str, conversation_data: Dict[str, Any]) -> None:
        """Save conversation to MongoDB (kept for backward compatibility)."""
        success = self.mongodb_service.save_conversation(conversation_data)
        if not success:
            logger.error(f"Error saving conversation {conversation_id} to MongoDB")
            raise Exception(f"Failed to save conversation {conversation_id} to MongoDB")
            
    def _update_ragas_summary(self, conversation: Dict[str, Any], new_metrics: Dict[str, float]) -> None:
        """Update the running RAGAS summary statistics."""
        messages = conversation["messages"]
        total_messages = len(messages)
        
        if total_messages == 0:
            return
            
        # Calculate running averages
        summary = conversation["ragas_summary"]
        summary["total_questions"] = total_messages
        
        # Calculate averages across all messages
        metrics_sum = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0, 
            "context_precision": 0.0,
            "context_recall": 0.0
        }
        
        for message in messages:
            if "ragas_metrics" in message:
                for key in metrics_sum:
                    metrics_sum[key] += message["ragas_metrics"].get(key, 0.0)
                    
        for key in metrics_sum:
            avg_key = f"avg_{key}"
            summary[avg_key] = metrics_sum[key] / total_messages
            
    def _export_to_markdown(self, conversation: Dict[str, Any], export_path: Path) -> None:
        """Export conversation to markdown format."""
        with open(export_path, "w") as f:
            f.write(f"# Conversation Report\n\n")
            f.write(f"**Document:** {conversation['document_name']}\n")
            f.write(f"**Conversation ID:** {conversation['conversation_id']}\n") 
            f.write(f"**Created:** {conversation['created_at']}\n")
            f.write(f"**Updated:** {conversation['updated_at']}\n\n")
            
            # RAGAS Summary
            f.write("## RAGAS Metrics Summary\n\n")
            summary = conversation["ragas_summary"]
            f.write(f"- **Total Questions:** {summary['total_questions']}\n")
            f.write(f"- **Average Faithfulness:** {summary['avg_faithfulness']:.3f}\n")
            f.write(f"- **Average Answer Relevancy:** {summary['avg_answer_relevancy']:.3f}\n")
            f.write(f"- **Average Context Precision:** {summary['avg_context_precision']:.3f}\n")
            f.write(f"- **Average Context Recall:** {summary['avg_context_recall']:.3f}\n\n")
            
            # Messages
            f.write("## Conversation\n\n")
            for i, message in enumerate(conversation["messages"], 1):
                f.write(f"### Question {i}\n")
                f.write(f"**Time:** {message['timestamp']}\n\n")
                f.write(f"**Q:** {message['question']}\n\n")
                f.write(f"**A:** {message['answer']}\n\n")
                
                if message.get("sources"):
                    f.write(f"**Sources:** {', '.join(message['sources'])}\n\n")
                    
                # RAGAS metrics for this message
                f.write("**RAGAS Metrics:**\n")
                metrics = message.get("ragas_metrics", {})
                for metric, value in metrics.items():
                    f.write(f"- {metric.replace('_', ' ').title()}: {value:.3f}\n")
                f.write("\n---\n\n")
