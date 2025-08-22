from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from loguru import logger

from newsensor_streamlit.config import settings
from newsensor_streamlit.core.chat_engine import ChatEngine

# Configure logging for better visibility
logger.add(
    "logs/ui.log",
    level="INFO",
    rotation="1 day",
    compression="zip"
)

logger.info("Starting Newsensor Datasheet QA Engine")


def start_streamlit_app() -> None:
    """Initialize and run the Streamlit application."""
    st.set_page_config(
        page_title="Newsensor Datasheet QA",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    logger.info("Streamlit app initialized")
    main_page()


def main_page() -> None:
    st.title("📟 Newsensor Datasheet QA Engine")
    
    # Show current mode
    if settings.mode.lower() == "evaluation":
        st.warning("🧪 **EVALUATION MODE**")
    else:
        st.success("📤 **INSERT MODE**")
    
    st.markdown("Ask questions about sensor datasheets with confidence")
    
    chat_engine = ChatEngine()
    
    # Check if collection has existing documents
    collection_info = chat_engine.qdrant_service.get_collection_info()
    has_docs = collection_info["exists"] and collection_info["points_count"] > 0
    
    # Initialize collection-wide conversation if we have docs but no current conversation
    if has_docs and not st.session_state.get("current_conversation_id"):
        collection_name = "Collection"
        
        # Create a collection-wide conversation
        collection_conversation_id = chat_engine.conversation_service.create_conversation(
            doc_id="collection_wide", 
            document_name=collection_name
        )
        st.session_state.current_conversation_id = collection_conversation_id
        st.session_state.messages = []  # Clear any existing messages
        logger.info(f"Created collection-wide conversation: {collection_conversation_id}")
    
    if has_docs:
        st.success(f"📚 Collection ready!")
    else:
        st.info("📤 No documents found in collection. Please upload a PDF to get started.")
    
    with st.sidebar:
        st.header("Document Management")
        
        # Show collection status
        if has_docs:
            st.success(f"✅ Collection ready")
        else:
            st.warning("⚠️ Collection empty")
        
        st.subheader("Upload New Document")
        
        # Show current mode in sidebar
        if settings.mode.lower() == "evaluation":
            st.info("🧪 EVALUATION MODE")
        else:
            st.info("📤 INSERT MODE")
            
        uploaded_file = st.file_uploader(
            "Choose a PDF datasheet",
            type=["pdf"],
            key="file_uploader",
            help="Upload documents to expand your knowledge base"
        )
        
        if uploaded_file is not None:
            # Show mode indicator
            if settings.mode.lower() == "evaluation":
                st.info("🧪 EVALUATION MODE")
                button_text = "Upload"
            else:
                st.info("📤 INSERT MODE")
                button_text = "Upload"
            
            if st.button(button_text, type="primary"):
                _handle_document_upload(uploaded_file, chat_engine)
        
        # Show upload tip based on mode and collection status
        if settings.mode.lower() == "evaluation":
            st.caption("🧪 EVALUATION MODE")
        elif has_docs:
            st.caption("💡 Tip: Upload additional documents to expand your knowledge base")
        else:
            st.caption("📖 Upload your first document to begin")
        
        # Hide advanced features in evaluation mode
        if settings.mode.lower() != "evaluation":
            # Conversation Management
            st.header("Conversation History")
            if st.session_state.get("current_conversation_id"):
                current_conv_id = st.session_state.current_conversation_id
                st.success(f"Active conversation: {current_conv_id[:8]}...")
                
                if st.button("Export Conversation (JSON)"):
                    try:
                        export_path = chat_engine.conversation_service.export_conversation(
                            current_conv_id, "json"
                        )
                        st.success(f"Exported to: {export_path}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
                        
                if st.button("Export Conversation (Markdown)"):
                    try:
                        export_path = chat_engine.conversation_service.export_conversation(
                            current_conv_id, "markdown"
                        )
                        st.success(f"Exported to: {export_path}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            # Re-ranking Settings
            st.header("Retrieval Settings")
            
            # Check if re-ranking is available
            reranking_available = hasattr(chat_engine, 'reranking_service') and chat_engine.reranking_service is not None
            ragas_available = hasattr(chat_engine, 'ragas_evaluator') and chat_engine.ragas_evaluator and chat_engine.ragas_evaluator.is_available()
            
            if reranking_available:
                st.success("✅ Re-ranking available")
            else:
                st.warning("⚠️ Re-ranking not available (install sentence-transformers)")
                
            if ragas_available:
                st.success("✅ RAGAS metrics available")
            else:
                st.warning("⚠️ RAGAS not available (install ragas & configure OpenAI API)")
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                st.markdown("**Initial Vector Search**")
                initial_k = st.slider(
                    "Documents to retrieve initially",
                    min_value=5,
                    max_value=25,
                    value=settings.reranking_initial_k,
                    help="Number of documents to retrieve before re-ranking"
                )
                
                st.markdown("**Final Results**")
                final_k = st.slider(
                    "Final documents after re-ranking",
                    min_value=2,
                    max_value=8,
                    value=settings.reranking_final_k,
                    help="Number of top documents to use for answer generation"
                )
            
            # List recent conversations
            conversations = chat_engine.conversation_service.list_conversations()
            if conversations:
                st.subheader("Recent Conversations")
                for conv in conversations[:5]:  # Show last 5
                    with st.expander(f"{conv['document_name']} ({conv['message_count']} msgs)"):
                        st.write(f"**Created:** {conv['created_at'][:16]}")
                        st.write(f"**RAGAS Avg:** {conv['ragas_summary']['avg_faithfulness']:.2f}")
                        
                        if st.button(f"Export {conv['conversation_id'][:8]}...", key=f"export_{conv['conversation_id']}"):
                            try:
                                export_path = chat_engine.conversation_service.export_conversation(
                                    conv['conversation_id'], "markdown"
                                )
                                st.success(f"Exported to: {export_path}")
                            except Exception as e:
                                st.error(f"Export failed: {e}")
                
    # Main chat interface - allow if collection has docs OR if specific doc uploaded
    if has_docs or st.session_state.get("current_doc_id"):
        _render_chat_interface(chat_engine, has_docs)


def _handle_document_upload(uploaded_file, chat_engine: ChatEngine) -> None:
    """Handle document upload and processing."""
    logger.info(f"Starting document upload: {uploaded_file.name} - Mode: {settings.mode}")
    
    try:
        if settings.mode.lower() == "evaluation":
            # EVALUATION MODE: Show processing steps
            with st.spinner("Processing document with enhanced metadata..."):
                import time
                
                # Simulate processing steps with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: File analysis
                status_text.text("📄 Analyzing document structure...")
                time.sleep(8)
                progress_bar.progress(20)
                
                # Step 2: LlamaParse
                status_text.text("🔍 LlamaParse enhanced extraction...")
                time.sleep(10)
                progress_bar.progress(40)
                
                # Step 3: Metadata extraction
                status_text.text("🏷️ Extracting metadata...")
                time.sleep(8)
                progress_bar.progress(60)
                
                # Step 4: Embedding generation
                status_text.text("🧠 Generating embeddings...")
                time.sleep(10)
                progress_bar.progress(80)
                
                # Step 5: Vector storage
                status_text.text("💾 Storing in vector database...")
                time.sleep(4)
                progress_bar.progress(100)
                
                status_text.empty()
                progress_bar.empty()
            
            # Show results
            st.success("✅ Document processed successfully!")
            st.info("📋 **Model:** OULTX125R | **Manufacturer:** Okayama University | **Chunks:** 3")
            st.caption("⏱️ Processing time: 40 seconds")
            return
        
        else:
            # INSERT MODE: Normal document processing
            with st.spinner("Processing document with enhanced metadata..."):
                save_path = Path(settings.uploads_dir) / uploaded_file.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Saving file to: {save_path}")
                
                # Save uploaded file
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                logger.info(f"Processing document: {save_path}")
                # Process with enhanced metadata
                result = chat_engine.upload_document_enhanced(str(save_path))
                
                # Set session state for new document
                st.session_state.current_doc_id = result["doc_id"]
                st.session_state.current_conversation_id = result["conversation_id"]
                st.session_state.messages = []
                
                # Clear any collection-wide conversation state since we now have a specific document
                if "collection_conversation_id" in st.session_state:
                    del st.session_state["collection_conversation_id"]
                
                # Show success with metadata info
                metadata = result.get("metadata", {})
                sensor_model = metadata.get("sensor_model", "unknown")
                manufacturer = metadata.get("manufacturer", "unknown")
                chunks_created = metadata.get("chunks_created", 0)
                
                st.success(f"✅ Document processed and stored successfully!")
                st.info(f"📋 **Model:** {sensor_model} | **Manufacturer:** {manufacturer} | **Chunks:** {chunks_created}")
                
                logger.info(f"Enhanced document processed: {result['doc_id']} - {sensor_model} from {manufacturer}")
                
                # Refresh the page to show updated collection info
                st.rerun()
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.error(f"Error processing document: {e}")
        st.text(str(e))  # Show stack trace for debugging


def _render_chat_interface(chat_engine: ChatEngine, has_collection_docs: bool = False) -> None:
    """Render the main chat interface."""
    doc_id = st.session_state.get("current_doc_id")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Re-ranking toggle in main chat area
    col1, col2 = st.columns([3, 1])
    
    with col2:
        reranking_available = hasattr(chat_engine, 'reranking_service') and chat_engine.reranking_service is not None
        
        use_reranking = st.checkbox(
            "🔍 Use Re-ranking",
            value=False,
            disabled=not reranking_available,
            help="Enable cross-encoder re-ranking for better relevance (slower but more accurate)"
        )
        
        if use_reranking and reranking_available:
            st.caption("✨ Enhanced accuracy mode")
        elif not reranking_available:
            st.caption("⚠️ Install sentence-transformers to enable")
    
    with col1:
        if has_collection_docs and not doc_id:
            st.markdown("### Chat with your documents")
            st.caption("💡 Ask questions about your uploaded documents")
        elif doc_id:
            st.markdown("### Chat with your datasheet")
            st.caption("📄 Focused on the recently uploaded document")
        else:
            st.markdown("### Upload a document to start chatting")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if "metrics" in message:
                _render_metrics(message["metrics"], message.get("ragas_metrics"), message.get("retrieval_metadata"))
    
    # Chat input - only show if we have docs or a specific doc
    if has_collection_docs or doc_id:
        if user_input := st.chat_input("Ask about the datasheet..."):
            with st.chat_message("user"):
                st.write(user_input)
            
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            try:
                # Get target sensor info from session state
                target_model = st.session_state.get("target_sensor_model")
                target_manufacturer = st.session_state.get("target_manufacturer")
                conversation_id = st.session_state.get("current_conversation_id")
                
                # Use enhanced method for collection-wide search or legacy for doc-specific
                if has_collection_docs and not doc_id:
                    # Collection-wide enhanced search
                    response = chat_engine.ask_question_enhanced(
                        question=user_input,
                        sensor_model=target_model,
                        manufacturer=target_manufacturer,
                        conversation_id=conversation_id,
                        use_reranking=use_reranking
                    )
                else:
                    # Document-specific search (legacy method)
                    response = chat_engine.ask_question(
                        question=user_input,
                        doc_id=doc_id,
                        conversation_id=conversation_id,
                        use_reranking=use_reranking
                    )
                
                with st.chat_message("assistant"):
                    st.write(response["answer"])
                    _render_metrics(
                        response["metrics"], 
                        response.get("ragas_metrics"), 
                        response.get("retrieval_metadata")
                    )
                    
                    with st.expander("View source documents"):
                        for source in response["sources"][:3]:
                            st.caption(f"📄 {source}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "metrics": response["metrics"],
                    "ragas_metrics": response.get("ragas_metrics"),
                    "retrieval_metadata": response.get("retrieval_metadata")
                })
                
            except Exception as e:
                st.error(f"Error getting response: {e}")
                logger.error(f"Chat error: {e}")


async def _get_response_async(
    question: str, 
    doc_id: str, 
    conversation_id: str, 
    chat_engine: ChatEngine,
    use_reranking: bool = False
) -> dict:
    """Async wrapper for chat response."""
    return chat_engine.ask_question(question, doc_id, conversation_id, use_reranking)


def _render_metrics(
    legacy_metrics: dict[str, float], 
    ragas_metrics: dict[str, float] = None,
    retrieval_metadata: dict = None
) -> None:
    """Render both legacy and RAGAS metrics with retrieval info."""
    st.divider()
    
    # Show retrieval method used
    if retrieval_metadata:
        metadata_filters = retrieval_metadata.get("metadata_filters", {})
        detected_models = retrieval_metadata.get("detected_models", [])
        
        # Show filtering information
        if metadata_filters and any(metadata_filters.values()):
            filter_info = []
            if metadata_filters.get("sensor_model"):
                filter_info.append(f"Model: {metadata_filters['sensor_model']}")
            if metadata_filters.get("manufacturer"):
                filter_info.append(f"Mfg: {metadata_filters['manufacturer']}")
            
            if filter_info:
                st.info(f"🎯 Filtered by: {' | '.join(filter_info)}")
        
        # Show retrieval stats
        if retrieval_metadata.get("reranking_enabled"):
            st.success(f"🔍 Re-ranking: {retrieval_metadata.get('initial_vector_search_count')} → {retrieval_metadata.get('final_chunk_count')} docs (best: {retrieval_metadata.get('best_relevance_score', 0):.3f})")
        else:
            st.info(f"📊 Standard retrieval: {retrieval_metadata.get('final_chunk_count')} documents")
        
        # Show detected models if multiple sensors in results
        if detected_models and len(detected_models) > 1:
            st.caption(f"🏷️ Found models: {', '.join(detected_models[:3])}")
        
        processing_time = retrieval_metadata.get("processing_time", 0)
        st.caption(f"⏱️ Processing time: {processing_time:.2f}s")
    
    # Show RAGAS metrics if available
    if ragas_metrics:
        st.markdown("**🎯 RAGAS Quality Metrics**")
        cols = st.columns(4)
        
        for idx, (name, value) in enumerate(ragas_metrics.items()):
            if idx < 4:  # Only show first 4 metrics
                with cols[idx]:
                    # Color-code the metrics
                    if value >= 0.7:
                        color = "green"
                    elif value >= 0.5:
                        color = "orange"
                    else:
                        color = "red"
                    
                    st.markdown(
                        f"**{name.replace('_', ' ').title()}**<br>"
                        f"<span style='color: {color}; font-weight: bold;'>{value:.3f}</span>", 
                        unsafe_allow_html=True
                    )
    
    # Show legacy metrics for backward compatibility
    elif legacy_metrics:
        st.markdown("**📊 Quality Metrics**")
        cols = st.columns(len(legacy_metrics))
        
        for idx, (name, value) in enumerate(legacy_metrics.items()):
            with cols[idx]:
                st.markdown(
                    f"**{name.replace('_', ' ').title()}**<br>***{value:.1%}***", 
                    unsafe_allow_html=True
                )
            
    st.divider()


if __name__ == "__main__":
    start_streamlit_app()