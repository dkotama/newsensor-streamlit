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
    st.markdown("Ask questions about sensor datasheets with confidence")
    
    chat_engine = ChatEngine()
    
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF datasheet",
            type=["pdf"],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                _handle_document_upload(uploaded_file, chat_engine)
                
    # Main chat interface
    if st.session_state.get("current_doc_id"):
        _render_chat_interface(chat_engine)


def _handle_document_upload(uploaded_file, chat_engine: ChatEngine) -> None:
    """Handle document upload and processing."""
    logger.info(f"Starting document upload: {uploaded_file.name}")
    
    try:
        with st.spinner("Processing document..."):
            save_path = Path(settings.uploads_dir) / uploaded_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Saving file to: {save_path}")
            
            # Save uploaded file
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            
            logger.info(f"Processing document: {save_path}")
            # Process and store
            doc_id = chat_engine.upload_document(save_path)
            st.session_state.current_doc_id = doc_id
            st.session_state.messages = []
            st.success("Document processed successfully!")
            logger.info(f"Document processed successfully: {doc_id}")
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.error(f"Error processing document: {e}")
        st.text(str(e))  # Show stack trace for debugging


def _render_chat_interface(chat_engine: ChatEngine) -> None:
    """Render the main chat interface."""
    doc_id = st.session_state.current_doc_id
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if "metrics" in message:
                _render_metrics(message["metrics"])
    
    # Chat input
    if user_input := st.chat_input("Ask about the datasheet..."):
        with st.chat_message("user"):
            st.write(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            response = asyncio.run(_get_response_async(user_input, doc_id, chat_engine))
            
            with st.chat_message("assistant"):
                st.write(response["answer"])
                _render_metrics(response["metrics"])
                
                with st.expander("View source documents"):
                    for source in response["sources"][:3]:
                        st.caption(f"📄 {source}")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "metrics": response["metrics"]
            })
            
        except Exception as e:
            st.error(f"Error getting response: {e}")


async def _get_response_async(question: str, doc_id: str, chat_engine: ChatEngine) -> dict:
    """Async wrapper for chat response."""
    return chat_engine.ask_question(question, doc_id)


def _render_metrics(metrics: dict[str, float]) -> None:
    """Render RAGAS metrics in a compact format."""
    st.divider()
    cols = st.columns(4)
    
    # Create very small, clean metrics
    for idx, (name, value) in enumerate(metrics.items()):
        with cols[idx]:
            st.markdown(
                f"**{name.replace('_', ' ').title()}**<br>***{value:.1%}***", 
                unsafe_allow_html=True
            )
            
    st.divider()


if __name__ == "__main__":
    start_streamlit_app()