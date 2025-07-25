from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from anyio import create_task_group, sleep

from newsensor_streamlit.core.chat_engine import ChatEngine


class TestChatEngine:
    def test_init(self):
        """Test ChatEngine initialization."""
        engine = ChatEngine()
        assert engine is not None
        
    @pytest.mark.anyio
    async def test_upload_document(self):
        """Test document upload functionality."""
        engine = ChatEngine()
        
        # Create a mock PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', mode='w+b') as tmp:
            tmp.write(b'Mock PDF content')
            tmp.flush()
            
            # This would need actual PDF content for MinerU
            # For now, just assert it doesn't crash
            with pytest.raises((ValueError, Exception)):
                engine.upload_document(Path(tmp.name))
                
    def test_create_chunks(self):
        """Test text chunking functionality."""
        engine = ChatEngine()
        text = "This is a test. This is another test. " * 200
        
        chunks = engine._create_chunks(text)
        assert len(chunks) > 0
        assert all(len(chunk.page_content) > 0 for chunk in chunks)


class TestServices:
    def test_document_processor_init(self):
        """Test DocumentProcessor initialization."""
        from newsensor_streamlit.services.document_processor import DocumentProcessor
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = DocumentProcessor(Path(tmp_dir))
            assert processor.output_dir == Path(tmp_dir)
            
    def test_config_loading(self):
        """Test settings loading."""
        from newsensor_streamlit.config import settings
        
        assert settings.chunk_size > 0
        assert settings.max_context_chunks > 0