from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from newsensor_streamlit.config import settings
from newsensor_streamlit.services.llama_parse_service import LlamaParseService


class DocumentProcessor:
    def __init__(self, output_dir) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use LlamaParse if API key available, otherwise PyPDF fallback
        if settings.llama_parse_api_key:
            logger.info("Using LlamaParse for advanced PDF processing")
            self.parser = LlamaParseService(settings.llama_parse_api_key)
        else:
            logger.warning("LlamaParse API key not found, using basic PDF processing")
            self.parser = LlamaParseService("")  # Will use fallback

    def process_pdf(self, file_path: Path) -> dict[str, Any]:
        """Process PDF using best available parser."""
        logger.info(f"Processing PDF: {file_path}")
        return self.parser.process_pdf(file_path)
            
    def process_batch(self, file_paths: list[Path]) -> list[dict[str, Any]]:
        """Process multiple PDFs."""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_pdf(file_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Skipping file {file_path}: {e}")
                continue
        return results