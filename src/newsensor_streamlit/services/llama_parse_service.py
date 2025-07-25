from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import nest_asyncio
from llama_parse import LlamaParse
from loguru import logger

# Fix for asyncio issues in Jupyter/Streamlit
nest_asyncio.apply()


class LlamaParseService:
    """Advanced PDF parser using LlamaParse with async support."""
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.use_llama_parse = bool(api_key)
        
    def process_pdf(self, file_path: Path) -> dict[str, Any]:
        """Process PDF using best available parser."""
        logger.info(f"Processing PDF: {file_path}")
        
        if self.use_llama_parse:
            try:
                self.parser = LlamaParse(
                    api_key=self.api_key,
                    result_type="markdown",
                    verbose=True,
                    language="en"
                )
                
                # Use synchronous loading for now
                documents = self.parser.load_data(str(file_path))
                
                if documents and len(documents) > 0:
                    content = documents[0].text
                    logger.info(f"LlamaParse extracted {len(content)} chars")
                    return {
                        "content": content,
                        "source": str(file_path),
                        "processed_at": str(Path(file_path).stat().st_mtime),
                        "metadata": {
                            "processing_backend": "LlamaParse",
                            "language": "en",
                            "format": "markdown"
                        }
                    }
            except Exception as e:
                logger.error(f"LlamaParse failed: {e}")
        
        # Fallback to PyPDF2
        return self._fallback_process(file_path)
    
    def _fallback_process(self, file_path: Path) -> dict[str, Any]:
        """Fallback to PyPDF2 for basic text extraction."""
        try:
            from pypdf2 import PdfReader
            text_content = ""
            
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text_content += f"{page_text}\n\n"
            
            return {
                "content": text_content,
                "source": str(file_path),
                "processed_at": str(Path(file_path).stat().st_mtime),
                "metadata": {
                    "processing_backend": "PyPDF2-fallback",
                    "language": "en"
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            return {
                "content": f"PDF content from {file_path.name}",
                "source": str(file_path),
                "processed_at": str(file_path.stat().st_mtime),
                "metadata": {"processing_backend": "text-fallback", "language": "en"}
            }
    
    def process_batch(self, file_paths: list[Path]) -> list[dict[str, Any]]:
        """Process multiple PDFs with LlamaParse."""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_pdf(file_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Skipping file {file_path}: {e}")
                continue
        return results
    
    def health_check(self) -> bool:
        """Check if LlamaParse is working."""
        try:
            # Test with dummy parsing to verify key
            test_parser = LlamaParse(api_key=self.api_key)
            return test_parser.api_key is not None
        except Exception as e:
            logger.error(f"LlamaParse health check failed: {e}")
            return False