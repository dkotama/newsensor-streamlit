from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from ..config import Settings
from .pdf_parser_service import PDFParserService


class LlamaParseService:
    """Dynamic PDF parser service - now acts as a facade for multiple parser backends."""
    
    def __init__(self, api_key: str) -> None:
        # For backward compatibility, we still accept api_key
        # but the actual configuration comes from Settings
        self.api_key = api_key
        
        # Initialize the dynamic parser service
        from ..config import settings
        self.parser_service = PDFParserService(settings)
        
    def process_pdf(self, file_path: Path) -> dict[str, Any]:
        """Process PDF using configured dynamic parser backend."""
        logger.info(f"Processing PDF: {file_path}")
        return self.parser_service.process_pdf(file_path)
    
    def process_batch(self, file_paths: list[Path]) -> list[dict[str, Any]]:
        """Process multiple PDFs."""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_pdf(file_path)
                results.append(result)
                logger.success(f"Processed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Add error result to maintain list consistency
                results.append({
                    "content": f"Failed to process {file_path.name}",
                    "source": str(file_path),
                    "processed_at": str(file_path.stat().st_mtime),
                    "metadata": {
                        "processing_backend": "error",
                        "error": str(e),
                        "sensor_model": "unknown",
                        "manufacturer": "unknown"
                    }
                })
        return results
    
    def health_check(self) -> bool:
        """Check if at least one parser backend is available."""
        health_status = self.parser_service.health_check()
        return any(health_status.values())
    
    def get_parser_status(self) -> dict[str, Any]:
        """Get detailed status of all parser backends."""
        health_status = self.parser_service.health_check()
        available_backends = self.parser_service.get_available_backends()
        
        return {
            "backend_health": health_status,
            "available_backends": [backend.value for backend in available_backends],
            "configured_backend": self.parser_service.settings.pdf_parser_backend,
            "fallback_chain": [backend.value for backend in self.parser_service.settings.pdf_parser_fallback_list]
        }
