from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

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
        """Process PDF using best available parser with enhanced metadata."""
        logger.info(f"Processing PDF: {file_path}")
        result = self.parser.process_pdf(file_path)
        
        # Enhance with document-level metadata
        result["metadata"]["filename"] = file_path.name
        result["metadata"]["file_stem"] = file_path.stem
        
        return result
    
    def create_metadata_aware_chunks(self, document: Dict[str, Any], chunk_size: int = 512) -> List[Dict[str, Any]]:
        """Create chunks with enhanced metadata for better retrieval."""
        content = document["content"]
        doc_metadata = document["metadata"]
        
        # Extract core identification metadata
        sensor_model = doc_metadata.get("sensor_model", "unknown")
        manufacturer = doc_metadata.get("manufacturer", "unknown")
        alternate_models = doc_metadata.get("alternate_models", [])
        sensor_type = doc_metadata.get("sensor_type", "unknown")
        
        # Create chunks using sentence-aware splitting
        chunks = self._split_into_semantic_chunks(content, chunk_size)
        
        # Enhance each chunk with metadata
        enhanced_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Determine chunk-specific context
            chunk_context = self._analyze_chunk_context(chunk_text, doc_metadata)
            
            chunk_metadata = {
                # Core document identification
                "sensor_model": sensor_model,
                "manufacturer": manufacturer,
                "alternate_models": alternate_models,
                "sensor_type": sensor_type,
                
                # Document metadata
                "filename": doc_metadata.get("filename", "unknown"),
                "file_stem": doc_metadata.get("file_stem", "unknown"),
                "source": document["source"],
                "processing_backend": doc_metadata.get("processing_backend", "unknown"),
                
                # Chunk-specific metadata
                "chunk_id": i,
                "chunk_context": chunk_context,
                "total_chunks": len(chunks),
                
                # Search enhancement fields
                "searchable_models": self._create_searchable_models(sensor_model, alternate_models),
                "specifications": doc_metadata.get("specifications", {}),
                "features": doc_metadata.get("features", []),
                "applications": doc_metadata.get("applications", []),
                
                # Page information
                "page_info": doc_metadata.get("page_info", {}),
                "total_pages": doc_metadata.get("page_info", {}).get("total_pages", "unknown"),
                "estimated_page": self._estimate_chunk_page(i, len(chunks), doc_metadata.get("page_info", {}))
            }
            
            enhanced_chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
            
        logger.info(f"Created {len(enhanced_chunks)} metadata-aware chunks for {sensor_model}")
        return enhanced_chunks
    
    def _split_into_semantic_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into semantic chunks respecting sentence boundaries."""
        # Split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _analyze_chunk_context(self, chunk_text: str, doc_metadata: Dict[str, Any]) -> str:
        """Analyze chunk content to determine its primary context/topic."""
        text_lower = chunk_text.lower()
        
        # Define context patterns
        context_patterns = {
            "specifications": ["specification", "spec", "parameter", "range", "accuracy", "tolerance"],
            "features": ["feature", "capability", "function", "benefit", "advantage"],
            "applications": ["application", "use case", "suitable for", "ideal for", "recommended"],
            "installation": ["install", "mount", "wiring", "connection", "setup"],
            "technical": ["technical", "diagram", "schematic", "circuit", "pin"],
            "overview": ["overview", "introduction", "general", "about", "description"]
        }
        
        # Score each context
        context_scores = {}
        for context, keywords in context_patterns.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                context_scores[context] = score
        
        # Return the highest scoring context, or "general" if none found
        if context_scores:
            return max(context_scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    def _create_searchable_models(self, primary_model: str, alternate_models: List[str]) -> List[str]:
        """Create comprehensive list of searchable model variations."""
        searchable = set()
        
        # Add primary model and alternates
        models_to_process = [primary_model] + alternate_models
        
        for model in models_to_process:
            if model and model != "unknown":
                # Add the original model
                searchable.add(model)
                
                # Add variations without hyphens and spaces
                searchable.add(model.replace("-", ""))
                searchable.add(model.replace(" ", ""))
                searchable.add(model.upper())
                searchable.add(model.lower())
                
                # Add base model (everything before first hyphen)
                if "-" in model:
                    base_model = model.split("-")[0]
                    searchable.add(base_model)
                    searchable.add(base_model.upper())
                    searchable.add(base_model.lower())
                
                # Add partial matches for longer models
                if len(model) > 4:
                    # Add first 3-4 characters for partial matching
                    searchable.add(model[:3])
                    searchable.add(model[:4])
                    searchable.add(model[:3].upper())
                    searchable.add(model[:4].upper())
        
        # Remove empty strings and duplicates
        return list(filter(None, searchable))
    
    def _estimate_chunk_page(self, chunk_index: int, total_chunks: int, page_info: Dict[str, Any]) -> str:
        """Estimate which page a chunk might be from based on position."""
        total_pages = page_info.get("total_pages")
        
        if not total_pages or total_pages == "unknown":
            return "unknown"
        
        try:
            total_pages_num = int(total_pages)
            if total_pages_num <= 0:
                return "unknown"
            
            # Simple estimation: distribute chunks evenly across pages
            estimated_page = int((chunk_index / total_chunks) * total_pages_num) + 1
            estimated_page = min(estimated_page, total_pages_num)  # Don't exceed total pages
            
            return str(estimated_page)
        except (ValueError, TypeError):
            return "unknown"
            
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