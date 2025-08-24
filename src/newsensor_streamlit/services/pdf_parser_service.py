from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from loguru import logger

from ..config import PDFParserBackend, Settings


class PDFParser(Protocol):
    """Protocol for PDF parser implementations."""
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF and return structured content with metadata."""
        ...
    
    def health_check(self) -> bool:
        """Check if parser is available and working."""
        ...


class LlamaParseParser:
    """LlamaParse implementation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.llama_parse_api_key
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using LlamaParse."""
        if not self.api_key:
            raise ValueError("LlamaParse API key not configured")
            
        try:
            import nest_asyncio
            from llama_parse import LlamaParse
            
            nest_asyncio.apply()
            
            # Content parser
            content_parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=True,
                language="en"
            )
            
            # Metadata parser with custom instructions
            metadata_parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                verbose=True,
                language="en",
                parsing_instruction=self._get_metadata_parsing_instruction()
            )
            
            content_docs = content_parser.load_data(str(file_path))
            metadata_docs = metadata_parser.load_data(str(file_path))
            
            if content_docs and len(content_docs) > 0:
                content = content_docs[0].text
                enhanced_metadata = self._extract_enhanced_metadata(metadata_docs)
                
                return {
                    "content": content,
                    "source": str(file_path),
                    "processed_at": str(Path(file_path).stat().st_mtime),
                    "metadata": {
                        "processing_backend": "LlamaParse",
                        "language": "en",
                        "format": "markdown",
                        **enhanced_metadata
                    }
                }
        except Exception as e:
            logger.error(f"LlamaParse failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if LlamaParse is working."""
        try:
            from llama_parse import LlamaParse
            return bool(self.api_key)
        except Exception as e:
            logger.error(f"LlamaParse health check failed: {e}")
            return False
    
    def _get_metadata_parsing_instruction(self) -> str:
        """Get the metadata parsing instruction for LlamaParse."""
        return """
        # Verbatim Datasheet Parser Prompt
        You are a **verbatim extractor**. Your job is to copy text snippets **exactly as they appear** in the provided PDF datasheet.
        Do **not** paraphrase, summarize, reformat, standardize units, correct typos, infer missing values, or merge/condense lines.
        Preserve capitalization, punctuation, symbols, units, whitespace, and line breaks within each extracted value.
        
        When a field is not present in the document, output exactly: `not specified`.
        
        Return the result in this exact schema format:
        SENSOR MODEL: [exact text]
        MANUFACTURER: [exact text]
        SENSOR TYPE: [exact text]
        OPERATING TEMPERATURE: [exact text]
        ACCURACY: [exact text]
        SUPPLY VOLTAGE: [exact text]
        OUTPUT TYPE: [exact text]
        FEATURES: [comma-separated list]
        APPLICATIONS: [comma-separated list]
        TOTAL PAGES: [numeric count]
        SPECIFICATIONS PAGE: [page number]
        FEATURES PAGE: [page number]
        WIRING PAGE: [page number]
        """
    
    def _extract_enhanced_metadata(self, metadata_docs: List[Any]) -> Dict[str, Any]:
        """Extract structured metadata from LlamaParse docs."""
        try:
            if not metadata_docs or len(metadata_docs) == 0:
                return self._get_default_metadata()
            
            metadata_text = metadata_docs[0].text
            return self._parse_structured_metadata(metadata_text)
            
        except Exception as e:
            logger.error(f"Enhanced metadata extraction failed: {e}")
            return self._get_default_metadata()
    
    def _parse_structured_metadata(self, text: str) -> Dict[str, Any]:
        """Parse structured text metadata format."""
        metadata = {}
        
        # Define patterns for extracting structured fields
        patterns = {
            "sensor_model": r"SENSOR MODEL:\s*(.+)",
            "manufacturer": r"MANUFACTURER:\s*(.+)",
            "alternate_models": r"ALTERNATE MODELS:\s*(.+)",
            "sensor_type": r"SENSOR TYPE:\s*(.+)",
            "operating_temperature": r"OPERATING TEMPERATURE:\s*(.+)",
            "accuracy": r"ACCURACY:\s*(.+)",
            "supply_voltage": r"SUPPLY VOLTAGE:\s*(.+)",
            "output_type": r"OUTPUT TYPE:\s*(.+)",
            "features": r"FEATURES:\s*(.+)",
            "applications": r"APPLICATIONS:\s*(.+)",
            "total_pages": r"TOTAL PAGES:\s*(.+)",
            "specifications_page": r"SPECIFICATIONS PAGE:\s*(.+)",
            "features_page": r"FEATURES PAGE:\s*(.+)",
            "wiring_page": r"WIRING PAGE:\s*(.+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value.lower() not in ["not specified", "", "none"]:
                    metadata[key] = value
        
        return self._validate_metadata(metadata)
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted metadata."""
        validated = {
            "sensor_model": str(metadata.get("sensor_model", "unknown")).strip(),
            "manufacturer": str(metadata.get("manufacturer", "unknown")).strip(),
            "alternate_models": metadata.get("alternate_models", []),
            "sensor_type": str(metadata.get("sensor_type", "unknown")).strip(),
            "specifications": metadata.get("specifications", {}),
            "features": metadata.get("features", []),
            "applications": metadata.get("applications", []),
            "document_sections": metadata.get("document_sections", []),
            "page_info": metadata.get("page_info", {})
        }
        
        return validated
    
    def _get_default_metadata(self) -> Dict[str, Any]:
        """Get default metadata when parsing fails."""
        return {
            "sensor_model": "unknown",
            "manufacturer": "unknown",
            "alternate_models": [],
            "sensor_type": "unknown",
            "specifications": {},
            "features": [],
            "applications": [],
            "document_sections": [],
            "page_info": {}
        }


class PyMuPDF4LLMParser:
    """PyMuPDF4LLM implementation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using PyMuPDF4LLM."""
        try:
            import pymupdf
            import pymupdf4llm
            
            doc = pymupdf.open(str(file_path))
            
            content = pymupdf4llm.to_markdown(
                doc, 
                hdr_info=None,
                write_images=self.settings.pymupdf_write_images,
                embed_images=self.settings.pymupdf_embed_images,
                show_progress=self.settings.pymupdf_show_progress,
                page_chunks=self.settings.pymupdf_page_chunks,
                margins=(0, 0, 0, 0)
            )
            
            enhanced_metadata = self._extract_pymupdf_metadata(doc, content)
            doc.close()
            
            return {
                "content": content,
                "source": str(file_path),
                "processed_at": str(Path(file_path).stat().st_mtime),
                "metadata": {
                    "processing_backend": "PyMuPDF4LLM",
                    "language": "en",
                    "format": "markdown",
                    **enhanced_metadata
                }
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF4LLM failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if PyMuPDF4LLM is working."""
        try:
            import pymupdf
            import pymupdf4llm
            test_doc = pymupdf.open()
            test_doc.close()
            return True
        except Exception as e:
            logger.error(f"PyMuPDF4LLM health check failed: {e}")
            return False
    
    def _extract_pymupdf_metadata(self, doc, content: str) -> Dict[str, Any]:
        """Extract metadata from PyMuPDF document."""
        try:
            metadata = {}
            
            # Extract document-level metadata
            doc_metadata = doc.metadata
            if doc_metadata:
                metadata.update({
                    "title": doc_metadata.get("title", ""),
                    "author": doc_metadata.get("author", ""),
                    "subject": doc_metadata.get("subject", ""),
                    "creator": doc_metadata.get("creator", ""),
                    "producer": doc_metadata.get("producer", "")
                })
            
            # Extract basic document info
            metadata["page_info"] = {
                "total_pages": doc.page_count
            }
            
            # Parse content for sensor-specific information
            sensor_metadata = self._parse_sensor_metadata_from_content(content)
            metadata.update(sensor_metadata)
            
            # Try to extract table of contents
            toc = doc.get_toc()
            if toc:
                metadata["document_sections"] = [
                    {
                        "section_title": item[1],  # Title
                        "page_numbers": [item[2]],  # Page number
                        "level": item[0],  # Hierarchy level
                        "content_summary": ""
                    }
                    for item in toc
                ]
            
            return self._validate_metadata(metadata)
            
        except Exception as e:
            logger.error(f"PyMuPDF metadata extraction failed: {e}")
            return self._extract_fallback_metadata(content)
    
    def _parse_sensor_metadata_from_content(self, content: str) -> Dict[str, Any]:
        """Extract sensor-specific metadata from markdown content."""
        metadata = {}
        
        # Enhanced patterns for sensor datasheets
        patterns = {
            "sensor_model": [
                r"(?:Model|Part\s+Number|Product\s+Code|Device|Sensor)\s*:?\s*([A-Z0-9\-_]+)",
                r"^#\s*([A-Z0-9\-_]+)\s*$",  # Often model is in first header
                r"\b([A-Z]{2,}\d{2,}[A-Z0-9\-_]*)\b"  # Common sensor naming pattern
            ],
            "manufacturer": [
                r"(?:Manufacturer|Company|Brand)\s*:?\s*([A-Za-z\s&.]+)",
                r"©\s*(\w+(?:\s+\w+)?)",  # Copyright line
                r"\*\*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\*\*"  # Bold company names
            ],
            "sensor_type": [
                r"(?:Temperature|Humidity|Pressure|Flow|Level|pH|Conductivity|Dissolved\s+Oxygen)\s+(?:Sensor|Transmitter|Probe)",
                r"(?:Sensor|Transmitter|Probe)\s+(?:Type|Category)\s*:?\s*([A-Za-z\s]+)"
            ]
        }
        
        # Extract specifications
        spec_patterns = {
            "operating_temperature": [
                r"(?:Operating|Working)\s+Temperature\s*:?\s*([-+]?\d+°?[CF]?\s*(?:to|~|-)\s*[-+]?\d+°?[CF]?)",
                r"Temperature\s+Range\s*:?\s*([-+]?\d+°?[CF]?\s*(?:to|~|-)\s*[-+]?\d+°?[CF]?)"
            ],
            "accuracy": [
                r"Accuracy\s*:?\s*(±?\s*[\d.]+\s*%?[A-Za-z]*)",
                r"(?:Precision|Error)\s*:?\s*(±?\s*[\d.]+\s*%?[A-Za-z]*)"
            ],
            "supply_voltage": [
                r"(?:Supply|Power|Operating)\s+Voltage\s*:?\s*([\d.-]+\s*(?:to|~|-)\s*[\d.-]+\s*V?)",
                r"(?:VDC|VAC|V)\s*:?\s*([\d.-]+(?:\s*(?:to|~|-)\s*[\d.-]+)?)"
            ],
            "output_type": [
                r"Output\s*:?\s*([\d.-]+\s*(?:to|~|-)\s*[\d.-]+\s*(?:mA|V|VDC))",
                r"Signal\s*:?\s*([\d.-]+\s*(?:to|~|-)\s*[\d.-]+\s*(?:mA|V|VDC))"
            ]
        }
        
        # Extract main metadata
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value and value.lower() not in ["not specified", "", "none", "unknown"]:
                        metadata[key] = value
                        break
        
        # Extract specifications
        specifications = {}
        for key, pattern_list in spec_patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value and value.lower() not in ["not specified", "", "none", "unknown"]:
                        specifications[key] = value
                        break
        
        if specifications:
            metadata["specifications"] = specifications
        
        return metadata
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted metadata."""
        return {
            "sensor_model": str(metadata.get("sensor_model", "unknown")).strip(),
            "manufacturer": str(metadata.get("manufacturer", "unknown")).strip(),
            "alternate_models": metadata.get("alternate_models", []),
            "sensor_type": str(metadata.get("sensor_type", "unknown")).strip(),
            "specifications": metadata.get("specifications", {}),
            "features": metadata.get("features", []),
            "applications": metadata.get("applications", []),
            "document_sections": metadata.get("document_sections", []),
            "page_info": metadata.get("page_info", {})
        }
    
    def _extract_fallback_metadata(self, content: str) -> Dict[str, Any]:
        """Extract basic metadata when advanced parsing fails."""
        return {
            "sensor_model": "unknown",
            "manufacturer": "unknown",
            "alternate_models": [],
            "sensor_type": "unknown",
            "specifications": {},
            "features": [],
            "applications": [],
            "document_sections": [],
            "page_info": {}
        }


class PyPDF2Parser:
    """PyPDF2 fallback implementation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using PyPDF2."""
        try:
            from PyPDF2 import PdfReader
            
            text_content = ""
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text_content += f"## Page {page_num + 1}\n\n{page_text}\n\n"
            
            fallback_metadata = self._extract_basic_metadata(text_content)
            fallback_metadata["page_info"] = {"total_pages": total_pages}
            
            return {
                "content": text_content,
                "source": str(file_path),
                "processed_at": str(file_path.stat().st_mtime),
                "metadata": {
                    "processing_backend": "PyPDF2",
                    "language": "en",
                    **fallback_metadata
                }
            }
            
        except Exception as e:
            logger.error(f"PyPDF2 failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if PyPDF2 is working."""
        try:
            from PyPDF2 import PdfReader
            return True
        except Exception as e:
            logger.error(f"PyPDF2 health check failed: {e}")
            return False
    
    def _extract_basic_metadata(self, content: str) -> Dict[str, Any]:
        """Extract basic metadata from text content."""
        # Simple regex patterns for common datasheet elements
        model_patterns = [
            r"Model\s*:?\s*([A-Z0-9\-]+)",
            r"Part\s*Number\s*:?\s*([A-Z0-9\-]+)",
            r"Product\s*Code\s*:?\s*([A-Z0-9\-]+)"
        ]
        
        sensor_model = "unknown"
        for pattern in model_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                sensor_model = match.group(1).strip()
                break
        
        # Look for manufacturer
        manufacturer_patterns = [
            r"Manufacturer\s*:?\s*([A-Za-z\s]+)",
            r"Company\s*:?\s*([A-Za-z\s]+)",
            r"Brand\s*:?\s*([A-Za-z\s]+)"
        ]
        
        manufacturer = "unknown"
        for pattern in manufacturer_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                manufacturer = match.group(1).strip()
                break
        
        return {
            "sensor_model": sensor_model,
            "manufacturer": manufacturer,
            "alternate_models": [],
            "sensor_type": "unknown",
            "specifications": {},
            "features": [],
            "applications": [],
            "document_sections": [],
            "page_info": {}
        }


class PDFParserService:
    """Dynamic PDF parser service with configurable backends."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._parsers = {
            PDFParserBackend.LLAMAPARSE: LlamaParseParser(settings),
            PDFParserBackend.PYMUPDF4LLM: PyMuPDF4LLMParser(settings),
            PDFParserBackend.PYPDF2: PyPDF2Parser(settings),
        }
        
        # Log the configured parser backend
        try:
            primary_backend = PDFParserBackend(self.settings.pdf_parser_backend)
            logger.info(f"PDF parser configured: {primary_backend.value} (fallback: {[b.value for b in self.settings.pdf_parser_fallback_list]})")
        except ValueError:
            logger.warning(f"Invalid backend '{self.settings.pdf_parser_backend}', will use fallback")
        
        
    def process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process PDF using configured backend with fallback chain."""
        logger.info(f"Processing PDF: {file_path}")
        
        # Get primary backend and fallback chain
        try:
            primary_backend = PDFParserBackend(self.settings.pdf_parser_backend)
        except ValueError:
            logger.warning(f"Invalid backend '{self.settings.pdf_parser_backend}', using pymupdf4llm")
            primary_backend = PDFParserBackend.PYMUPDF4LLM
        
        fallback_chain = self.settings.pdf_parser_fallback_list
        
        logger.info(f"Primary backend: {primary_backend}")
        logger.info(f"Fallback chain: {fallback_chain}")
        
        # Try primary backend first, then fallbacks
        backends_to_try = [primary_backend] + [
            backend for backend in fallback_chain 
            if backend != primary_backend
        ]
        
        last_error = None
        for backend in backends_to_try:
            try:
                parser = self._parsers[backend]
                if parser.health_check():
                    logger.info(f"Trying {backend.value} parser...")
                    result = parser.parse(file_path)
                    logger.success(f"Successfully parsed with {backend.value}")
                    return result
                else:
                    logger.warning(f"{backend.value} health check failed")
                    continue
                    
            except Exception as e:
                logger.warning(f"{backend.value} failed: {e}")
                last_error = e
                continue
        
        # All parsers failed
        logger.error(f"All PDF parsers failed. Last error: {last_error}")
        return self._create_fallback_result(file_path, last_error)
    
    def _create_fallback_result(self, file_path: Path, error: Exception) -> Dict[str, Any]:
        """Create a minimal result when all parsers fail."""
        return {
            "content": f"PDF content from {file_path.name} (parsing failed)",
            "source": str(file_path),
            "processed_at": str(file_path.stat().st_mtime),
            "metadata": {
                "processing_backend": "fallback",
                "language": "en",
                "error": str(error),
                "sensor_model": "unknown",
                "manufacturer": "unknown",
                "alternate_models": [],
                "sensor_type": "unknown",
                "specifications": {},
                "features": [],
                "applications": [],
                "document_sections": [],
                "page_info": {}
            }
        }
    
    def get_available_backends(self) -> List[PDFParserBackend]:
        """Get list of available backends based on health checks."""
        available = []
        for backend, parser in self._parsers.items():
            if parser.health_check():
                available.append(backend)
        return available
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all configured parsers."""
        return {
            backend.value: parser.health_check() 
            for backend, parser in self._parsers.items()
        }
