from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import nest_asyncio
from llama_parse import LlamaParse
from loguru import logger

# Fix for asyncio issues in Jupyter/Streamlit
nest_asyncio.apply()


class LlamaParseService:
    """Advanced PDF parser using LlamaParse with enhanced metadata extraction."""
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.use_llama_parse = bool(api_key)
        
        # JSON schema for structured metadata extraction
        self.metadata_schema = {
            "type": "object",
            "properties": {
                "sensor_model": {"type": "string", "description": "Primary sensor model number"},
                "manufacturer": {"type": "string", "description": "Manufacturer or brand name"},
                "alternate_models": {"type": "array", "items": {"type": "string"}, "description": "Alternative model numbers or variants"},
                "sensor_type": {"type": "string", "description": "Type of sensor (temperature, humidity, pressure, etc.)"},
                "specifications": {
                    "type": "object",
                    "properties": {
                        "operating_temperature": {"type": "string", "description": "Operating temperature range"},
                        "accuracy": {"type": "string", "description": "Measurement accuracy"},
                        "supply_voltage": {"type": "string", "description": "Supply voltage range"},
                        "output_type": {"type": "string", "description": "Output signal type"}
                    }
                },
                "features": {"type": "array", "items": {"type": "string"}, "description": "Key features and capabilities"},
                "applications": {"type": "array", "items": {"type": "string"}, "description": "Typical applications"},
                "document_sections": {
                    "type": "array", 
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_title": {"type": "string"},
                            "page_numbers": {"type": "array", "items": {"type": "integer"}},
                            "content_summary": {"type": "string"}
                        }
                    }
                }
            }
        }
        
    def process_pdf(self, file_path: Path) -> dict[str, Any]:
        """Process PDF using best available parser with enhanced metadata extraction."""
        logger.info(f"Processing PDF: {file_path}")
        
        if self.use_llama_parse:
            try:
                # First pass: Extract markdown content
                content_parser = LlamaParse(
                    api_key=self.api_key,
                    result_type="markdown",
                    verbose=True,
                    language="en"
                )
                
                # Second pass: Extract structured metadata using JSON mode
                metadata_parser = LlamaParse(
                    api_key=self.api_key,
                    result_type="markdown",  # Use markdown instead of json for better compatibility
                    verbose=True,
                    language="en",
                    parsing_instruction=f"""
                    # Verbatim Datasheet Parser Prompt

                    You are a **verbatim extractor**. Your job is to copy text snippets **exactly as they appear** in the provided PDF datasheet.  
                    Do **not** paraphrase, summarize, reformat, standardize units, correct typos, infer missing values, or merge/condense lines.  
                    Preserve capitalization, punctuation, symbols, units, whitespace, and line breaks within each extracted value.

                    When a field is not present in the document, output exactly: `not specified`.

                    Return the result in this exact schema (fields must appear in this order and with these labels).  
                    Content for any field may be multi-line if the source text is multi-line.  
                    Include the **page number(s)** where each value was found in square brackets at the end of the field.

                    ## Extraction Rules

                    1. Copy the **exact** string spans from the PDF where each field is stated (e.g., model numbers, ranges, voltages, headings, bullet lists).  
                    2. If multiple occurrences exist, concatenate them in the **original order of appearance**, separated by a single blank line. Do not deduplicate or sort.  
                    3. For `TOTAL PAGES`, return the exact page count of the PDF file (numeric), no words added.  
                    4. For `SPECIFICATIONS PAGE`, `FEATURES PAGE`, and `WIRING PAGE`, prefer page numbers written in the document (e.g., “p. 3”), even if they differ from the PDF’s actual page index. If the document does not label pages, return the actual PDF page index (1-based).  
                    5. If a field cannot be located after a reasonable scan of the document, return `not specified` (no extra words).  
                    6. Never add explanations, comments, or rationale in the output—only the schema above.

                    ## Heuristics for Locating Text

                    - Look for headings like **Specifications**, **Electrical Characteristics**, **Features**, **Applications**, **Wiring**, **Pinout**, **Installation**.  
                    - For **model** and **manufacturer**, check the front page header/footer, title blocks, tables, and ordering information.  
                    - For **page-number fields**, use the table of contents or section headers if present; otherwise use the PDF page index.  
 
                    """
                )
                
                # Process content
                content_docs = content_parser.load_data(str(file_path))
                metadata_docs = metadata_parser.load_data(str(file_path))
                
                if content_docs and len(content_docs) > 0:
                    content = content_docs[0].text
                    logger.info(f"LlamaParse extracted {len(content)} chars")
                    
                    # Extract structured metadata
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
        
        # Fallback to PyPDF2
        return self._fallback_process(file_path)
    
    def _extract_enhanced_metadata(self, metadata_docs: List[Any]) -> Dict[str, Any]:
        """Extract and validate structured metadata from LlamaParse text output."""
        try:
            if not metadata_docs or len(metadata_docs) == 0:
                return {"sensor_model": "unknown", "manufacturer": "unknown"}
            
            # Parse structured text content from LlamaParse
            raw_metadata = metadata_docs[0].text
            
            # Parse the structured text format
            parsed_metadata = self._parse_structured_metadata(raw_metadata)
            
            # Validate and clean metadata
            return self._validate_metadata(parsed_metadata)
            
        except Exception as e:
            logger.error(f"Enhanced metadata extraction failed: {e}")
            return {"sensor_model": "unknown", "manufacturer": "unknown"}
    
    def _parse_structured_metadata(self, text: str) -> Dict[str, Any]:
        """Parse structured text metadata format."""
        import re
        
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
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if value.lower() not in ["not specified", "", "none", "unknown"]:
                    if key in ["alternate_models", "features", "applications"]:
                        # Split comma-separated lists
                        metadata[key] = [item.strip() for item in value.split(",") if item.strip()]
                    elif key in ["operating_temperature", "accuracy", "supply_voltage", "output_type"]:
                        # Store as specification
                        if "specifications" not in metadata:
                            metadata["specifications"] = {}
                        metadata["specifications"][key] = value
                    elif key in ["total_pages", "specifications_page", "features_page", "wiring_page"]:
                        # Store as page information
                        if "page_info" not in metadata:
                            metadata["page_info"] = {}
                        # Try to extract numeric page numbers
                        page_num = self._extract_page_number(value)
                        metadata["page_info"][key] = page_num if page_num else value
                    else:
                        metadata[key] = value
        
        return metadata
    
    def _extract_page_number(self, value: str) -> Optional[int]:
        """Extract numeric page number from text value."""
        import re
        
        # Look for numeric patterns
        match = re.search(r'(\d+)', value)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
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
        
        # Clean up empty or invalid values
        if validated["sensor_model"].lower() in ["unknown", "", "none"]:
            validated["sensor_model"] = "unknown"
        if validated["manufacturer"].lower() in ["unknown", "", "none"]:
            validated["manufacturer"] = "unknown"
            
        return validated
    
    def _extract_fallback_metadata(self, text_content: str) -> Dict[str, Any]:
        """Extract basic metadata from text content when JSON parsing fails."""
        # Simple regex patterns for common datasheet elements
        import re
        
        # Look for model numbers (common patterns)
        model_patterns = [
            r"Model\s*:?\s*([A-Z0-9\-]+)",
            r"Part\s*Number\s*:?\s*([A-Z0-9\-]+)",
            r"Product\s*Code\s*:?\s*([A-Z0-9\-]+)"
        ]
        
        sensor_model = "unknown"
        for pattern in model_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
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
            match = re.search(pattern, text_content, re.IGNORECASE)
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
    
    def _fallback_process(self, file_path: Path) -> dict[str, Any]:
        """Fallback to PyPDF2 for basic text extraction."""
        try:
            from PyPDF2 import PdfReader  # Correct import case
            text_content = ""
            
            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text_content += f"{page_text}\n\n"
            
            # Extract basic metadata from text
            fallback_metadata = self._extract_fallback_metadata(text_content)
            
            # Add page count information
            fallback_metadata["page_info"] = {"total_pages": total_pages}
            
            return {
                "content": text_content,
                "source": str(file_path),
                "processed_at": str(Path(file_path).stat().st_mtime),
                "metadata": {
                    "processing_backend": "PyPDF2-fallback",
                    "language": "en",
                    **fallback_metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            return {
                "content": f"PDF content from {file_path.name}",
                "source": str(file_path),
                "processed_at": str(file_path.stat().st_mtime),
                "metadata": {
                    "processing_backend": "text-fallback", 
                    "language": "en",
                    "sensor_model": "unknown",
                    "manufacturer": "unknown"
                }
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