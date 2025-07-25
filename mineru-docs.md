### Mineru: High-Quality PDF to Data Conversion ###

## 1. Introduction ##

Mineru is a powerful, open-source toolkit designed to convert PDF documents into structured, machine-readable formats like Markdown and JSON. Born out of the data processing needs for training Large Language Models like InternLM, Mineru excels at parsing complex documents, especially scientific and technical literature. It intelligently extracts and organizes content, making it an ideal tool for RAG (Retrieval-Augmented Generation) pipelines, data analysis, and document digitalization.

### Key Features ###

- **Intelligent Cleaning**: Automatically identifies and removes headers, footers, page numbers, and other non-semantic elements to preserve the core content flow.
- **Structure Preservation**: Accurately recognizes and maintains the document's original structure, including headings, paragraphs, lists, and reading order, even in complex multi-column layouts.
- **Rich Content Extraction**: Extracts not just text, but also images, tables, and mathematical formulas, converting them into accessible formats (e.g., tables to HTML/Markdown, formulas to LaTeX).
- **Advanced OCR**: Automatically detects scanned or image-based PDFs and applies OCR. It supports over 80 languages, thanks to its integration with PaddleOCR.
- **Flexible Backends**: Offers a choice between a traditional pipeline backend with specialized models and a modern VLM (Vision Language Model) backend for end-to-end processing.
- **Developer-Friendly Outputs**: Generates multiple output files for different needs, including clean Markdown, a rich intermediate JSON for programmatic access, and visualization files (bounding boxes) for easy quality inspection.
- **Performance**: Supports both CPU and GPU (CUDA) environments for accelerated processing.

## 2. Installation ##

You can install Mineru using pip. For the best experience, it is recommended to use `uv`, a fast Python package installer.

```bash
# Install uv (if you don't have it)
pip install uv

# Install the core MinerU package
uv pip install "mineru[core]"

# To include all features, including VLM acceleration with sglang
# Note: Requires a compatible NVIDIA GPU (Turing architecture or newer)
uv pip install "mineru[all]"
```

Alternatively, you can install from the source on GitHub:

```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
uv pip install -e ".[core]"
```

## 3. Direct API Usage (Quick Start) ##

Mineru provides a straightforward Python API for integration into your projects. The primary entry point is the `parse_doc` function, which handles the entire parsing workflow.

### The parse_doc Function ###

This function takes a list of document paths and several configuration options, processing each document and saving the results to the specified output directory.

Here is the function signature and an explanation of its parameters:

```python
def parse_doc(
    path_list: list[Path],
    output_dir: str,
    lang: str = "ch",
    backend: str = "pipeline",
    method: str = "auto",
    server_url: str = None,
    start_page_id: int = 0,
    end_page_id: int = None
):
    """
    Parses a list of documents (PDFs or images) and saves the output.

    Args:
        path_list (list[Path]): List of document paths to be parsed.
        output_dir (str): Directory to store the parsing results.
        lang (str): Language hint for OCR. Improves accuracy.
                    Default is 'ch'. Supported values include 'en', 'korean', 'japan', etc.
                    Only used with the 'pipeline' backend.
        backend (str): The parsing engine to use.
                       - "pipeline": A general-purpose, rule-based engine.
                       - "vlm-transformers": Uses a general Vision Language Model.
                       - "vlm-sglang-engine": A faster VLM engine.
                       - "vlm-sglang-client": Connects to a running sglang server.
        method (str): The parsing method. Only used with the 'pipeline' backend.
                      - "auto": Automatically determines if the PDF is text-based or scanned.
                      - "txt": Forces text extraction (for text-based PDFs).
                      - "ocr": Forces OCR (for scanned/image-based PDFs).
        server_url (str): The URL of the sglang server (e.g., "http://127.0.0.1:30000")
                          Required when using the "vlm-sglang-client" backend.
        start_page_id (int): The page number to start parsing from (0-indexed).
        end_page_id (int): The page number to stop parsing at. If None, parses to the end.
    """
```

### Full Code Example ###

The following script demonstrates how to use `parse_doc` to process all PDFs in a directory.

```python
# Copyright (c) Opendatalab. All rights reserved.
import os
from pathlib import Path
from loguru import logger

# Import Mineru's core parsing function
from mineru.cli.common import read_fn
from mineru.main_api import parse_doc

if __name__ == '__main__':
    # 1. Define input and output directories
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "pdfs")  # Create a 'pdfs' folder
    output_dir = os.path.join(__dir__, "output")
    
    # Ensure directories exist
    os.makedirs(pdf_files_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Find all PDF and image files to process
    doc_path_list = []
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    for doc_path in Path(pdf_files_dir).glob('*'):
        if doc_path.suffix.lower() in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    if not doc_path_list:
        logger.warning(f"No documents found in '{pdf_files_dir}'. Please add some PDFs to process.")
    else:
        # 3. Call the parsing function
        
        # Option 1: Use the 'pipeline' backend (Recommended for general use)
        logger.info("Starting parsing with the 'pipeline' backend...")
        parse_doc(
            path_list=doc_path_list, 
            output_dir=output_dir, 
            backend="pipeline",
            lang="en"  # Specify 'en' for English documents
        )
        logger.info(f"Pipeline processing complete. Check the '{output_dir}' directory.")

        # Option 2: Use a VLM backend (Requires GPU and `mineru[all]` installation)
        # logger.info("Starting parsing with the 'vlm-transformers' backend...")
        # parse_doc(
        #     path_list=doc_path_list, 
        #     output_dir=output_dir, 
        #     backend="vlm-transformers"
        # )
        # logger.info(f"VLM processing complete. Check the '{output_dir}' directory.")

        # Option 3: Connect to a running VLM server
        # logger.info("Starting parsing with the 'vlm-sglang-client' backend...")
        # parse_doc(
        #     path_list=doc_path_list,
        #     output_dir=output_dir,
        #     backend="vlm-sglang-client",
        #     server_url="http://127.0.0.1:30000"
        # )
        # logger.info(f"VLM client processing complete. Check the '{output_dir}' directory.")
```

## 4. Understanding the Output ##

When you run Mineru, it creates a subdirectory for each input file inside your specified output directory. This subfolder contains various artifacts that provide deep insight into the parsing process:

- **{filename}.md**: The final, clean Markdown representation of the document. This is the primary output for most use cases.
- **{filename}_middle.json**: A rich intermediate JSON file containing detailed information about every element, including text blocks, spans, fonts, bounding boxes, and structural roles (e.g., title, paragraph, figure_caption). This is extremely useful for developers building custom applications on top of Mineru's output.
- **{filename}_content_list.json**: A simplified JSON output that presents the document content as a clean list of strings and objects, preserving the reading order.
- **{filename}_layout.pdf**: A visual debugging file that draws colored bounding boxes around the detected layout elements (like paragraphs, tables, and figures) on top of the original PDF pages.
- **{filename}_span.pdf**: A more granular visualization showing bounding boxes for individual text spans.
- **{filename}_model.json (or _model_output.txt)**: The raw JSON or text output from the underlying layout/VLM models before post-processing. Useful for advanced debugging.
- **images folder**: Contains all the images extracted from the document.

## 5. Backends Explained ##

Mineru's flexibility comes from its support for different parsing backends.

### pipeline Backend ###

This is the default and most versatile backend. It uses a multi-stage process where different specialized models handle different tasks:

- **Layout Detection**: Identifies the overall structure and regions (text, tables, images).
- **OCR (if needed)**: Extracts text from scanned portions.
- **Table Recognition**: Parses the structure of tables.
- **Formula Recognition**: Converts mathematical formulas to LaTeX.
- **Post-processing**: Assembles all the pieces into a coherent, ordered output.

This backend offers fine-grained control via the `method` (auto/txt/ocr) and `lang` parameters.

### vlm- Backends ###

These backends leverage a single, powerful Vision Language Model (VLM) to perform end-to-end document understanding. The VLM processes page images and directly outputs the structured content. This approach can be faster and more robust for certain document types but requires more powerful hardware (GPU).

- **vlm-transformers**: A general implementation using the Hugging Face transformers library.
- **vlm-sglang-engine / vlm-sglang-client**: High-performance versions that use the sglang inference engine for significant speedups. The engine runs the model locally, while the client connects to a dedicated sglang server instance.