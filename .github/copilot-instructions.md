## Project Overview
Datasheet Query Engine for IoT makers and electronics engineers using Streamlit, LangChain, and Qdrant RAG system.

## Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Python with LangChain orchestration
- **Vector Database**: Qdrant for semantic search
- **LLM**: GPT-4o via OpenRouter
- **Embeddings**: Google Gemini API
- **PDF Processing**: MinerU library
- **Evaluation**: RAGAS metrics system

## Quick Start
```bash
# Clone and install
git clone <repo>
cd newsensor-streamlit
uv sync

cp .env.example .env
# Edit .env with your API keys

# Run
cd src
uv run streamlit run newsensor_streamlit/ui/app.py
```

## Development Commands
- **Install**: `uv add package-name`
- **Tests**: `uv run pytest`
- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .`
- **Type check**: `uv run pyright`
- **Pre-commit**: `uv run pre-commit run --all`

## Project Structure
```
src/newsensor_streamlit/
├── config.py              # Settings management
├── core/
│   └── chat_engine.py      # Main orchestration
├── services/
│   ├── document_processor.py  # PDF processing via MinerU
│   ├── embedding_service.py    # Google Gemini embeddings
│   ├── qdrant_service.py       # Vector database operations
│   └── rag_service.py        # RAG implementation
├── ui/
│   └── app.py              # Streamlit interface
└── __init__.py
```

## Key Components
- **ChatEngine**: Core orchestrator for document QA
- **DocumentProcessor**: PDF parsing with MinerU
- **QdrantService**: Vector storage and retrieval
- **EmbeddingService**: Google Gemini embeddings
- **RagService**: LLM integration and evaluation

## Environment Variables
See `.env.example` for required API keys and configuration.

## Testing
Tests use pytest with async support via anyio. Place tests in `tests/` directory.