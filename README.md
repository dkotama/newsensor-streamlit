# Newsensor Streamlit - Datasheet Query Engine

A powerful datasheet query engine for IoT makers and electronics engineers, built with Streamlit and powered by AI.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Docker Setup](#docker-setup)
- [Development](#development)
- [Project Structure](#project-structure)

## ✨ Features

- 🔍 **Intelligent Datasheet Search**: Query sensor datasheets using natural language
- 🤖 **AI-Powered Responses**: Integration with Google Gemini, OpenAI, and OpenRouter
- 📊 **Vector Database**: Powered by Qdrant for semantic search
- 📁 **Document Processing**: Support for PDF parsing with LlamaParse
- 🎯 **Re-ranking**: Advanced result re-ranking with sentence transformers
- 💬 **Conversation Management**: Persistent chat history
- 📈 **Evaluation**: RAGAs integration for quality assessment

## 🔧 Prerequisites

- Python 3.11 or higher
- [UV](https://github.com/astral-sh/uv) package manager
- Docker (optional, for Qdrant)

### Install UV

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dkotama/newsensor-streamlit.git
   cd newsensor-streamlit
   ```

2. **Install dependencies using UV**
   ```bash
   uv sync
   ```

   This will:
   - Create a virtual environment at `.venv`
   - Install all project dependencies
   - Install the project in editable mode

3. **Optional: Install development dependencies**
   ```bash
   uv sync --extra dev
   ```

## ⚙️ Configuration

1. **Create environment file**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your API keys**
   ```bash
   # API Keys
   google_api_key=your_google_api_key_here
   openai_api_key=your_openai_api_key_here
   openrouter_api_key=your_openrouter_api_key_here
   llama_parse_api_key=your_llama_parse_api_key_here

   # Database
   qdrant_host=localhost
   qdrant_port=6333

   # Optional
   chunk_size=1000
   chunk_overlap=200
   max_context_chunks=5
   ```

## 🏃 Running the Application

### Method 1: Using UV (Recommended)

```bash
uv run streamlit run src/newsensor_streamlit/ui/app.py
```

### Method 2: Using the project script

```bash
uv run newsensor
```

### Method 3: Direct Python execution

```bash
# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows

# Run the application
python -m newsensor_streamlit
```

The application will be available at `http://localhost:8501`

## 🐳 Docker Setup

### Start Qdrant Database

```bash
# Start Qdrant using Docker Compose
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333/health
```

### Stop Services

```bash
docker-compose down
```

## 🛠️ Development

### Install Development Dependencies

```bash
uv sync --extra dev
```

### Run Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run ruff format .
```

### Code Linting

```bash
uv run ruff check .
```

### Pre-commit Hooks

```bash
uv run pre-commit install
```

## 📁 Project Structure

```
newsensor-streamlit/
├── src/newsensor_streamlit/
│   ├── __init__.py
│   ├── __main__.py              # Entry point
│   ├── config.py                # Configuration settings
│   ├── core/
│   │   └── chat_engine.py       # Main chat engine
│   ├── services/                # Service layer
│   │   ├── conversation_service.py
│   │   ├── document_processor.py
│   │   ├── embedding_service.py
│   │   ├── llama_parse_service.py
│   │   ├── qdrant_service.py
│   │   ├── rag_service.py
│   │   ├── ragas_evaluator.py
│   │   └── reranking_service.py
│   └── ui/
│       └── app.py               # Streamlit UI
├── tests/                       # Test files
├── data/                        # Data storage
│   ├── conversations/           # Chat history
│   └── evaluation/              # Evaluation datasets
├── scripts/                     # Utility scripts
├── docker-compose.yml           # Docker services
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
└── README.md                   # This file
```

## 🔍 Usage Tips

1. **First Run**: The application will create necessary directories and initialize the Qdrant collection
2. **Document Upload**: Use the sidebar to upload PDF datasheets
3. **Querying**: Ask natural language questions about your uploaded documents
4. **Conversations**: Chat history is automatically saved and can be resumed
5. **Settings**: Adjust model parameters and search settings in the sidebar

## 🐛 Troubleshooting

### Common Issues

1. **Browser auto-open error**: Add `--server.headless=true` flag
   ```bash
   uv run streamlit run src/newsensor_streamlit/ui/app.py --server.headless=true
   ```

2. **Qdrant connection error**: Ensure Qdrant is running
   ```bash
   docker-compose up -d
   ```

3. **Missing API keys**: Check your `.env` file configuration

4. **Port already in use**: Change the port
   ```bash
   uv run streamlit run src/newsensor_streamlit/ui/app.py --server.port=8502
   ```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📧 Support

For questions and support, please open an issue on GitHub.
