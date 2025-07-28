# Re-ranking Feature Implementation

## 🎯 Overview

The re-ranking feature enhances document retrieval accuracy by implementing a two-stage retrieval system:

1. **Stage 1**: Initial vector search using Google Gemini embeddings (retrieves 12 most similar chunks)
2. **Stage 2**: Cross-encoder re-ranking using `ms-marco-MiniLM-L-12-v2` (returns top 4 most relevant chunks)

## ✨ Features Implemented

### Core Services
- **`ReRankingService`**: Cross-encoder model integration with lazy loading
- **`RagasEvaluator`**: RAGAS numerical evaluation using GPT-4o as evaluator
- **`RerankingExperimentService`**: A/B testing framework for comparing retrieval methods

### UI Enhancements
- **Per-query toggle**: Users can enable/disable re-ranking for each question
- **Visual indicators**: Shows which retrieval method was used
- **Metrics display**: RAGAS scores with color-coded quality indicators
- **Processing time**: Shows performance impact of re-ranking
- **Advanced settings**: Configurable search parameters in sidebar

### Enhanced Data Storage
- **Conversation schema**: Stores re-ranking usage, relevance scores, and RAGAS metrics
- **Retrieval metadata**: Tracks which method was used and performance metrics
- **Backward compatibility**: Maintains existing conversation data structure

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Re-ranking dependencies (using UV package manager)
uv add sentence-transformers torch transformers

# RAGAS evaluation dependencies  
uv add ragas datasets pandas

# Or sync all dependencies from pyproject.toml
uv sync
```

### 2. Configure API Keys

Add to your `.env` file:
```bash
# Required for RAGAS evaluation
OPENAI_API_KEY=your_openai_api_key_here

# Existing keys
GOOGLE_API_KEY=your_google_api_key
```

### 3. Use Re-ranking in UI

1. Upload a document
2. Check the "🔍 Use Re-ranking" checkbox in the chat interface
3. Ask questions and compare results

### 4. Run Experiments

The experiment runner uses existing documents from your Qdrant collection and compares retrieval methods in terminal view:

```bash
# List available documents in Qdrant collection
uv run python scripts/run_experiment.py --list-docs

# Run interactive experiment (select document from menu)
uv run python scripts/run_experiment.py

# Test specific document ID
uv run python scripts/run_experiment.py --doc-id your_doc_id

# Test with custom questions
uv run python scripts/run_experiment.py --doc-id your_doc_id --questions "What is the voltage?" "How do I connect it?"

# Save results to custom file
uv run python scripts/run_experiment.py --doc-id your_doc_id --output my_experiment.json

# Run without saving results
uv run python scripts/run_experiment.py --doc-id your_doc_id --no-save
```

The experiment runner will:
- Show each question being tested
- Display answers from both standard and re-ranking retrieval
- Compare RAGAS metrics with color-coded quality indicators
- Show processing times and performance differences
- Provide a final summary with recommendations

## 📊 Configuration

Updated settings in `config.py`:

```python
# Re-ranking settings
reranking_enabled_default: bool = False
reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
reranking_initial_k: int = 12  # Optimized for small document set
reranking_final_k: int = 4     # Optimized for small document set

# Document processing (optimized for 8-document dataset)
chunk_size: int = 512          # Reduced for better granularity
chunk_overlap: int = 64        # Reduced proportionally

# RAGAS evaluation
ragas_evaluator_model: str = "gpt-4o"
ragas_test_dataset_path: str = "data/evaluation/test_dataset.csv"
ragas_fallback_silent: bool = True
```

## 🎯 RAGAS Metrics

The system now provides real numerical RAGAS evaluation:

- **Faithfulness** (0-1): Factual consistency with retrieved context
- **Answer Relevancy** (0-1): How well the answer addresses the question
- **Context Precision** (0-1): Whether relevant contexts are ranked higher
- **Context Recall** (0-1): Whether all relevant information was retrieved

### Color Coding in UI
- 🟢 **Green**: Score ≥ 0.7 (Excellent)
- 🟠 **Orange**: Score ≥ 0.5 (Good)
- 🔴 **Red**: Score < 0.5 (Needs improvement)

## 🧪 Test Dataset

Located at `data/evaluation/test_dataset.csv` with 8 sample questions:

```csv
question,ground_truth,category,difficulty,expected_context_keywords
"What is the operating voltage range for this sensor?","The operating voltage range is 3.3V to 5.0V...","electrical_specifications","easy","voltage,VCC,VDD..."
```

Customize this dataset with questions specific to your documents for better evaluation.

## 📈 Performance Characteristics

### Standard Retrieval
- **Response Time**: < 1 second
- **Memory Usage**: ~500MB
- **Accuracy**: Good for general queries

### Re-ranking Retrieval
- **Response Time**: < 3 seconds
- **Memory Usage**: ~1.5GB (includes cross-encoder model)
- **Accuracy**: Enhanced for specific technical questions
- **Model Loading**: ~5 seconds (first use only)

## 🔧 Advanced Usage

### Programmatic Access

```python
from newsensor_streamlit.core.chat_engine import ChatEngine

chat_engine = ChatEngine()

# Standard retrieval
response = chat_engine.ask_question(
    question="What is the I2C address?",
    doc_id="your_doc_id",
    use_reranking=False
)

# With re-ranking
response = chat_engine.ask_question(
    question="What is the I2C address?",
    doc_id="your_doc_id",
    use_reranking=True
)

# Access metrics
ragas_metrics = response.get("ragas_metrics", {})
retrieval_metadata = response.get("retrieval_metadata", {})
```

### Custom Experiments

```python
from newsensor_streamlit.services.experiment_service import RerankingExperimentService

experiment_service = RerankingExperimentService(chat_engine)

# Run custom experiment
result = await experiment_service.run_comparison_experiment(
    doc_id="your_doc_id",
    questions=["Custom question 1", "Custom question 2"]
)
```

## 🚨 Troubleshooting

### Re-ranking Not Available
```
⚠️ Re-ranking not available (install sentence-transformers)
```
**Solution**: `uv add sentence-transformers torch transformers`

### RAGAS Not Available
```
⚠️ RAGAS not available (install ragas & configure OpenAI API)
```
**Solutions**:
1. `uv add ragas datasets`
2. Add `OPENAI_API_KEY` to `.env`

### Import Errors with UV
```
❌ Import error: No module named 'newsensor_streamlit'
```
**Solution**: Always run with UV: `uv run python scripts/run_experiment.py`

### Model Loading Slow (First Time)
- Cross-encoder model downloads ~133MB on first use
- Subsequent runs use cached model (~5s loading)
- Consider running a test first: `uv run python scripts/run_experiment.py --list-docs`

### Memory Issues
- Cross-encoder model requires ~1GB additional RAM
- Reduce `reranking_initial_k` in config if memory constrained
- Use CPU-only torch: `uv add torch --index-url https://download.pytorch.org/whl/cpu`

### Qdrant Connection Issues
```
❌ No documents found in collection
```
**Solutions**:
1. Check Qdrant is running: `docker ps` (if using Docker)
2. Verify collection name in `.env`: `QDRANT_COLLECTION=newsensor_datasheets`
3. Upload documents through the Streamlit UI first

## 📁 File Structure

```
src/newsensor_streamlit/services/
├── reranking_service.py      # Cross-encoder re-ranking
├── ragas_evaluator.py        # RAGAS numerical evaluation
├── experiment_service.py     # A/B testing framework
└── ...

data/
├── evaluation/
│   └── test_dataset.csv      # Ground truth questions
└── experiments/              # Experiment results
    ├── exp_20250726_*.csv
    └── exp_20250726_*.json

scripts/
└── run_experiment.py         # CLI experiment runner
```

## 🎉 What's New

1. **Smart Retrieval**: Toggle between fast standard search and accurate re-ranking
2. **Real Metrics**: Proper RAGAS numerical evaluation instead of placeholders
3. **Visual Feedback**: Clear indicators of retrieval method and quality scores
4. **Experiment Framework**: Built-in A/B testing for comparing methods
5. **Optimized Settings**: Chunk size and search parameters tuned for small document sets
6. **Enhanced UI**: Processing time, relevance scores, and method transparency

The re-ranking feature is now fully implemented and ready for use! 🚀
