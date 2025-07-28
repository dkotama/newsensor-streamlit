# Re-ranking Feature Specification

## Version: 1.1 (Enhancement)

**Status**: Ready for Implementation  
**Date**: July 26, 2025  
**Parent Document**: prd.md v1.0

## 1. Goals

### 1.1. Feature Goal
To enhance the accuracy and relevance of retrieved documents in the RAG pipeline by implementing an optional cross-encoder re-ranking system that users can toggle per query. This will improve answer quality while maintaining transparency and user control over the retrieval process.

### 1.2. Problem Statement
The current bi-encoder semantic search (Google Gemini embeddings) provides good initial document retrieval but may not always rank the most relevant documents first. Cross-encoder models can provide more accurate relevance scoring by examining query-document pairs together, leading to better context for LLM generation.

### 1.3. Target User
Same as base system: IoT makers and electronics engineers who need highly accurate technical information from sensor datasheets.

### 1.4. Success Metrics
- Users can toggle re-ranking on/off for individual queries
- Re-ranking improves RAGAS metrics (Context Precision, Faithfulness) by measurable amounts
- Conversation data includes re-ranking usage information for analysis
- System performance remains acceptable (< 3 seconds total response time)

## 2. User Stories

**US-RE1 (Query-Level Control)**: As an IoT maker, I want to toggle re-ranking on/off for each individual question I ask, so I can compare answer quality and choose the best retrieval method for my needs.

**US-RE2 (Default Behavior)**: As an IoT maker, I want re-ranking to be disabled by default, so the system remains fast for routine queries unless I specifically request enhanced accuracy.

**US-RE3 (Transparency)**: As an IoT maker, I want to see whether re-ranking was used for each answer, along with relevance scores, so I can understand how the system retrieved information.

**US-RE4 (Performance Visibility)**: As an IoT maker, I want to see processing time for each query, so I can make informed decisions about when to use re-ranking vs. standard retrieval.

**US-RE5 (Data Tracking)**: As a product owner, I want all conversation data to include re-ranking usage information, so I can analyze the impact of re-ranking on answer quality and user behavior.

## 3. Scope (In Scope)

### 3.1. Frontend (Streamlit UI)

**Chat Interface Enhancements**:
- Toggle switch/checkbox in chat input area to enable/disable re-ranking per query
- Default state: re-ranking OFF
- Visual indicator showing whether re-ranking was used for each response
- Processing time display for each response
- Re-ranking score display in expandable section

**Settings Panel** (Sidebar):
- Advanced re-ranking configuration options
- Initial vector search limit slider (default: 12 chunks from Qdrant, optimized for small dataset)
- Final document count slider (default: 4 chunks after re-ranking) 
- Cross-encoder model selection (advanced users)

### 3.2. Backend Logic

**Re-ranking Service**:
- Cross-encoder model integration (`cross-encoder/ms-marco-MiniLM-L-12-v2`)
- Lazy loading of re-ranking model for performance
- Document scoring and ranking functionality
- Error handling with fallback to standard retrieval

**Enhanced Retrieval Pipeline**:
- Two-stage retrieval system:
  1. **Stage 1**: Google Gemini embeddings vector search in Qdrant (top_k=12 vector records/chunks, optimized for smaller document set)
  2. **Stage 2**: Cross-encoder scoring and re-ranking of those 12 chunks (return top 4 most relevant)
- Query-level toggle implementation
- Performance metrics collection

**Conversation Data Enhancement**:
- Re-ranking usage flag per message
- Relevance scores storage
- Processing time tracking
- Retrieval method metadata
- **Real RAGAS evaluation** using ChatGPT-4o as evaluator
- **Sample dataset integration** for ground truth comparisons
- **Graceful fallback** when RAGAS evaluation fails (no metrics displayed)

### 3.3. Data Storage

**Enhanced Conversation Schema**:
```json
{
  "message": {
    "timestamp": "ISO-8601",
    "question": "string",
    "answer": "string", 
    "context": "string",
    "sources": ["string"],
    "ragas_metrics": {
      "faithfulness": "float|null",        // 0-1: factual consistency with context
      "answer_relevancy": "float|null",    // 0-1: relevance to the question  
      "context_precision": "float|null",   // 0-1: relevant contexts ranked higher
      "context_recall": "float|null"       // 0-1: retrieval of all relevant info
    },
    "retrieval_metadata": {
      "reranking_enabled": "boolean",
      "reranking_model": "string|null",
      "initial_vector_search_count": "int",  // Number of chunks from Qdrant (e.g., 12)
      "final_chunk_count": "int",            // Number of chunks after re-ranking (e.g., 4)
      "processing_time": "float",
      "relevance_scores": ["float"],         // Cross-encoder scores for final chunks
      "best_relevance_score": "float"
    }
  }
}
```

## 4. Out of Scope

- **Automatic Re-ranking**: No intelligent auto-selection of when to use re-ranking
- **Multiple Re-ranking Models**: Single cross-encoder model only for MVP
- **Batch Re-ranking**: No bulk re-processing of existing conversations
- **Advanced Analytics**: No sophisticated re-ranking performance analysis dashboard
- **Custom Model Training**: No training of custom cross-encoder models

## 5. Technical Requirements

### 5.1. Additional Dependencies
- **sentence-transformers**: >=2.2.2 (cross-encoder models)
- **torch**: >=2.0.0 (PyTorch backend)
- **transformers**: >=4.21.0 (model support)
- **ragas**: >=0.1.0 (RAGAS evaluation framework with numerical metrics)
- **datasets**: >=2.14.0 (for RAGAS data handling)
- **pandas**: >=2.0.0 (for CSV dataset handling)

### 5.2. Infrastructure
- **Model Storage**: Local caching of cross-encoder model (~400MB)
- **Memory Requirements**: Additional ~1GB RAM for model loading
- **Processing**: CPU-based inference (GPU optional for performance)

### 5.3. Integration Points
- **Existing Embedding Service**: No changes required
- **Qdrant Service**: Enhanced retriever wrapper
- **Chat Engine**: Modified to support query-level toggle
- **Conversation Service**: Extended schema for re-ranking metadata
- **RAGAS Service**: Real evaluation using ChatGPT-4o as judge model
- **Sample Dataset**: Ground truth answers for evaluation comparison

## 6. Implementation Details

### 6.1. Configuration Changes
```python
class Settings:
    # Re-ranking settings
    reranking_enabled_default: bool = False
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranking_initial_k: int = 12  # Reduced for smaller document set
    reranking_final_k: int = 4     # Reduced for smaller document set
    reranking_timeout: int = 30  # seconds
    
    # Document processing settings (optimized for small dataset)
    chunk_size: int = 512          # Reduced chunk size for better granularity
    chunk_overlap: int = 64        # Reduced overlap proportionally
    
    # RAGAS evaluation settings
    ragas_evaluator_model: str = "gpt-4o"
    ragas_test_dataset_path: str = "data/evaluation/test_dataset.csv"
    ragas_fallback_silent: bool = True  # Don't show metrics if evaluation fails
```

### 6.2. API Changes
```python
def ask_question(
    self, 
    question: str, 
    doc_id: str, 
    conversation_id: str = None,
    use_reranking: bool = False  # New parameter
) -> dict:
```

### 6.3. Performance Targets
- **Standard Retrieval**: < 1 second
- **Re-ranking Retrieval**: < 3 seconds
- **Model Loading**: < 5 seconds (first use only)
- **Memory Usage**: < 2GB total

### 6.4. RAGAS Evaluation Implementation
```python
from ragas.metrics import (
    answer_relevancy,
    faithfulness, 
    context_recall,
    context_precision
)
from ragas import evaluate
from datasets import Dataset
import pandas as pd

class RagasEvaluator:
    def __init__(self):
        self.evaluator_llm = ChatOpenAI(model="gpt-4o")
        self.test_dataset = self._load_test_dataset()
    
    def evaluate_answer(self, question: str, answer: str, contexts: list, ground_truth: str = None) -> dict:
        """Evaluate answer using RAGAS numerical metrics (0-1 scores)."""
        try:
            # Prepare data for RAGAS
            data = {
                'question': [question],
                'answer': [answer], 
                'contexts': [contexts],  # List of retrieved context chunks
                'ground_truth': [ground_truth] if ground_truth else [answer]
            }
            
            dataset = Dataset.from_dict(data)
            
            # Evaluate with RAGAS metrics
            result = evaluate(
                dataset,
                llm=self.evaluator_llm,
                metrics=[
                    faithfulness,        # 0-1: factual consistency with context
                    answer_relevancy,    # 0-1: relevance to the question
                    context_precision,   # 0-1: relevant contexts ranked higher
                    context_recall       # 0-1: retrieval of all relevant info
                ]
            )
            
            # Extract scores
            scores = result.to_pandas().iloc[0]
            return {
                "faithfulness": round(float(scores['faithfulness']), 3),
                "answer_relevancy": round(float(scores['answer_relevancy']), 3), 
                "context_precision": round(float(scores['context_precision']), 3),
                "context_recall": round(float(scores['context_recall']), 3)
            }
            
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            return {}  # Return empty dict - UI will not show metrics
    
    def _load_test_dataset(self) -> pd.DataFrame:
        """Load test dataset from CSV."""
        try:
            return pd.read_csv(settings.ragas_test_dataset_path)
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
            return pd.DataFrame()
```

### 6.5. Test Dataset Structure (CSV)
```csv
question,ground_truth,category,difficulty,expected_context_keywords
"What is the operating voltage range for this sensor?","The operating voltage range is 3.3V to 5.0V as specified in the electrical characteristics section","electrical_specifications","easy","voltage,VCC,VDD,operating,supply,3.3V,5.0V"
"What is the I2C address of the sensor?","The default I2C address is 0x76. An alternative address 0x77 can be selected by connecting the SDO pin to VCC","communication_interface","easy","I2C,address,0x76,0x77,SDO,slave"
"How do I configure the sensor for continuous measurement mode?","Set the MODE bits in CTRL_REG1 register to 11 binary to enable continuous measurement mode","register_configuration","medium","continuous,measurement,mode,register,CTRL_REG1,MODE,bits"
"What are the pin connections needed for Arduino?","Connect VCC to 3.3V or 5V, GND to ground, SDA to A4 pin, and SCL to A5 pin on Arduino Uno","integration_guidance","medium","pin,connection,Arduino,SDA,SCL,VCC,GND,A4,A5"
"What is the measurement accuracy of this sensor?","Temperature accuracy is ±0.5°C and humidity accuracy is ±3% RH at 25°C room temperature","performance_characteristics","easy","accuracy,temperature,humidity,±0.5°C,±3%,RH,25°C"
"How do I read sensor data via I2C?","Send I2C start condition, write device address, write register address, read data bytes, then send stop condition","communication_interface","medium","I2C,read,data,start,stop,condition,device,address,register"
"What is the power consumption in sleep mode?","The sensor consumes 0.1µA in sleep mode with 2ms wake-up time","electrical_specifications","easy","power,consumption,sleep,mode,0.1µA,wake-up,2ms"
"How do I calibrate the sensor?","Perform two-point calibration using known reference values and store calibration coefficients in EEPROM","register_configuration","hard","calibrate,calibration,two-point,reference,coefficients,EEPROM"
```

## 7. Testing Strategy

### 7.1. A/B Testing
- Same questions with/without re-ranking
- RAGAS metrics comparison using discrete pass/fail evaluation
- User preference tracking

### 7.2. Experiment-Driven Evaluation
```python
@experiment()
async def compare_retrieval_methods(row):
    """Compare standard vs re-ranking retrieval."""
    # Standard retrieval
    standard_response = chat_engine.ask_question(
        row["question"], use_reranking=False
    )
    
    # Re-ranking retrieval  
    reranked_response = chat_engine.ask_question(
        row["question"], use_reranking=True
    )
    
    # Evaluate both
    standard_score = ragas_evaluator.evaluate_answer(
        row["question"], standard_response["answer"], row["grading_notes"]
    )
    
    reranked_score = ragas_evaluator.evaluate_answer(
        row["question"], reranked_response["answer"], row["grading_notes"]
    )
    
    return {
        **row,
        "standard_answer": standard_response["answer"],
        "reranked_answer": reranked_response["answer"], 
        "standard_correctness": standard_score.get("correctness", "fail"),
        "reranked_correctness": reranked_score.get("correctness", "fail"),
        "standard_relevance": standard_score.get("relevance", "fail"),
        "reranked_relevance": reranked_score.get("relevance", "fail"),
        "improvement": "yes" if reranked_score > standard_score else "no"
    }
```

### 7.3. Performance Testing
- Response time measurement
- Memory usage monitoring
- Error rate tracking

### 7.4. Quality Metrics
- Correctness improvement (pass/fail comparison)
- Answer relevance enhancement (pass/fail comparison)
- User satisfaction scores
- **Discrete RAGAS metrics comparison** (standard vs re-ranking)
- **Experiment results analysis** using CSV outputs

## 8. Migration Plan

### 8.1. Backward Compatibility
- Existing conversations remain unaffected
- Default behavior maintains current performance
- Optional feature activation

### 8.2. Rollout Strategy
1. **Phase 1**: Backend implementation with feature flag
2. **Phase 2**: UI integration with toggle
3. **Phase 3**: Performance optimization
4. **Phase 4**: User testing and feedback collection

## 9. Success Criteria

### 9.1. Functional Success
- ✅ Users can toggle re-ranking per query
- ✅ Default behavior is standard retrieval
- ✅ All conversation data includes re-ranking metadata
- ✅ System performance meets targets


