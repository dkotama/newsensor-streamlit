### Datasheet Query Engine (MVP) ###

## Version: 1.0 (MVP) ##

**Status**: Scoping Finalized  
**Date**: July 25, 2025  

### 1. Goals ###

#### 1.1. Product Goal ####
To provide IoT makers and electronics engineers with a tool that accelerates their research and development process. The engine allows users to have a natural language conversation with technical sensor datasheets, receiving reliable and verifiably accurate answers to their questions.

#### 1.2. Problem Statement ####
Searching for specific technical information (e.g., pinouts, voltage ratings, register values) within dense, lengthy PDF datasheets is a time-consuming and error-prone process. This tool aims to eliminate manual searching and provide instant, trustworthy answers.

#### 1.3. Target User ####
IoT Makers and hardware engineers who need to quickly understand a sensor's specifications and how to integrate it with microcontroller boards (e.g., Arduino, ESP32).

#### 1.4. Success Metrics for MVP ####
- A user can successfully upload a PDF and ask a question.
- The system generates a relevant answer based on the document's content.
- For every answer, the four specified RAGAS quality metrics are calculated and displayed.
- The entire conversation turn (question, answer, context, scores) is successfully saved to the backend database.

### 2. User Stories ###

**US-1 (File Upload)**: As an IoT maker, I want to upload a single sensor datasheet PDF to initiate a new, focused conversation session.  

**US-2 (Interaction)**: As an IoT maker, I want to ask questions about technical specifications and board integration in a simple chat interface.  

**US-3 (Trust & Verification)**: As an IoT maker, I want to see a detailed quality report next to every answer, showing scores for Faithfulness, Answer Relevancy, Context Precision, and Context Recall, so I can immediately assess the information's reliability.  

**US-4 (Session Context)**: As an IoT maker, I want my current conversation history (Q&A) to be visible on the screen so I can easily reference my previous questions.  

**US-5 (Data Persistence)**: As the product owner, I want every conversation turn and its corresponding RAGAS scores to be saved to a database for future benchmarking and system improvement.

### 3. Scope (In Scope for MVP) ###

#### 3.1. Frontend (Streamlit UI) ####
- A file uploader that accepts a single PDF file per session.
- A chat display area that shows the history of the current session.
- A text input box for the user to type questions.
- A display area next to each generated answer to show the four RAGAS metric scores.

#### 3.2. Backend Logic ####
**Data Ingestion**:  
- On PDF upload, the system will process the document through a modular pipeline.  
- **PDF Parsing**: A dedicated module will parse the PDF. The initial implementation will use the MinerU library.  
- **Text Splitting**: The parsed text will be split into chunks using LangChain's RecursiveCharacterTextSplitter.  
- **Embedding & Storage**: Chunks will be converted to vectors using the Gemini embedding model and stored in a Qdrant vector database.  

**Query & Response**:  
- User questions will be embedded using the same Gemini model.  
- Qdrant will be queried to retrieve the most relevant context chunks.  
- The context and question will be sent to an LLM (GPT-4o via OpenRouter) to generate an answer.  
- The generated answer will include necessary code comments if code is generated.  

**Evaluation & Persistence**:  
- Each answer will be evaluated by RAGAS against the retrieved context to generate scores for Faithfulness, Answer Relevancy, Context Precision, and Context Recall.  
- The question, answer, context, and the four RAGAS scores will be saved as a single record in the Qdrant database.

### 4. Out of Scope ###

- **User Accounts**: No user login, profiles, or authentication.  
- **Persistent Chat History**: The UI will not support loading or browsing chats from previous sessions. Chat history is cleared on page refresh.  
- **Multi-Document Interaction**: The user can only interact with one PDF at a time.  
- **Advanced UI**: No complex UI features beyond a simple chat interface.  
- **Testing and Debugging**: A dedicated testing suite and interactive debugging functionalities for the generated code are not part of the MVP.

### 5. Technical Requirements ###

- **Frontend**: Streamlit  
- **Orchestration Framework**: LangChain  
- **LLM**: GPT-4o (or other compatible models via OpenRouter)  
- **Embedding Model**: Google Gemini API (text-embedding-004 or newer)  
- **Vector Database**: Self-hosted Qdrant  
- **PDF Parsing Module**: MinerU library  
- **Text Splitting**: langchain.text_splitter.RecursiveCharacterTextSplitter  
- **Evaluation Framework**: RAGAS