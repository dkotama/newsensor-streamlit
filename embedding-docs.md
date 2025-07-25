### Google AI: Text Embeddings and Vector Search with Qdrant ###

## 1. Introduction to Google's Text Embeddings ##

Google's Gemini API provides state-of-the-art text embedding models, such as `gemini-embedding-001`, designed to convert text (words, sentences, or entire documents) into numerical vector representations. These embeddings capture the semantic meaning and context of the text, enabling powerful applications far beyond simple keyword matching.

### Common Use Cases ###

- **Semantic Search**: Finding documents based on meaning, not just keywords.
- **Retrieval-Augmented Generation (RAG)**: Enhancing LLM responses by providing relevant context retrieved from a knowledge base.
- **Classification**: Categorizing text into predefined labels (e.g., sentiment analysis).
- **Clustering**: Grouping similar documents together.

## 2. Generating Embeddings ##

You can generate embeddings using the `embedContent` method in the `google-genai` Python library.

### Generate a Single Embedding ###

```python
from google import genai

# Configure with your API key
# genai.configure(api_key="YOUR_API_KEY")

client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is the meaning of life?"
)

# The embedding is a list of floating-point numbers
print(result.embeddings[0].values[:10])  # Print first 10 dimensions
```

### Generate Batch Embeddings ###

For efficiency, you can process multiple text chunks in a single API call by passing a list of strings.

```python
from google import genai

client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[
        "What is the meaning of life?",
        "How do I bake a cake?",
        "What is the best sci-fi movie?"
    ]
)

for embedding in result.embeddings:
    print(embedding.values[:10])  # Print first 10 dimensions of each embedding
```

## 3. Optimizing Embedding Performance ##

### Specifying Task Type ###

You can significantly improve the quality of your embeddings for a specific task by providing the `task_type` parameter. This optimizes the vectors for the intended use case.

| Task Type             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| RETRIEVAL_QUERY       | The text is a query used to find relevant documents.                        |
| RETRIEVAL_DOCUMENT    | The text is a document to be indexed and retrieved.                         |
| SEMANTIC_SIMILARITY   | The embeddings will be used to compare semantic similarity.                 |
| CLASSIFICATION        | The embeddings will be used for classifying text.                           |
| CLUSTERING            | The embeddings will be used for clustering similar texts.                   |
| QUESTION_ANSWERING    | The text is a question in a Q&A system.                                    |

#### Example: Semantic Similarity ####

```python
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client()

texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?"
]

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

# Calculate and print cosine similarity
embeddings_matrix = np.array([e.values for e in result.embeddings])
similarity_matrix = cosine_similarity(embeddings_matrix)
print(similarity_matrix)
```

### Controlling Embedding Size ###

The `gemini-embedding-001` model uses Matryoshka Representation Learning (MRL), which means you can truncate the full-length (3072 dimensions) embedding to a smaller size (e.g., 768) with minimal loss in quality. This is excellent for saving storage space and improving computational speed.

```python
from google import genai
from google.genai import types

client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is the meaning of life?",
    config=types.EmbedContentConfig(output_dimensionality=768)
)

embedding_obj = result.embeddings[0]
print(f"Length of embedding: {len(embedding_obj.values)}")
# Output: Length of embedding: 768
```

**Note on Normalization**: The full 3072-dimension embedding is pre-normalized. If you use a smaller dimensionality, you should normalize the vector yourself before calculating cosine similarity.

## 4. Integration with Qdrant: Movie Recommendation System ##

This example demonstrates how to build a movie recommendation system by creating embeddings for movie metadata and using Qdrant, a vector database, for efficient similarity search.

### Step 1: Prepare and Clean Data ###

First, load a movie dataset and clean it, keeping only relevant fields like title, overview, genres, and release_date. Combine these fields into a single text block for each movie to create a rich source for embeddings.

```python
import pandas as pd

# Assume df_relevant is a DataFrame with movie data
def create_embedding_text(row):
    """Combines movie metadata into a single string."""
    title_str = f"Title: {row['title']}"
    overview_str = f"Overview: {row['overview']}" if row['overview'] else ""
    genre_str = f"Genres: {row['genres']}" if row['genres'] else ""
    
    parts = [title_str, overview_str, genre_str]
    return "\\n".join(part for part in parts if part)

df_relevant['text_for_embedding'] = df_relevant.apply(create_embedding_text, axis=1)
```

### Step 2: Generate Embeddings and Index in Qdrant ###

Initialize the Qdrant client and create a collection. Then, process the movie data in batches: generate embeddings using Gemini and "upsert" them into the Qdrant collection. Each point in Qdrant consists of a unique ID, its vector embedding, and a "payload" of its original metadata.

```python
from qdrant_client import QdrantClient, models

# Use in-memory storage for this example
qdrant_client = QdrantClient(":memory:")

COLLECTION_NAME = "tmdb_movies"
# Vector size for gemini-embedding-001 is 3072
VECTOR_SIZE = 3072

qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
)

# Get embeddings in batches
# Note: make_embed_text_fn should call the Gemini API
# 'documents' is a list of dictionaries, each with movie text and metadata
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=idx,
            vector=make_embed_text_fn(doc["content"], task_type="RETRIEVAL_DOCUMENT"),
            payload=doc  # Store original content as payload
        )
        for idx, doc in enumerate(documents)
    ]
)
```

### Step 3: Query for Recommendations ###

To find similar movies, embed a user's query using the `RETRIEVAL_QUERY` task type and use it to search the Qdrant collection. Qdrant will return the most similar movies based on cosine similarity.

```python
def recommend_movies(query_text, top_k=5):
    """Finds movies similar to the query."""
    
    # Generate embedding for the user query
    query_embedding = make_embed_text_fn(query_text, task_type="RETRIEVAL_QUERY")

    if query_embedding is None:
        return []

    # Perform a semantic search on Qdrant
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    return search_result

# Get recommendations
query = "A mind-bending sci-fi movie about dreams and reality"
recommendations = recommend_movies(query)

for hit in recommendations:
    print(f"Score: {hit.score:.4f} - Title: {hit.payload.get('title')}")
```

This workflow demonstrates a complete RAG-like pipeline where Gemini provides the semantic understanding (embeddings) and Qdrant provides the efficient retrieval mechanism.    