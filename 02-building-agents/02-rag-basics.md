# RAG Basics

## Why RAG

LLMs have a training cutoff. They do not know your company's internal documents, today's news, or your codebase. RAG (Retrieval-Augmented Generation) gives the LLM access to external knowledge by retrieving relevant documents and including them in the prompt.

Without RAG: LLM answers from training data. May be wrong, outdated, or irrelevant.

With RAG: LLM answers from your specific documents. Grounded, current, and verifiable.

## The RAG Pipeline

```
Documents -> Chunk -> Embed -> Store in Vector DB
                                          |
Query -> Embed -> Search Vector DB -> Retrieve Top K Chunks
                                          |
                          Retrieved Chunks + Query -> LLM -> Response
```

Four stages: ingest, index, retrieve, generate.

> 🖼️ **[IMAGE_PLACEHOLDER]** — RAG pipeline chunk embed store vector DB retrieve generate

## Stage 1: Ingest (Chunk Documents)

Documents are too large to embed whole. Split them into chunks.

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks.

    chunk_size: target tokens per chunk (roughly characters / 4)
    overlap: characters to overlap between chunks (preserves context at boundaries)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

# Better: chunk by structure (paragraphs, sections, headers)
def chunk_by_structure(markdown_text: str) -> list[dict]:
    """Split markdown by headers. Each section is a chunk."""
    chunks = []
    current_header = ""
    current_content = []

    for line in markdown_text.split("\n"):
        if line.startswith("#"):
            if current_content:
                chunks.append({
                    "header": current_header,
                    "content": "\n".join(current_content).strip()
                })
            current_header = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        chunks.append({
            "header": current_header,
            "content": "\n".join(current_content).strip()
        })

    return chunks
```

Chunking strategy matters. Fixed-size chunks are simple but split mid-sentence. Structure-based chunks preserve semantics but vary in size. Choose based on your content.

> 🖼️ **[IMAGE_PLACEHOLDER]** — document chunking strategies fixed-size overlap vs structural headers

## Stage 2: Index (Embed and Store)

Convert chunks to vector embeddings and store in a vector database.

```python
from openai import OpenAI

client = OpenAI()

def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

# Simple in-memory vector store for demonstration
import numpy as np

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.documents = []

    def add(self, texts: list[str]):
        embeddings = embed_texts(texts)
        for text, embedding in zip(texts, embeddings):
            self.vectors.append(embedding)
            self.documents.append(text)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = embed_texts([query])[0]
        similarities = [
            np.dot(query_embedding, vec) / (np.linalg.norm(query_embedding) * np.linalg.norm(vec))
            for vec in self.vectors
        ]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [
            {"text": self.documents[i], "score": float(similarities[i])}
            for i in top_indices
        ]

# Usage
store = SimpleVectorStore()
documents = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLM generation.",
    "Vector databases store data as high-dimensional vectors for similarity search.",
    "LangChain is a framework for building LLM-powered applications.",
    "Embeddings are dense vector representations of text that capture semantic meaning.",
]
store.add(documents)
```

In production, use a real vector database (Pinecone, Weaviate, Qdrant, pgvector). The in-memory store above is for understanding the concept.

## Stage 3: Retrieve

Given a user query, find the most relevant chunks.

```python
def retrieve(query: str, store: SimpleVectorStore, top_k: int = 3) -> list[str]:
    """Retrieve relevant document chunks for a query."""
    results = store.search(query, top_k=top_k)
    return [r["text"] for r in results]

# Example
chunks = retrieve("What is RAG?", store)
# Returns the chunk about RAG definition
```

Retrieval quality depends on: chunking strategy, embedding model, and query formulation. If the right chunk is not retrieved, the generation will be wrong regardless of how good the LLM is.

## Stage 4: Generate

Feed retrieved chunks + query to the LLM.

```python
def generate(query: str, context_chunks: list[str]) -> str:
    """Generate a response using retrieved context."""
    context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Answer the user's question using only the provided context.
If the context does not contain enough information, say "I don't have enough information to answer that."
Always cite the source by referencing the number in brackets, e.g. [1]."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

# Full RAG pipeline
def rag_query(query: str, store: SimpleVectorStore) -> str:
    chunks = retrieve(query, store)
    return generate(query, chunks)

answer = rag_query("What is RAG?", store)
print(answer)
# "RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLM generation. [2]"
```

## Failure Modes

- **Retrieval misses**: The right chunk was not in the top-k results. Mitigate with hybrid search (semantic + keyword), re-ranking, and larger top-k values.
- **Stale data**: The vector store is not updated when documents change. Mitigate with incremental indexing and document versioning.
- **Context overload**: Too many retrieved chunks fill the context window. Mitigate with relevance thresholds and smart truncation.
- **Hallucination despite context**: The LLM ignores the retrieved context and answers from training data. Mitigate with strong system instructions ("use ONLY the provided context") and citation requirements.
