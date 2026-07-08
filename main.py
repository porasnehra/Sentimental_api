from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict

app = FastAPI(
    title="Local Semantic Search API",
    description="A local natural language search backend using FastAPI, Sentence-Transformers, and FAISS.",
    version="1.0.0"
)

# 1. Initialize the Open-Source Embedding Model
# 'all-MiniLM-L6-v2' is fast, lightweight, and perfect for local deployment.
print("Loading local embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()

# 2. Initialize the FAISS Index for efficient vector similarity search
# IndexFlatL2 uses Euclidean distance (L2), which works well with normalized embeddings.
index = faiss.IndexFlatL2(embedding_dim)

# In-memory document registry to map FAISS vector IDs back to actual document text
document_registry: Dict[int, Dict[str, str]] = {}
current_id = 0


# --- Pydantic Schemas ---
class DocumentInput(BaseModel):
    text: str
    metadata: Dict[str, str] = {}

class BatchDocumentInput(BaseModel):
    documents: List[DocumentInput]

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Local Semantic Search API is up and running!"}


@app.post("/index", status_code=status.HTTP_201_CREATED)
def index_documents(payload: BatchDocumentInput):
    """
    Extracts text, generates vector embeddings, and indexes documents into FAISS.
    """
    global current_id
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    texts = [doc.text for doc in payload.documents]
    
    # Generate embeddings locally using the NLP model
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # FAISS requires float32 numpy arrays
    embeddings = embeddings.astype('float32')
    
    # Add vectors to the FAISS index
    index.add(embeddings)
    
    # Store text and metadata in our local registry mapped to the vector IDs
    indexed_docs = []
    for doc in payload.documents:
        document_registry[current_id] = {
            "text": doc.text,
            "metadata": doc.metadata
        }
        indexed_docs.append({"id": current_id, "text": doc.text[:50] + "..."})
        current_id += 1

    return {
        "status": "success",
        "message": f"Successfully indexed {len(payload.documents)} documents.",
        "indexed_documents": indexed_docs
    }


@app.post("/search")
def search_documents(payload: SearchQuery):
    """
    Performs a context-aware semantic search using the input query string.
    """
    if index.ntotal == 0:
        return {"query": payload.query, "results": [], "message": "Index is empty. Please upload documents first."}

    # 1. Embed the incoming search query
    query_vector = model.encode([payload.query], convert_to_numpy=True).astype('float32')

    # 2. Query FAISS for the nearest vectors
    # k must not exceed the total number of indexed items
    k = min(payload.top_k, index.ntotal)
    distances, indices = index.search(query_vector, k)

    # 3. Compile the ranked results
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 if fewer items exist than top_k requested
        
        doc_data = document_registry.get(int(idx))
        if doc_data:
            results.append({
                "document_id": int(idx),
                "text": doc_data["text"],
                "metadata": doc_data["metadata"],
                "score": float(score)  # Lower score = closer match in L2 distance
            })

    return {
        "query": payload.query,
        "total_results_found": len(results),
        "results": results
    }


@app.get("/status")
def get_status():
    """
    Returns the total number of documents currently indexed.
    """
    return {
        "total_indexed_documents": index.ntotal,
        "embedding_dimension": embedding_dim
  }
  
