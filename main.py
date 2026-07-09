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


model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()


index = faiss.IndexFlatL2(embedding_dim)

document_registry: Dict[int, Dict[str, str]] = {}
current_id = 0



class DocumentInput(BaseModel):
    text: str
    metadata: Dict[str, str] = {}

class BatchDocumentInput(BaseModel):
    documents: List[DocumentInput]

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3




@app.get("/")
def read_root():
    return {"message": "Local Semantic Search API is up and running!"}


@app.post("/index", status_code=status.HTTP_201_CREATED)
def index_documents(payload: BatchDocumentInput):
    
    global current_id
    if not payload.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    texts = [doc.text for doc in payload.documents]
    
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    
    embeddings = embeddings.astype('float32')
    
    
    index.add(embeddings)
    
    
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
    
    if index.ntotal == 0:
        return {"query": payload.query, "results": [], "message": "Index is empty. Please upload documents first."}

    
    query_vector = model.encode([payload.query], convert_to_numpy=True).astype('float32')

    
    k = min(payload.top_k, index.ntotal)
    distances, indices = index.search(query_vector, k)

    
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue  
        
        doc_data = document_registry.get(int(idx))
        if doc_data:
            results.append({
                "document_id": int(idx),
                "text": doc_data["text"],
                "metadata": doc_data["metadata"],
                "score": float(score)  
            })

    return {
        "query": payload.query,
        "total_results_found": len(results),
        "results": results
    }


@app.get("/status")
def get_status():
    
    return {
        "total_indexed_documents": index.ntotal,
        "embedding_dimension": embedding_dim
  }
  
