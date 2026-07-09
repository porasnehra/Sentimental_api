# 🔍 Semantic Document Search API

A local, natural-language document search backend built with **FastAPI**, **Sentence-Transformers**, and **FAISS**. It lets you index raw text documents and then search them using plain-English queries instead of exact keyword matches — powered by vector embeddings and similarity search.

---

## ✨ Features

- **Semantic indexing** — converts document text into dense vector embeddings using the `all-MiniLM-L6-v2` Sentence-Transformer model
- **Fast similarity search** — uses FAISS (`IndexFlatL2`) for efficient nearest-neighbor lookup over embeddings
- **Natural language queries** — ask questions in plain English and get back the most relevant indexed documents
- **Adjustable Top-K results** — control how many matches are returned per query
- **Metadata support** — attach a source tag (e.g. `wiki`, `pdf_doc`) to each indexed document
- **Simple Streamlit frontend** — index, search, and check system status from a browser UI
- **Lightweight & local** — no external vector database required, runs entirely on your machine

---

## 🖥️ Screenshots

### Index Documents
Paste raw text, tag it with source metadata, and submit it to the FAISS index.

![Index Documents screenshot](index-documents.jpg)

### Search Documents
Enter a natural language query and get back the most semantically relevant document(s), ranked by distance score.

![Search Documents screenshot](search-documents.jpg)

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Embedding Model | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (`faiss-cpu`) |
| Frontend | Streamlit |
| Data Validation | Pydantic |
| Server | Uvicorn |
| Language | Python |

---

## 📂 Project Structure

```
Sentimental_api/
├── main.py             # FastAPI backend — indexing, search, and status endpoints
├── Frontend.py          # Streamlit UI for interacting with the API
├── requirements.txt      # Python dependencies
└── screenshots/          # UI screenshots (used in this README)
```

---

## ⚙️ API Endpoints

### `GET /`
Health check — confirms the API is running.

### `POST /index`
Indexes one or more documents into the FAISS vector store.

**Request body:**
```json
{
  "documents": [
    {
      "text": "Capital of India is Delhi",
      "metadata": { "source": "wiki" }
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully indexed 1 documents.",
  "indexed_documents": [
    { "id": 0, "text": "Capital of India is Delhi..." }
  ]
}
```

### `POST /search`
Runs a semantic search against the indexed documents.

**Request body:**
```json
{
  "query": "What is the capital of India?",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "What is the capital of India?",
  "total_results_found": 1,
  "results": [
    {
      "document_id": 0,
      "text": "Capital of India is Delhi",
      "metadata": { "source": "wiki" },
      "score": 0.3173
    }
  ]
}
```

### `GET /status`
Returns the total number of indexed documents and the embedding dimension.

```json
{
  "total_indexed_documents": 1,
  "embedding_dimension": 384
}
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/porasnehra/Sentimental_api.git
cd Sentimental_api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI backend
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

### 4. Run the Streamlit frontend
```bash
streamlit run Frontend.py
```

---

## 🌐 Live Demo

The backend is deployed on Render: `https://sentimental-api-1.onrender.com`

> Note: free-tier Render instances spin down when idle, so the first request after inactivity may take a few seconds to respond.

---

## 🔮 Future Improvements

- Persist the FAISS index and document registry to disk (currently in-memory only)
- Support file uploads (PDF, TXT) instead of pasted text
- Add authentication for indexing endpoints
- Swap `IndexFlatL2` for an approximate index (e.g. `IndexIVFFlat`) for larger-scale search

---

## 👤 Author

**Poras Nehra**
- GitHub: [@porasnehra](https://github.com/porasnehra)

---

## 📄 License

This project is open source and available for learning and personal use.

