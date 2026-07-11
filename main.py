from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict
import pypdf  # Library to read PDF files

# ==========================================
# 1. API Application Setup
# ==========================================
app = FastAPI(
    title="Friendly PDF Semantic Search API",
    description="An AI search engine that reads uploaded PDFs and searches their contents semantically.",
    version="2.0.0"
)

# ==========================================
# 2. AI Model & Database Initialization
# ==========================================
print("Waking up the AI model... (This might take a few seconds)")

ai_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_size = ai_model.get_sentence_embedding_dimension()

# Set up FAISS vector database
vector_database = faiss.IndexFlatL2(vector_size)

# Maps a vector ID back to the exact PDF name, page number, and text content
text_storage: Dict[int, Dict[str, any]] = {}
next_available_id = 0


# ==========================================
# 3. Data Blueprints (What the API expects)
# ==========================================
class SearchRequest(BaseModel):
    query: str
    max_results: int = 3


# ==========================================
# 4. The API Endpoints
# ==========================================

@app.get("/")
def check_health():
    return {"message": "The PDF Semantic Search API is online and ready!"}


@app.post("/index-pdf", status_code=status.HTTP_201_CREATED)
async def save_pdf_document(file: UploadFile = File(...)):
    """Accepts a PDF file upload, extracts text page-by-page, and indexes it."""
    global next_available_id

    # Safety check: Is it actually a PDF?
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .pdf file.")

    try:
        # Step 1: Read the uploaded file into memory using pypdf
        pdf_reader = pypdf.PdfReader(file.file)
        total_pages = len(pdf_reader.pages)
        
        pages_indexed = []

        # Step 2: Loop through the PDF page by page
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            # Skip pages that have completely empty text (like images or blank sheets)
            if not page_text.strip():
                continue
            
            # Step 3: Convert this specific page's text into an AI math vector
            number_vector = ai_model.encode([page_text], convert_to_numpy=True)
            formatted_vector = number_vectors = number_vector.astype('float32')
            
            # Step 4: Add the vector to the database
            vector_database.add(formatted_vector)
            
            # Step 5: Save the text, filename, and page tracking details into storage
            text_storage[next_available_id] = {
                "filename": file.filename,
                "page_number": page_num + 1,  # human readable page count (1-indexed)
                "text": page_text
            }
            
            pages_indexed.append({
                "vector_id": next_available_id,
                "page": page_num + 1
            })
            
            next_available_id += 1

        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}'. Indexed {len(pages_indexed)} out of {total_pages} pages.",
            "details": pages_indexed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file. Error: {str(e)}")


@app.post("/search")
def find_similar_pdf_pages(payload: SearchRequest):
    """Searches through all stored PDF pages to find the most contextually relevant matches."""
    
    if vector_database.ntotal == 0:
        return {
            "message": "The database is empty. Please upload a PDF file first!",
            "results": []
        }

    # Step 1: Translate the user's search query into a math vector
    query_vector = ai_model.encode([payload.query], convert_to_numpy=True).astype('float32')

    # Step 2: Ask FAISS to find the closest page vectors
    results_to_fetch = min(payload.max_results, vector_database.ntotal)
    distances, closest_ids = vector_database.search(query_vector, results_to_fetch)

    # Step 3: Match the vectors back to their human-readable PDF pages
    final_results = []
    for distance_score, doc_id in zip(distances[0], closest_ids[0]):
        if doc_id == -1:
            continue  
        
        original_page_data = text_storage.get(int(doc_id))
        
        if original_page_data:
            final_results.append({
                "document_id": int(doc_id),
                "source_file": original_page_data["filename"],
                "page_number": original_page_data["page_number"],
                "matched_text": original_page_data["text"],
                "match_score": float(distance_score)  # Lower is closer matching
            })

    return {
        "your_query": payload.query,
        "matches_found": len(final_results),
        "results": final_results
    }


@app.get("/status")
def check_database_status():
    return {
        "total_pdf_pages_in_memory": vector_database.ntotal,
        "vector_size": vector_size
    }
    
