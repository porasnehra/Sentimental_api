import streamlit as st
import requests

st.set_page_config(page_title="Local Semantic Search", layout="centered")
st.title("🔍 Semantic Document Search")

# Define the backend URL (You will update this after deploying the backend)
BACKEND_URL = st.sidebar.text_input(
    "Backend API URL", 
    value="http://127.0.0.1:8000"  # Default for local testing
)

tabs = st.tabs(["Search Documents", "Index Documents", "System Status"])

# 1. Indexing Tab
with tabs[1]:
    st.header("📄 Add Documents to Index")
    doc_text = st.text_area("Paste document text here:", height=150)
    doc_source = st.text_input("Source Metadata (e.g., wiki, pdf_doc):")
    
    if st.button("Submit to Index", type="primary"):
        if doc_text:
            payload = {
                "documents": [
                    {
                        "text": doc_text,
                        "metadata": {"source": doc_source}
                    }
                ]
            }
            try:
                response = requests.post(f"{BACKEND_URL}/index", json=payload)
                if response.status_code == 201:
                    st.success("Document indexed successfully!")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
        else:
            st.warning("Please enter text before submitting.")

# 2. Search Tab
with tabs[0]:
    st.header("🔎 Find Relevant Context")
    query = st.text_input("Enter your natural language query:")
    top_k = st.slider("Number of results (Top K)", min_value=1, max_value=5, value=3)
    
    if st.button("Search", type="primary"):
        if query:
            payload = {"query": query, "top_k": top_k}
            try:
                response = requests.post(f"{BACKEND_URL}/search", json=payload)
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if not results:
                        st.info("No documents matched or index is empty.")
                    for res in results:
                        with st.expander(f"Document ID: {res['document_id']} (Distance Score: {res['score']:.4f})"):
                            st.write(res["text"])
                            st.caption(f"Metadata: {res['metadata']}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
        else:
            st.warning("Please enter a query.")

# 3. Status Tab
with tabs[2]:
    st.header("⚙️ Server Status")
    if st.button("Check Status"):
        try:
            response = requests.get(f"{BACKEND_URL}/status")
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error("Failed to fetch status.")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
