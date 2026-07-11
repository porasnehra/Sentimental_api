import streamlit as st
import requests

st.set_page_config(
    page_title="PDF Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8000"

st.sidebar.title("⚙️ Configuration")
st.session_state.backend_url = st.sidebar.text_input(
    "Backend API URL", 
    value=st.session_state.backend_url
).rstrip("/")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Status")

try:
    status_response = requests.get(f"{st.session_state.backend_url}/status", timeout=5)
    if status_response.status_code == 200:
        status_data = status_response.json()
        st.sidebar.success("Connected to Backend")
        st.sidebar.metric("Indexed Pages", status_data.get("total_pdf_pages_in_memory", 0))
    else:
        st.sidebar.error("Backend returned an error status.")
except Exception:
    st.sidebar.warning("🔴 Unable to connect to backend service.")

st.title("🔍 PDF Semantic Document Search")
st.write("Upload PDF documents to extract their contents into a vector space, then search them using natural language queries.")

upload_tab, search_tab = st.tabs(["📤 Upload & Index PDF", "🔎 Semantic Search"])

with upload_tab:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Process and Index Document", type="primary"):
            with st.spinner("Reading PDF, generating embeddings, and updating FAISS index..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(
                        f"{st.session_state.backend_url}/index-pdf", 
                        files=files
                    )
                    
                    if response.status_code == 201:
                        result = response.json()
                        st.success(result.get("message", "PDF successfully indexed!"))
                        st.json(result.get("details", []))
                    else:
                        st.error(f"Failed to index: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"An error occurred while connecting to the backend: {str(e)}")

with search_tab:
    st.header("Search Knowledge Base")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("Enter your natural language query", placeholder="e.g., What are the terms of the agreement?")
    with col2:
        max_results = st.number_input("Max matches", min_value=1, max_value=20, value=3)
        
    if st.button("🔍 Search", type="primary"):
        if not search_query.strip():
            st.warning("Please enter a valid search query.")
        else:
            with st.spinner("Searching through index vectors..."):
                try:
                    payload = {"query": search_query, "max_results": max_results}
                    response = requests.post(
                        f"{st.session_state.backend_url}/search", 
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        search_data = response.json()
                        matches = search_data.get("results", [])
                        
                        st.subheader(f"Matches Found: {search_data.get('matches_found', 0)}")
                        
                        if not matches:
                            st.info("No matching content found for your query.")
                            
                        for idx, match in enumerate(matches):
                            with st.container():
                                col_meta, col_score = st.columns([3, 1])
                                with col_meta:
                                    st.markdown(f"#### 📄 {match['source_file']} (Page {match['page_number']})")
                                with col_score:
                                    st.info(f"Distance Score: {match['match_score']:.4f}")
                                
                                st.text_area(
                                    f"Extracted Match Fragment #{idx+1}", 
                                    value=match['matched_text'], 
                                    height=150, 
                                    disabled=True
                                )
                                st.markdown("---")
                    else:
                        st.error("The search service returned an unexpected error response.")
                except Exception as e:
                    st.error(f"Could not reach search backend: {str(e)}")
                    
