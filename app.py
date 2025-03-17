import os
import tempfile
import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag import RAG

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.author-info {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f0f2f6;
    margin: 1rem 0;
}
.hire-button {
    display: inline-block;
    padding: 0.5rem 1rem;
    background-color: #ff4b4b;
    color: white;
    text-decoration: none;
    border-radius: 0.3rem;
    font-weight: bold;
    margin: 1rem 0;
}
.hire-button:hover {
    background-color: #ff3333;
    color: white;
}
.interactive-box {
    padding: 1rem;
    border: 2px solid #4CAF50;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
    
if "rag" not in st.session_state:
    st.session_state.rag = RAG(st.session_state.vector_store)
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_document(uploaded_file):
    """Process an uploaded document."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write the uploaded file data to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Process the document
        chunks = st.session_state.processor.process_document(tmp_path)
        
        # Add to vector store
        st.session_state.vector_store.add_documents(chunks)
        
        # Add to uploaded files list if not already there
        if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "chunks": len(chunks)
            })
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

def clear_documents():
    """Clear all uploaded documents and reset the vector store."""
    st.session_state.vector_store = VectorStore()
    st.session_state.uploaded_files = []
    st.session_state.rag = RAG(st.session_state.vector_store)

# App header with author info
st.title("🤖 Document Assistant")
st.markdown("""
<div class="author-info">
    <h3>Created by Chaitanya Vankadaru</h3>
    <p>AI/ML Engineer | Python Developer | Data Scientist</p>
    <a href="https://www.linkedin.com/in/chaitanyavankadaru" target="_blank" class="hire-button">🚀 Hire Me!</a>
</div>
""", unsafe_allow_html=True)

st.write("Upload documents and ask questions about their content using advanced AI technology")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                process_document(uploaded_file)
    
    st.header("Settings")
    with st.expander("Advanced Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        
        if chunk_size != st.session_state.processor.chunk_size or chunk_overlap != st.session_state.processor.chunk_overlap:
            st.session_state.processor = DocumentProcessor(chunk_size, chunk_overlap)
    
    if st.button("Clear All Documents", type="secondary"):
        clear_documents()
    
    st.header("Uploaded Documents")
    if st.session_state.uploaded_files:
        for doc in st.session_state.uploaded_files:
            with st.container():
                st.markdown(f"""
                <div class="interactive-box">
                    📄 <b>{doc['name']}</b><br>
                    Chunks: {doc['chunks']} | Size: {doc['size']/1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No documents uploaded")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    question = st.text_input("Ask a question about your documents", placeholder="e.g., What are the key points about AI?")
    
    if st.button("Submit Question", type="primary"):
        if not st.session_state.uploaded_files:
            st.warning("Please upload and process at least one document first.")
        elif not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤔 Thinking..."):
                answer = st.session_state.rag.generate_answer(question)
                # Add to chat history
                st.session_state.chat_history.append({"question": question, "answer": answer})
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💭 Conversation History")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"""
                <div class="interactive-box" style="border-color: #2196F3;">
                    <b>You:</b> {chat['question']}<br><br>
                    <b>Assistant:</b> {chat['answer']}
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.header("🔍 RAG Architecture")
    with st.expander("How it works", expanded=True):
        st.write("""
        This application uses a Retrieval Augmented Generation (RAG) architecture:
        
        1. **Document Processing**: Documents are chunked into smaller segments
        2. **Vector Embeddings**: Chunks are converted to vector embeddings
        3. **FAISS Index**: Embeddings are stored in a FAISS vector index
        4. **Semantic Search**: User queries retrieve the most relevant chunks
        5. **Generation**: Retrieved context is sent to an LLM to generate answers
        """)
    
    # System stats
    st.header("📊 System Statistics")
    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        if st.session_state.uploaded_files:
            total_chunks = sum(doc["chunks"] for doc in st.session_state.uploaded_files)
            st.metric("📚 Documents", len(st.session_state.uploaded_files))
    with col_stats2:
        if st.session_state.uploaded_files:
            st.metric("🧩 Total Chunks", total_chunks)
        else:
            st.metric("📚 Documents", 0)
            st.metric("🧩 Total Chunks", 0)
    
    # Contact info
    st.markdown("""
    <div class="author-info">
        <h4>About the Author</h4>
        <p>Chaitanya Vankadaru is an experienced AI/ML Engineer with expertise in building intelligent applications.</p>
        <p>Connect with me:</p>
        <a href="https://www.linkedin.com/in/chaitanyavankadaru" target="_blank" class="hire-button">LinkedIn Profile</a>
    </div>
    """, unsafe_allow_html=True) 