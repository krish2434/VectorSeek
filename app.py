"""
Streamlit chat interface for Hybrid RAG system.
"""

import os
import io
import tempfile

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from embeddings.build_index import build_and_save_index, EmbeddingsBuilder
from rag.gemini_llm import GeminiLLM
from rag.rag_pipeline import RAGPipeline
from rag.retriever import initialize_retriever

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="VectorSeek - Hybrid RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stChatMessage[data-testid="chatMessage-user"] {
        background-color: #e3f2fd;
    }
    .stChatMessage[data-testid="chatMessage-assistant"] {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline with caching."""
    index_dir = "indexes"

    # Check if index exists, if not build it
    if not os.path.exists(os.path.join(index_dir, "faiss_index.bin")):
        with st.spinner("Building embeddings index from documents..."):
            build_and_save_index("data/documents", index_dir)

    # Initialize retriever and LLM
    retriever = initialize_retriever(index_dir)
    llm = GeminiLLM()
    pipeline = RAGPipeline(retriever, llm, top_k=5)

    return pipeline


def main():
    """Main Streamlit application."""
    st.title("📚 VectorSeek - Document Analysis")
    st.markdown("Intelligent semantic search and Q&A for your documents")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("📖 Document Analysis")

        st.markdown("### Search Settings")
        top_k = st.slider("Number of Sections to Retrieve", min_value=1, max_value=10, value=5, step=1)
        st.markdown("Retrieve most relevant document sections")

        st.markdown("---")
        st.markdown("### Status")
        
        index_exists = os.path.exists("indexes/faiss_index.bin")
        if index_exists:
            st.success("✅ Embeddings Index: Ready")
        else:
            st.warning("⚠️ Embeddings Index: Not built")
            
            # Add manual build button
            if st.button("🔨 Build Index Now", use_container_width=True):
                with st.spinner("Building embeddings index..."):
                    try:
                        build_and_save_index("data/documents", "indexes")
                        st.success("✅ Index built successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error building index: {str(e)}")
                        st.info("📝 Make sure you have PDF/TXT files in data/documents/ folder")

        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("✅ Gemini API: Connected")
        else:
            st.error("❌ Gemini API: Not configured")
            st.info("⚠️ Add GEMINI_API_KEY to your .env file")

        st.markdown("---")
        st.markdown("### 📥 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file is not None:
            if st.button("🔍 Process Document", use_container_width=True):
                with st.spinner("Indexing document..."):
                    try:
                        # Extract text from uploaded PDF with UTF-8 encoding
                        pdf_text = ""
                        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text += page_text + "\n"
                        
                        if not pdf_text.strip():
                            st.error("Could not extract text from PDF. Try a different file.")
                            st.stop()
                        
                        # Create temp file to store PDF text with UTF-8 encoding
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                            tmp.write(pdf_text)
                            tmp_path = tmp.name
                        
                        # Build index from uploaded PDF
                        builder = EmbeddingsBuilder()
                        chunks = builder.chunk_text(pdf_text, uploaded_file.name)
                        builder.chunks = chunks
                        builder.metadata = [uploaded_file.name] * len(chunks)
                        
                        # Generate embeddings and build index
                        chunk_texts = [chunk[0] for chunk in chunks]
                        import numpy as np
                        embeddings = builder.model.encode(chunk_texts, show_progress_bar=False)
                        embeddings = np.array(embeddings, dtype=np.float32)
                        
                        import faiss
                        builder.index = faiss.IndexFlatL2(builder.embedding_dim)
                        builder.index.add(embeddings)
                        
                        # Store in session state
                        st.session_state.uploaded_retriever = builder
                        st.session_state.uploaded_file_name = uploaded_file.name
                        
                        st.success(f"✅ Document indexed! ({len(chunks)} sections)")
                        st.info(f"📄 Processing: {uploaded_file.name}")
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        st.info("💡 Try uploading a different PDF file")
        
        if "uploaded_file_name" in st.session_state:
            st.info(f"📄 Current: {st.session_state.uploaded_file_name}")
            if st.button("🔄 Load Different Document", use_container_width=True):
                del st.session_state.uploaded_retriever
                del st.session_state.uploaded_file_name
                st.rerun()

    # Main chat interface
    col = st.container()

    with col:
        st.subheader("❓ Ask a Question")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("� Cited Sections"):
                            for source in message["sources"]:
                                st.caption(f"📄 {source}")

        # User input
        st.markdown("---")
        user_question = st.chat_input("Ask a question about the document...")

        if user_question:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_question})

            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_question)

            # Generate answer
            try:
                with st.spinner("Retrieving documents and generating answer..."):
                    llm = GeminiLLM()
                    
                    # Check if using uploaded PDF or default index
                    if "uploaded_retriever" in st.session_state:
                        # Use uploaded PDF retriever
                        builder = st.session_state.uploaded_retriever
                        query_embedding = builder.model.encode([user_question], convert_to_numpy=True)
                        import numpy as np
                        query_embedding = np.array(query_embedding, dtype=np.float32)
                        distances, indices = builder.index.search(query_embedding, min(top_k, builder.index.ntotal))
                        
                        context_parts = []
                        for i, idx in enumerate(indices[0]):
                            if idx < len(builder.chunks):
                                context_parts.append(builder.chunks[idx][0])
                        context = "\n\n".join(context_parts)
                        sources = [st.session_state.uploaded_file_name]
                    else:
                        # Use default index
                        pipeline = initialize_rag_pipeline()
                        pipeline.top_k = top_k
                        context = pipeline.retriever.get_context(user_question, top_k)
                        sources = pipeline.retriever.get_sources(user_question, top_k)
                        result = pipeline.answer_question(user_question)
                    
                    # Generate answer if not already done
                    if "uploaded_retriever" not in st.session_state or "result" not in locals():
                        answer = llm.generate_answer(context, user_question)
                    else:
                        answer = result["answer"]
                        context = result["context"]
                        sources = result["sources"]

                    # Add assistant message to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                    # Display assistant message
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                            with st.expander("� Cited Sections"):
                                for source in sources:
                                    st.caption(f"📄 {source}")

            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(error_message)

if __name__ == "__main__":
    main()
