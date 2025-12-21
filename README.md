# VectorSeek — AI-Powered Document Analysis & Semantic Search

## 🎯 Project Overview

VectorSeek is a **production-ready AI system for intelligent analysis and question-answering over documents** such as **research papers, books, technical documentation, reports, and manuals**.

It uses a **Hybrid Retrieval-Augmented Generation (RAG)** architecture that combines **local semantic search** with **Google Gemini 2.5 Flash** for accurate, grounded answers — without running any LLMs locally.

The system is designed to scale across domains while enforcing **evidence-based responses** and **hallucination control**.

---

## 🚀 Key Features

- 📄 Upload and analyze PDFs / text documents
- 🔍 Semantic search across large document collections
- ☁️ Cloud-based LLM inference using Google Gemini 2.5 Flash
- 🧠 Context-aware prompting for technical and academic queries
- 📖 Intelligent chunking preserving semantic coherence
- 📌 Source attribution and citation tracking
- 💬 Interactive chat-style interface
- 🔐 Secure API key handling (no secrets in repo)

---

## 🏗️ Architecture

### Hybrid Document Analysis Pipeline

**Local Components:**
1. **Document Ingestion** — Loads PDFs / TXT files from `data/documents/`
2. **Intelligent Chunking** — Splits documents into semantically meaningful segments
3. **Embeddings** — Dense vector embeddings via `all-MiniLM-L6-v2`
4. **Vector Database** — FAISS index for fast similarity search

**Cloud Component:**
1. **LLM Inference** — Google Gemini 2.5 Flash
2. **Grounded Prompting** — Answers constrained strictly to retrieved context

**Frontend:**
1. **Streamlit UI** — Chat-based document exploration
2. **Source Attribution** — Displays which document sections support each answer

---

## 🔄 Data Flow

User Question
↓
Query Embedding (Sentence Transformers)
↓
FAISS Semantic Search (top-k chunks)
↓
Retrieved Context
↓
Gemini 2.5 Flash API (with grounding constraints)
↓
Answer + Source References

yaml
Copy code

---

## 📁 Project Structure

VectorSeek/
├── data/
│ └── documents/ # PDFs / TXT documents
├── embeddings/
│ └── build_index.py # FAISS index builder
├── rag/
│ ├── retriever.py # Semantic retrieval
│ ├── gemini_llm.py # Gemini API integration
│ └── rag_pipeline.py # RAG orchestration
├── indexes/ # FAISS indices (auto-generated)
├── app.py # Streamlit application
├── requirements.txt
├── .env.example # Environment variable template
└── README.md

yaml
Copy code

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11+
- Google Gemini API key (free tier available)

### 1️⃣ Setup Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
2️⃣ Configure API Key
bash
Copy code
cp .env.example .env
# Add: GEMINI_API_KEY=your_key_here
Get API key: https://ai.google.dev

3️⃣ Add Documents
bash
Copy code
# Place PDFs or TXT files here
data/documents/
Supported content:

Research papers

Books / chapters

Technical documentation

Reports and manuals

Notes and whitepapers

4️⃣ Run the App
bash
Copy code
streamlit run app.py
App opens at: http://localhost:8501

📚 Supported Use Cases
📘 Research paper analysis

📕 Book and chapter Q&A

🛠️ Technical documentation assistant

🧾 Policy and compliance search

🎓 Study and exam preparation

💼 Enterprise knowledge base search

💡 Why Hybrid RAG?
Advantages:
✅ No LLM runs locally

✅ Works on low-resource machines

✅ Scales to large document collections

✅ Strong hallucination control

✅ Industry-standard architecture

Why Semantic Search?
Goes beyond keyword matching

Understands meaning and context

Handles technical and academic language

Enables cross-document reasoning

🎯 Why Google Gemini 2.5 Flash?
High-quality reasoning

Fast response time

Large context window

Cloud-managed reliability

Free tier suitable for development

Production-grade scalability

🔒 Security & Privacy
API keys managed via environment variables / Streamlit Secrets

No secrets committed to GitHub

Documents processed locally

Only retrieved context sent to LLM

No persistent cloud storage of documents

📊 Performance
Component	Latency
Vector Retrieval	<100 ms
LLM Response	2–5 s
End-to-End	~3–6 s

🎓 Resume-Ready Highlights
Designed and implemented a Hybrid RAG system using FAISS and cloud LLMs

Built semantic document search with transformer embeddings

Integrated Google Gemini 2.5 Flash for scalable inference

Implemented hallucination-resistant Q&A with source attribution

Deployed end-to-end AI system using Streamlit Cloud

🚀 Future Enhancements
Multi-language document support

Document summarization

Conversational memory

Metadata-based filtering

Domain-specific embedding fine-tuning

Usage analytics dashboard

📝 License
Open-source project for educational and commercial use.