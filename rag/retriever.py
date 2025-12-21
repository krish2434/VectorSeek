"""
Retrieve relevant documents using FAISS semantic search.
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    """Retrieve relevant document chunks using semantic similarity."""

    def __init__(self, index_path: str, metadata_path: str, chunks_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize retriever with FAISS index.

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata pickle file
            chunks_path: Path to chunks pickle file
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.chunks = []

        self._load_index(index_path, metadata_path, chunks_path)

    def _load_index(self, index_path: str, metadata_path: str, chunks_path: str) -> None:
        """Load FAISS index and associated data."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Retriever initialized with {self.index.ntotal} document chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve

        Returns:
            List of (chunk_text, source, distance) tuples
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk_text = self.chunks[idx][0]
                source = self.chunks[idx][1]
                distance = distances[0][i]
                results.append((chunk_text, source, distance))

        return results

    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        Get concatenated context from top-k relevant chunks.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve

        Returns:
            Concatenated context string
        """
        results = self.retrieve(query, top_k)
        context_parts = [chunk[0] for chunk in results]
        return "\n\n".join(context_parts)

    def get_sources(self, query: str, top_k: int = 5) -> List[str]:
        """
        Get unique source documents for retrieved chunks.

        Args:
            query: User query string
            top_k: Number of chunks to retrieve

        Returns:
            List of unique source document names
        """
        results = self.retrieve(query, top_k)
        sources = list(set(chunk[1] for chunk in results))
        return sources


def initialize_retriever(index_dir: str = "indexes") -> DocumentRetriever:
    """Initialize retriever from index directory."""
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    chunks_path = os.path.join(index_dir, "chunks.pkl")

    return DocumentRetriever(index_path, metadata_path, chunks_path)
