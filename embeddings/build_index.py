"""
Build and manage FAISS embeddings index for document retrieval.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingsBuilder:
    """Build and manage FAISS index for semantic search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize embeddings builder.

        Args:
            model_name: Sentence transformer model name
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.metadata = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, "rb") as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text

    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from text file."""
        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file {txt_path}: {e}")
            return ""

    def chunk_text(self, text: str, source: str) -> List[Tuple[str, str]]:
        """
        Split text into overlapping chunks optimized for research papers.
        
        Strategy:
        - Preserve complete sentences to maintain semantic coherence
        - Larger overlaps to maintain context across citations
        - Avoids breaking in the middle of important sections

        Args:
            text: Input text to chunk
            source: Source document name

        Returns:
            List of (chunk_text, source) tuples
        """
        # Split by sentences first to preserve meaning
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_word_count + sentence_words > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                if chunk_text.strip():
                    chunks.append((chunk_text, source))
                
                # Keep last few sentences for overlap
                overlap_sentences = []
                word_count = 0
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if word_count + sent_words <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        word_count += sent_words
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_word_count = word_count
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            if chunk_text.strip():
                chunks.append((chunk_text, source))
        
        return chunks

    def load_documents(self, documents_dir: str) -> None:
        """
        Load all documents from directory.

        Args:
            documents_dir: Directory containing PDF and TXT files
        """
        documents_path = Path(documents_dir)

        if not documents_path.exists():
            print(f"Documents directory {documents_dir} not found")
            return

        for file_path in documents_path.iterdir():
            if file_path.suffix.lower() == ".pdf":
                print(f"Processing PDF: {file_path.name}")
                text = self.extract_text_from_pdf(str(file_path))
                chunks = self.chunk_text(text, file_path.name)
                self.chunks.extend(chunks)

            elif file_path.suffix.lower() == ".txt":
                print(f"Processing TXT: {file_path.name}")
                text = self.extract_text_from_txt(str(file_path))
                chunks = self.chunk_text(text, file_path.name)
                self.chunks.extend(chunks)

        print(f"Total chunks created: {len(self.chunks)}")

    def build_index(self) -> None:
        """Build FAISS index from chunks."""
        if not self.chunks:
            print("No chunks to index")
            return

        # Extract chunk texts
        chunk_texts = [chunk[0] for chunk in self.chunks]
        self.metadata = [chunk[1] for chunk in self.chunks]

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Create FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)

        print(f"Index built with {self.index.ntotal} vectors")

    def save_index(self, index_path: str, metadata_path: str, chunks_path: str) -> None:
        """
        Save FAISS index and metadata.

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
            chunks_path: Path to save chunks
        """
        if self.index is None:
            print("Index not built yet")
            return

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
        print(f"Chunks saved to {chunks_path}")

    @staticmethod
    def load_index(index_path: str, metadata_path: str, chunks_path: str) -> Tuple["EmbeddingsBuilder", dict]:
        """
        Load FAISS index and metadata.

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata
            chunks_path: Path to chunks

        Returns:
            Tuple of (builder, index_data)
        """
        builder = EmbeddingsBuilder()

        if not os.path.exists(index_path):
            print(f"Index file not found: {index_path}")
            return builder, {}

        builder.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            builder.metadata = pickle.load(f)

        with open(chunks_path, "rb") as f:
            builder.chunks = pickle.load(f)

        print(f"Index loaded from {index_path}")
        return builder, {"metadata": builder.metadata, "chunks": builder.chunks}


def build_and_save_index(documents_dir: str, output_dir: str = "indexes") -> None:
    """Build and save embeddings index."""
    builder = EmbeddingsBuilder()
    builder.load_documents(documents_dir)
    builder.build_index()

    os.makedirs(output_dir, exist_ok=True)
    builder.save_index(
        os.path.join(output_dir, "faiss_index.bin"),
        os.path.join(output_dir, "metadata.pkl"),
        os.path.join(output_dir, "chunks.pkl"),
    )


if __name__ == "__main__":
    build_and_save_index("data/documents")
