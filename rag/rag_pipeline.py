"""
Hybrid RAG pipeline combining document retrieval and Gemini LLM.
"""

from typing import List, Tuple

from .gemini_llm import GeminiLLM
from .retriever import DocumentRetriever


class RAGPipeline:
    """End-to-end Hybrid RAG pipeline."""

    def __init__(self, retriever: DocumentRetriever, llm: GeminiLLM = None, top_k: int = 5):
        """
        Initialize RAG pipeline.

        Args:
            retriever: DocumentRetriever instance
            llm: GeminiLLM instance (initialized if not provided)
            top_k: Number of chunks to retrieve
        """
        self.retriever = retriever
        self.llm = llm or GeminiLLM()
        self.top_k = top_k

    def answer_question(self, question: str) -> dict:
        """
        Answer a question using RAG pipeline.

        Args:
            question: User question

        Returns:
            Dictionary with answer, context, and sources
        """
        # Retrieve context
        context = self.retriever.get_context(question, self.top_k)
        sources = self.retriever.get_sources(question, self.top_k)

        # Generate answer
        answer = self.llm.generate_answer(context, question)

        return {
            "answer": answer,
            "context": context,
            "sources": sources,
            "question": question,
        }

    def stream_answer(self, question: str):
        """
        Stream answer generation.

        Args:
            question: User question

        Yields:
            Answer chunks as they are generated
        """
        context = self.retriever.get_context(question, self.top_k)

        for chunk in self.llm.stream_answer(context, question):
            yield chunk

    def batch_answer_questions(self, questions: List[str]) -> List[dict]:
        """
        Answer multiple questions.

        Args:
            questions: List of questions

        Returns:
            List of answer dictionaries
        """
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)

        return results
