"""
Google Gemini 2.5 Flash LLM integration for answer generation.
"""

import os

import google.generativeai as genai


class GeminiLLM:
    """Wrapper for Google Gemini 2.5 Flash API."""

    def __init__(self, api_key: str = None):
        """
        Initialize Gemini LLM.

        Args:
            api_key: Google Generative AI API key
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided or found in environment")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer using Gemini with provided context.

        Args:
            context: Retrieved document context
            question: User question

        Returns:
            Generated answer string
        """
        prompt = self._build_prompt(context, question)

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    @staticmethod
    def _build_prompt(context: str, question: str) -> str:
        """Build structured prompt for document analysis."""
        prompt = f"""You are an expert AI assistant analyzing documents and answering questions.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANALYSIS GUIDELINES:
- Answer ONLY using evidence from the provided document context
- Be clear and direct in your response
- If the document is a research paper, provide rigorous analysis with evidence
- If citing specific findings or data, reference relevant sections
- Do not introduce external knowledge or information not in the provided context
- If the question cannot be answered from the context, respond: "This information is not available in the provided document."
- Be professional and accurate in tone
- Include relevant details, data points, or methodological notes when applicable

RESPONSE:
Provide a clear, well-structured answer based solely on the document context."""

        return prompt

    def stream_answer(self, context: str, question: str):
        """
        Stream answer generation.

        Args:
            context: Retrieved document context
            question: User question

        Yields:
            Text chunks as they are generated
        """
        prompt = self._build_prompt(context, question)

        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error generating answer: {str(e)}"
