"""
context_builder.py

This module builds the final context for the LLM by combining
cleaned sentence-level records into a single context string.

Input:
    List of cleaned sentence dictionaries

Output:
    Context string
"""

from typing import List, Dict


class ContextBuilder:
    """
    Builds context for the Language Model.
    """

    def build_context(self, sentence_records: List[Dict]) -> str:
        """
        Build a context string from sentence records.

        Parameters:
            sentence_records (List[Dict])

        Returns:
            str
        """

        if not sentence_records:
            return ""

        # Preserve retrieval order
        sorted_sentences = sorted(
            sentence_records,
            key=lambda x: (
                x.get("rank", float("inf")),
                x.get("sentence_id", 0)
            )
        )

        # Extract sentence text
        context_sentences = [
            record["sentence_text"]
            for record in sorted_sentences
        ]

        # Join sentences with blank lines
        context = "\n\n".join(context_sentences)

        return context


if __name__ == "__main__":

    sample_sentences = [
        {
            "doc_id": 2,
            "title": "Mumbai",
            "rank": 2,
            "retriever_score": 0.91,
            "sentence_id": 1,
            "sentence_text": "Mumbai is India's financial capital."
        },
        {
            "doc_id": 1,
            "title": "Delhi",
            "rank": 1,
            "retriever_score": 0.95,
            "sentence_id": 1,
            "sentence_text": "Delhi is the capital of India."
        },
        {
            "doc_id": 1,
            "title": "Delhi",
            "rank": 1,
            "retriever_score": 0.95,
            "sentence_id": 2,
            "sentence_text": "It is located in northern India."
        }
    ]

    builder = ContextBuilder()

    context = builder.build_context(sample_sentences)

    print("\nGenerated Context:\n")
    print(context)