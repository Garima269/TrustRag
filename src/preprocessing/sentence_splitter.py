"""
sentence_splitter.py

This module splits retrieved documents into individual sentences
using spaCy.

Input:
    List of retrieved documents

Output:
    List of sentence-level dictionaries
"""

from typing import List, Dict
import spacy


class SentenceSplitter:
    """
    Splits retrieved documents into individual sentences while
    preserving document-level metadata.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Load the spaCy language model.

        Parameters:
            model_name (str): Name of the spaCy model.
        """
        self.nlp = spacy.load(model_name)

    def split_document(self, document: Dict) -> List[Dict]:
        """
        Split a single document into sentences.

        Parameters:
            document (dict): Retrieved document containing
                - doc_id
                - text
                - (optional) title
                - (optional) rank
                - (optional) retriever_score

        Returns:
            List[dict]: Sentence-level records.
        """

        # Check if document contains text
        if "text" not in document or not document["text"].strip():
            return []

        doc = self.nlp(document["text"])

        sentences = []

        for idx, sent in enumerate(doc.sents, start=1):

            sentence_record = {
                "doc_id": document.get("doc_id"),
                "title": document.get("title", ""),
                "rank": document.get("rank", None),
                "retriever_score": document.get("retriever_score", None),
                "sentence_id": idx,
                "sentence_text": sent.text.strip()
            }

            sentences.append(sentence_record)

        return sentences

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split multiple retrieved documents into sentences.

        Parameters:
            documents (List[dict])

        Returns:
            List[dict]
        """

        all_sentences = []

        for document in documents:
            all_sentences.extend(self.split_document(document))

        return all_sentences


if __name__ == "__main__":

    sample_documents = [
        {
            "doc_id": 1,
            "title": "Delhi",
            "rank": 1,
            "retriever_score": 0.95,
            "text": "Delhi is the capital of India. It is located in northern India."
        },
        {
            "doc_id": 2,
            "title": "Mumbai",
            "rank": 2,
            "retriever_score": 0.91,
            "text": "Mumbai is India's financial capital. It has a large population."
        }
    ]

    splitter = SentenceSplitter()

    sentences = splitter.split_documents(sample_documents)

    print("\nSentence Segmentation Output:\n")

    for sentence in sentences:
        print(sentence)