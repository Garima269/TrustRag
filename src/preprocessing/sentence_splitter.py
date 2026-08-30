"""
sentence_splitter.py

Splits retrieved documents into individual sentences.

Input:
    Retrieved documents from Retriever.

Output:
    Sentence-level records containing:
        - doc_id
        - title
        - rank
        - retriever_score
        - sentence_id
        - sentence_text
"""

from typing import List, Dict
import spacy


class SentenceSplitter:
    """
    Splits retrieved documents into sentences using spaCy.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):

        print("Loading Sentence Splitter...")

        self.nlp = spacy.load(model_name)

        print("Sentence Splitter Ready!\n")

    def split_document(self, document: Dict) -> List[Dict]:
        """
        Split one retrieved document into sentences.

        Parameters:
            document (dict):
                Retrieved document containing:
                doc_id
                title
                rank
                retriever_score
                text

        Returns:
            List of sentence records.
        """

        text = document.get("text", "")

        if not text or not str(text).strip():
            return []

        doc = self.nlp(str(text))

        sentences = []

        for idx, sent in enumerate(doc.sents, start=1):

            sentence_text = sent.text.strip()

            if not sentence_text:
                continue

            sentence_record = {
                "doc_id": document.get("doc_id"),
                "title": document.get("title", ""),
                "rank": document.get("rank"),
                "retriever_score": document.get("retriever_score"),
                "sentence_id": idx,
                "sentence_text": sentence_text,
            }

            sentences.append(sentence_record)

        return sentences

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split all retrieved documents into sentences.

        Parameters:
            documents (list):
                Retrieved documents from Retriever.

        Returns:
            List of sentence-level records.
        """

        all_sentences = []

        for document in documents:

            sentences = self.split_document(document)

            all_sentences.extend(sentences)

        return all_sentences


if __name__ == "__main__":

    sample_documents = [
        {
            "doc_id": 1,
            "title": "Alessandro Volta",
            "rank": 1,
            "retriever_score": 0.7718,
            "text": (
                "Alessandro Volta was an Italian physicist. "
                "He invented the electric battery. "
                "The battery was developed in 1799."
            ),
        }
    ]

    splitter = SentenceSplitter()

    sentences = splitter.split_documents(sample_documents)

    print("Sentence Segmentation Output\n")

    for sentence in sentences:
        print(sentence)
