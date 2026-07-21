import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(
        self,
        index_path="data/indexes/wiki.index",
        corpus_path="data/raw/wiki_real.csv"
    ):

        print("Loading FAISS Index...")
        self.index = faiss.read_index(index_path)

        print("Loading Corpus...")
        self.corpus = pd.read_csv(corpus_path)

        print("Loading Embedding Model...")
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        print("Retriever Ready!\n")

    def search(self, question, top_k=5):

        question_embedding = self.model.encode(
            question,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            question_embedding.reshape(1, -1),
            top_k
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            row = self.corpus.iloc[idx]

            results.append({
                "doc_id": int(row["id"]),
                "title": row["title"],
                "rank": rank,
                "retriever_score": float(score),
                "text": row["text"]
            })

        return results
