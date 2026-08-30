"""
semantic_relevance.py

Computes semantic relevance between a user question
and a retrieved sentence using a CrossEncoder.

Output:
    Semantic relevance score in [0, 1]
"""

from sentence_transformers import CrossEncoder
from math import exp


class SemanticRelevance:
    """
    Computes semantic relevance using a CrossEncoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):

        print("Loading Semantic Relevance Model...")

        self.model = CrossEncoder(model_name)

        print("Semantic Relevance Model Ready!\n")

    def compute_score(self, question: str, sentence: str) -> float:
        """
        Compute semantic relevance between a question
        and a sentence.

        Returns:
            float: normalized score in [0, 1]
        """

        if not question or not question.strip():
            return 0.0

        if not sentence or not sentence.strip():
            return 0.0

        # Get raw CrossEncoder relevance score
        raw_score = self.model.predict([(question, sentence)])[0]

        raw_score = float(raw_score)

        # Sigmoid converts the raw score to [0, 1]
        normalized_score = 1.0 / (1.0 + exp(-raw_score))

        return round(normalized_score, 4)

    def compute_scores(self, question: str, sentence_records: list) -> list:
        """
        Compute semantic relevance for all
        retrieved sentence records.

        Adds:
            semantic_score
        """

        for record in sentence_records:

            score = self.compute_score(question, record.get("sentence_text", ""))

            # This is the field used by TrustEstimator
            # and NoiseClassifier.
            record["semantic_score"] = score

            # Keep this too if other parts of the
            # project use the older field name.
            record["semantic_relevance"] = score

        return sentence_records


if __name__ == "__main__":

    question = "Who invented the electric battery?"

    sentence = "Alessandro Volta invented " "the electric battery in 1799."

    relevance = SemanticRelevance()

    score = relevance.compute_score(question, sentence)

    print("\nSemantic Relevance Score\n")
    print(score)

    print("\nScore Range Check")

    if 0.0 <= score <= 1.0:
        print("Valid: score is within [0, 1]")
    else:
        print("ERROR: score outside [0, 1]")
