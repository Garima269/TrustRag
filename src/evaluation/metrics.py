"""
metrics.py

Implements evaluation metrics for the baseline RAG pipeline.

Metrics:
1. Exact Match (EM)
2. F1 Score
"""

import string
class EvaluationMetrics:
    """
    Compute evaluation metrics for Question Answering.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text before comparison.

        - Lowercase
        - Remove punctuation
        - Remove extra spaces
        """

        text = text.lower()

        text = text.translate(str.maketrans("", "", string.punctuation))

        text = " ".join(text.split())

        return text

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> int:
        """
        Compute Exact Match (EM).

        Returns:
            1 if answers match exactly
            0 otherwise
        """

        prediction = EvaluationMetrics.normalize_text(prediction)
        ground_truth = EvaluationMetrics.normalize_text(ground_truth)

        return int(prediction == ground_truth)

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """
        Compute token-level F1 score.
        """

        prediction_tokens = EvaluationMetrics.normalize_text(prediction).split()
        ground_truth_tokens = EvaluationMetrics.normalize_text(ground_truth).split()

        common_tokens = set(prediction_tokens) & set(ground_truth_tokens)

        if len(common_tokens) == 0:
            return 0.0

        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(ground_truth_tokens)

        f1 = (2 * precision * recall) / (precision + recall)

        return round(f1, 4)
