import re
import string
from collections import Counter


class EvaluationMetrics:
    """
    Calculate QA evaluation metrics.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text before evaluation.

        Operations:
        1. Convert to lowercase.
        2. Remove punctuation.
        3. Remove articles.
        4. Normalize whitespace.
        """

        text = str(text).strip().lower()
        text = text.replace("\n", " ")

        # Remove punctuation.
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove common English articles.
        text = re.sub(r"\b(a|an|the)\b", " ", text)

        # Normalize whitespace.
        text = " ".join(text.split())

        return text

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> int:
        """
        Calculate Exact Match.

        Returns:
            1 if normalized prediction exactly matches
            normalized ground truth, otherwise 0.
        """

        prediction = EvaluationMetrics.normalize_text(prediction)

        ground_truth = EvaluationMetrics.normalize_text(ground_truth)

        return int(prediction == ground_truth)

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score.
        """

        prediction_tokens = EvaluationMetrics.normalize_text(prediction).split()

        ground_truth_tokens = EvaluationMetrics.normalize_text(ground_truth).split()

        if not prediction_tokens or not ground_truth_tokens:
            return float(prediction_tokens == ground_truth_tokens)

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)

        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(prediction_tokens)

        recall = num_common / len(ground_truth_tokens)

        f1 = 2 * precision * recall / (precision + recall)

        return f1


if __name__ == "__main__":

    prediction = (
        "The electric battery was invented "
        "by Alessandro Giuseppe Antonio Anastasio Volta."
    )

    ground_truth = "Alessandro Volta"

    print("Exact Match:", EvaluationMetrics.exact_match(prediction, ground_truth))

    print("F1 Score:", EvaluationMetrics.f1_score(prediction, ground_truth))
