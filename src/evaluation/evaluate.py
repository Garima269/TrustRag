"""
evaluate.py

Evaluates generated answers using
Exact Match (EM) and F1 Score.
"""

from src.evaluation.metrics import EvaluationMetrics

class Evaluator:
    """
    Evaluate generated answers.
    """

    def evaluate(self, prediction: str, ground_truth: str):
        """
        Evaluate one prediction.
        Returns:
            Dictionary containing EM and F1.
        """

        em = EvaluationMetrics.exact_match(prediction, ground_truth)

        f1 = EvaluationMetrics.f1_score(prediction, ground_truth)

        return {"Exact Match": em, "F1 Score": f1}

if __name__ == "__main__":

    ground_truth = "Delhi"

    prediction = "The capital of India is Delhi."

    evaluator = Evaluator()

    results = evaluator.evaluate(prediction, ground_truth)

    print("\nEvaluation Results\n")

    print(f"Exact Match : {results['Exact Match']}")
    print(f"F1 Score    : {results['F1 Score']}")
