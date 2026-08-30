from src.evaluation.metrics import EvaluationMetrics


class Evaluator:
    """
    Evaluate generated answers.
    """

    def evaluate(self, prediction: str, ground_truth: str) -> dict:
        """
        Evaluate one prediction against one ground truth.

        Returns:
            Dictionary containing EM and F1.
        """

        exact_match = EvaluationMetrics.exact_match(prediction, ground_truth)

        f1 = EvaluationMetrics.f1_score(prediction, ground_truth)

        return {"exact_match": exact_match, "f1": f1}


if __name__ == "__main__":

    prediction = "The electric battery was invented " "by Alessandro Volta."

    ground_truth = "Alessandro Volta"

    evaluator = Evaluator()

    results = evaluator.evaluate(prediction, ground_truth)

    print("\nEvaluation Results\n")

    print(f"Exact Match : " f"{results['exact_match']}")

    print(f"F1 Score    : " f"{results['f1']:.4f}")
