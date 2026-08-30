"""
evidence_agreement.py

Computes evidence agreement between a target sentence
and supporting retrieved sentences using an NLI CrossEncoder.

The final evidence agreement score is normalized to [0, 1].
"""

import numpy as np
from sentence_transformers import CrossEncoder


class EvidenceAgreement:

    def __init__(
        self,
        model_name="cross-encoder/nli-deberta-v3-base",
        max_evidence_sentences=10,
    ):

        print("Loading Evidence Agreement Model...")

        self.model = CrossEncoder(model_name)

        self.max_evidence_sentences = max_evidence_sentences

        print("Evidence Agreement Model Ready!\n")

    def compute_score(self, target_sentence, sentence_list):
        """
        Compute evidence agreement between the target sentence
        and supporting retrieved sentences.

        Returns:
            float: Evidence agreement score in [0, 1]
        """

        if not sentence_list:
            return 0.0

        # Remove the target sentence itself
        candidates = [
            sentence for sentence in sentence_list if sentence != target_sentence
        ]

        if not candidates:
            return 0.0

        # Limit number of evidence candidates
        candidates = candidates[: self.max_evidence_sentences]

        # Create NLI pairs
        pairs = [(target_sentence, sentence) for sentence in candidates]

        # Get raw NLI logits
        results = self.model.predict(pairs, apply_softmax=False)

        agreement_scores = []

        for result in results:

            result = np.asarray(result, dtype=float)

            # -------------------------------------------------
            # 3-class NLI output
            # -------------------------------------------------
            #
            # [contradiction, entailment, neutral]
            #
            # Convert logits -> probabilities using softmax.
            # -------------------------------------------------

            if result.ndim == 1 and len(result) >= 3:

                # Numerical-stable softmax
                exp_scores = np.exp(result - np.max(result))

                probabilities = exp_scores / exp_scores.sum()

                entailment_probability = float(probabilities[1])

                agreement_scores.append(entailment_probability)

            else:

                # Fallback for scalar output
                score = float(result)

                # Keep score safely inside [0, 1]
                score = max(0.0, min(1.0, score))

                agreement_scores.append(score)

        if not agreement_scores:
            return 0.0

        # Average agreement across evidence sentences
        score = sum(agreement_scores) / len(agreement_scores)

        # Final safety clamp
        score = max(0.0, min(1.0, score))

        return round(score, 4)


# =============================================================
# Standalone Test
# =============================================================

if __name__ == "__main__":

    target = "Alessandro Volta invented " "the electric battery."

    retrieved_sentences = [
        "Volta developed the voltaic pile in 1799.",
        "Alessandro Volta was an Italian physicist.",
        "The electric battery was invented by Volta.",
        "Lionel Messi is an Argentine footballer.",
    ]

    agreement = EvidenceAgreement(max_evidence_sentences=10)

    score = agreement.compute_score(target, retrieved_sentences)

    print("\nEvidence Agreement Score\n")
    print(score)

    print("\nScore Range Check")

    if 0.0 <= score <= 1.0:
        print("Valid: score is within [0, 1]")
    else:
        print("ERROR: score is outside [0, 1]")
