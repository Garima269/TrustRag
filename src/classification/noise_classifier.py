"""
noise_classifier.py

Multi-class noise classification for TrustRAG.

Input:
    sentence_records containing trust-related features.

Output:
    sentence_records with a new "noise_label" field.

Labels:
    Relevant
    Irrelevant
    Contradictory
    Factually Incorrect
"""


class NoiseClassifier:
    """
    Rule-based multi-class noise classifier.

    Current classification signals:
        - semantic_score
        - evidence_score
        - trust_score

    Important:
        Low evidence agreement alone is NOT treated as contradiction.
        Contradiction should ideally be detected using an explicit
        NLI contradiction score.
    """

    LABEL_RELEVANT = "Relevant"
    LABEL_IRRELEVANT = "Irrelevant"
    LABEL_CONTRADICTORY = "Contradictory"
    LABEL_FACTUALLY_INCORRECT = "Factually Incorrect"

    def __init__(
        self,
        relevant_trust_threshold=0.70,
        irrelevant_semantic_threshold=0.30,
        factual_error_trust_threshold=0.30,
    ):

        self.relevant_trust_threshold = relevant_trust_threshold
        self.irrelevant_semantic_threshold = irrelevant_semantic_threshold
        self.factual_error_trust_threshold = factual_error_trust_threshold

        print("Noise Classifier Ready!\n")

    def classify_sentence(self, sentence_record):
        """
        Classify one sentence record.

        Expected fields:
            semantic_score
            evidence_score
            trust_score

        Optional field:
            contradiction_score

        contradiction_score can later be produced by an NLI model.
        """

        semantic_score = float(sentence_record.get("semantic_score", 0.0))

        evidence_score = float(sentence_record.get("evidence_score", 0.0))

        trust_score = float(sentence_record.get("trust_score", 0.0))

        # -------------------------------------------------
        # 1. Explicit Contradiction Detection
        # -------------------------------------------------
        #
        # IMPORTANT:
        # Low evidence agreement is NOT contradiction.
        #
        # We only classify a sentence as contradictory if
        # an explicit contradiction score is available.
        #

        contradiction_score = sentence_record.get("contradiction_score", None)

        if contradiction_score is not None:

            contradiction_score = float(contradiction_score)

            if contradiction_score >= 0.70:
                return self.LABEL_CONTRADICTORY

        # -------------------------------------------------
        # 2. Relevant
        # -------------------------------------------------
        #
        # High semantic relevance + high overall trust.
        #

        if (
            semantic_score >= self.relevant_trust_threshold
            and trust_score >= self.relevant_trust_threshold
        ):
            return self.LABEL_RELEVANT

        # -------------------------------------------------
        # 3. Irrelevant
        # -------------------------------------------------
        #
        # Low semantic relevance means the sentence is not
        # useful for answering the query.
        #

        if semantic_score < self.irrelevant_semantic_threshold:
            return self.LABEL_IRRELEVANT

        # -------------------------------------------------
        # 4. Factually Incorrect / Low-Trust Relevant
        # -------------------------------------------------
        #
        # The sentence appears relevant to the query but
        # has very low overall trust.
        #
        # NOTE:
        # This is an operational label for the current
        # prototype. A proper factual-error detector should
        # eventually use external evidence or NLI verification.
        #

        if (
            semantic_score >= self.relevant_trust_threshold
            and trust_score < self.factual_error_trust_threshold
        ):
            return self.LABEL_FACTUALLY_INCORRECT

        # -------------------------------------------------
        # 5. Intermediate Relevant Case
        # -------------------------------------------------
        #
        # Sentence is semantically relevant but its trust
        # is between the high and low thresholds.
        #
        # We keep it as Relevant rather than falsely calling
        # it contradictory.
        #

        if semantic_score >= self.relevant_trust_threshold:
            return self.LABEL_RELEVANT

        # -------------------------------------------------
        # 6. Default
        # -------------------------------------------------

        return self.LABEL_IRRELEVANT

    def classify_sentences(self, sentence_records):
        """
        Classify all sentence records.

        The original records are updated in-place
        and returned.

        Adds:
            noise_label
        """

        for sentence_record in sentence_records:

            sentence_record["noise_label"] = self.classify_sentence(sentence_record)

        return sentence_records


# =============================================================
# Standalone Test
# =============================================================

if __name__ == "__main__":

    classifier = NoiseClassifier()

    test_records = [
        {
            "sentence_text": "Alessandro Volta invented the electric battery.",
            "semantic_score": 1.0,
            "evidence_score": 0.90,
            "trust_score": 0.90,
        },
        {
            "sentence_text": "Lionel Messi is an Argentine footballer.",
            "semantic_score": 0.10,
            "evidence_score": 0.20,
            "trust_score": 0.10,
        },
        {
            "sentence_text": "Alessandro Volta invented the electric battery in 1900.",
            "semantic_score": 0.90,
            "evidence_score": 0.20,
            "trust_score": 0.40,
        },
        {
            "sentence_text": "Volta did not invent the electric battery.",
            "semantic_score": 0.90,
            "evidence_score": 0.20,
            "trust_score": 0.40,
            "contradiction_score": 0.95,
        },
    ]

    results = classifier.classify_sentences(test_records)

    print("\nNoise Classification Results\n")

    for record in results:

        print("Sentence:", record["sentence_text"])
        print("Semantic Score:", record["semantic_score"])
        print("Evidence Score:", record["evidence_score"])
        print("Trust Score:", record["trust_score"])
        print("Noise Label:", record["noise_label"])
        print("-" * 60)
