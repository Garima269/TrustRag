
class ContextReconstructor:
   
    REMOVED_LABELS = {
        "Irrelevant",
        "Contradictory",
        "Factually Incorrect",
    }

    def __init__(self, minimum_trust_score=0.50, max_sentences=None):

        self.minimum_trust_score = minimum_trust_score
        self.max_sentences = max_sentences

        print("Context Reconstructor Ready!\n")

    def should_keep(self, sentence_record):
        """
        Decide whether a sentence should be retained.

        A sentence is retained only when:
            1. Its noise label is Relevant.
            2. Its trust score meets the minimum threshold.
        """

        noise_label = sentence_record.get("noise_label", "Irrelevant")

        trust_score = float(sentence_record.get("trust_score", 0.0))

        # Explicitly remove noisy classes
        if noise_label in self.REMOVED_LABELS:
            return False

        # Keep only sufficiently trusted Relevant sentences
        if noise_label == "Relevant":
            return trust_score >= self.minimum_trust_score

        return False

    def reconstruct_context(self, sentence_records):
        """
        Return only the trusted sentence text as a context string.
        """

        trusted_sentences = []

        for sentence_record in sentence_records:

            if not self.should_keep(sentence_record):
                continue

            sentence_text = sentence_record.get(
                "sentence_text", sentence_record.get("text", "")
            ).strip()

            if not sentence_text:
                continue

            trusted_sentences.append(sentence_text)

        # Apply maximum context limit AFTER collecting
        # trusted sentences.
        if self.max_sentences is not None:
            trusted_sentences = trusted_sentences[: self.max_sentences]

        return "\n".join(trusted_sentences)

    def reconstruct(self, sentence_records):
        """
        Reconstruct the final trust-aware context.

        Returns:
            refined_context
            trusted_sentences
            removed_sentences
            trusted_count
            removed_count
        """

        trusted_sentences = []
        removed_sentences = []

        for sentence_record in sentence_records:

            sentence_text = sentence_record.get(
                "sentence_text", sentence_record.get("text", "")
            ).strip()

            if not sentence_text:
                continue

            if self.should_keep(sentence_record):
                trusted_sentences.append(sentence_text)
            else:
                removed_sentences.append(sentence_text)

        # Limit only the final trusted context
        context_sentences = trusted_sentences

        if self.max_sentences is not None:
            context_sentences = trusted_sentences[: self.max_sentences]

        refined_context = "\n".join(context_sentences)

        return {
            "refined_context": refined_context,
            "trusted_sentences": context_sentences,
            "removed_sentences": removed_sentences,
            "trusted_count": len(context_sentences),
            "removed_count": len(removed_sentences),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    reconstructor = ContextReconstructor()

    test_records = [
        {
            "sentence_text": "Alessandro Volta invented the electric battery.",
            "noise_label": "Relevant",
            "trust_score": 0.90,
        },
        {
            "sentence_text": "Lionel Messi is an Argentine footballer.",
            "noise_label": "Irrelevant",
            "trust_score": 0.10,
        },
        {
            "sentence_text": "Alessandro Volta invented the battery in 1900.",
            "noise_label": "Contradictory",
            "trust_score": 0.40,
        },
    ]

    result = reconstructor.reconstruct(test_records)

    print("\nContext Reconstruction Results\n")

    print("Refined Context:")
    print(result["refined_context"])

    print("\nTrusted Sentences:")
    print(result["trusted_sentences"])

    print("\nRemoved Sentences:")
    print(result["removed_sentences"])

    print("\nTrusted Count:", result["trusted_count"])
    print("Removed Count:", result["removed_count"])
