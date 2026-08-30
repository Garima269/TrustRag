"""
TrustRAG - Trust-Aware RAG Pipeline

Pipeline:

Question
    ↓
Dense Retriever
    ↓
Sentence Splitting
    ↓
Sentence-Level Trust Estimation
    ↓
Multi-Class Noise Classification
    ↓
Trust-Aware Context Reconstruction
    ↓
LLM Answer Generation
    ↓
Evaluation
"""

from src.retrieval.retriever import Retriever
from src.preprocessing.sentence_splitter import SentenceSplitter
from src.trust.trust_estimator import TrustEstimator
from src.classification.noise_classifier import NoiseClassifier
from src.classification.context_reconstructor import ContextReconstructor
from src.generation.generator import AnswerGenerator
from src.evaluation.evaluate import Evaluator


def main():

    print("=" * 60)
    print("TrustRAG - Trust-Aware RAG Pipeline")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Initialize Retriever
    # ---------------------------------------------------------

    print("\nInitializing Retriever...")
    retriever = Retriever()

    # ---------------------------------------------------------
    # 2. Initialize Sentence Splitter
    # ---------------------------------------------------------

    print("Initializing Sentence Splitter...")
    splitter = SentenceSplitter()

    # ---------------------------------------------------------
    # 3. Initialize Trust Estimator
    # ---------------------------------------------------------

    print("Initializing Trust Estimator...")
    trust_estimator = TrustEstimator()

    # ---------------------------------------------------------
    # 4. Initialize Noise Classifier
    # ---------------------------------------------------------

    print("Initializing Noise Classifier...")
    noise_classifier = NoiseClassifier()

    # ---------------------------------------------------------
    # 5. Initialize Context Reconstructor
    # ---------------------------------------------------------

    print("Initializing Context Reconstructor...")
    reconstructor = ContextReconstructor()

    # ---------------------------------------------------------
    # 6. Get Question
    # ---------------------------------------------------------

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Error: Question cannot be empty.")
        return

    # ---------------------------------------------------------
    # 7. Retrieve Documents
    # ---------------------------------------------------------

    print("\nRetrieving documents...")

    retrieved_documents = retriever.search(question, top_k=5)

    if not retrieved_documents:
        print("\nNo documents were retrieved.")
        return

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for document in retrieved_documents:

        print(f"\nRank : {document['rank']}")
        print(f"Title : {document['title']}")
        print(f"Retriever Score : " f"{document['retriever_score']:.4f}")

    # ---------------------------------------------------------
    # 8. Split Documents into Sentences
    # ---------------------------------------------------------

    print("\nSplitting retrieved documents into sentences...")

    sentence_records = splitter.split_documents(retrieved_documents)

    if not sentence_records:
        print("\nNo sentences were generated.")
        return

    # ---------------------------------------------------------
    # Limit Sentence Candidates
    # ---------------------------------------------------------
    #
    # The retrieved documents can contain hundreds of
    # sentences. Running CrossEncoder models on every
    # sentence can be very expensive.
    #
    # We therefore use a limited candidate pool for
    # the current implementation.
    # ---------------------------------------------------------

    MAX_SENTENCES = 30

    sentence_records = sentence_records[:MAX_SENTENCES]

    print(
        f"\nUsing {len(sentence_records)} sentence candidates " f"for trust estimation."
    )

    # ---------------------------------------------------------
    # 9. Compute Evidence Agreement
    # ---------------------------------------------------------

    print("\nComputing evidence agreement...")

    all_sentence_texts = [record["sentence_text"] for record in sentence_records]

    MAX_EVIDENCE_CANDIDATES = 20

    evidence_candidates = all_sentence_texts[:MAX_EVIDENCE_CANDIDATES]

    for record in sentence_records:

        other_sentences = [
            sentence
            for sentence in evidence_candidates
            if sentence != record["sentence_text"]
        ]

        record["evidence_score"] = trust_estimator.evidence.compute_score(
            record["sentence_text"], other_sentences
        )

    # ---------------------------------------------------------
    # 10. Compute Sentence Trust Scores
    # ---------------------------------------------------------

    print("\nComputing sentence trust scores...")

    max_retriever_score = max(
        (
            record["retriever_score"]
            for record in sentence_records
            if record["retriever_score"] is not None
        ),
        default=0.0,
    )

    total_documents = len(retrieved_documents)

    for record in sentence_records:

        result = trust_estimator.compute_trust_score(
            question=question,
            sentence=record["sentence_text"],
            retriever_score=record["retriever_score"],
            max_retriever_score=max_retriever_score,
            rank=record["rank"],
            total_documents=total_documents,
            evidence_score=record["evidence_score"],
        )

        record.update(result)

    # ---------------------------------------------------------
    # 11. Multi-Class Noise Classification
    # ---------------------------------------------------------

    print("\nClassifying retrieved sentences...")

    sentence_records = noise_classifier.classify_sentences(sentence_records)

    # ---------------------------------------------------------
    # 12. Display Top Sentence-Level Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP SENTENCE-LEVEL TRUST RESULTS")
    print("=" * 60)

    # Sort by semantic relevance so that the most
    # query-relevant sentences are displayed first.

    sorted_records = sorted(
        sentence_records, key=lambda x: x.get("semantic_score", 0.0), reverse=True
    )

    DISPLAY_LIMIT = 10

    for record in sorted_records[:DISPLAY_LIMIT]:

        print(f"\nSentence : " f"{record['sentence_text']}")

        print(f"Semantic Score : " f"{record['semantic_score']:.4f}")

        print(f"Evidence Score : " f"{record['evidence_score']:.4f}")

        print(f"Retriever Score : " f"{record['retriever_score']:.4f}")

        print(f"Rank Score : " f"{record['rank_score']:.4f}")

        print(f"Coverage Score : " f"{record['coverage_score']:.4f}")

        print(f"Trust Score : " f"{record['trust_score']:.4f}")

        print(f"Noise Label : " f"{record['noise_label']}")

        print("-" * 60)

    # ---------------------------------------------------------
    # 13. Reconstruct Trust-Aware Context
    # ---------------------------------------------------------

    print("\nReconstructing trust-aware context...")

    reconstruction = reconstructor.reconstruct(sentence_records)

    refined_context = reconstruction["refined_context"]

    print("\n" + "=" * 60)
    print("TRUST-AWARE CONTEXT")
    print("=" * 60)

    if refined_context.strip():

        print(refined_context)

    else:

        print(
            "No sentences satisfied the current " "trust and classification thresholds."
        )

    print(f"\nTrusted sentences : " f"{reconstruction['trusted_count']}")

    print(f"Removed sentences : " f"{reconstruction['removed_count']}")

    # ---------------------------------------------------------
    # 14. Generate Answer
    # ---------------------------------------------------------

    print("\nInitializing Answer Generator...")

    generator = AnswerGenerator()

    print("\nGenerating answer...")

    # If no trusted context remains, explicitly inform
    # the generator rather than passing an empty string.

    if refined_context.strip():

        answer = generator.generate_answer(question, refined_context)

    else:

        answer = (
            "I don't have enough trusted information "
            "in the retrieved context to answer this question."
        )

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(f"\n{answer}")

    # ---------------------------------------------------------
    # 15. Evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    ground_truth = input(
        "\nEnter ground truth answer " "(press Enter to skip evaluation): "
    ).strip()

    if ground_truth:

        evaluator = Evaluator()

        results = evaluator.evaluate(prediction=answer, ground_truth=ground_truth)

        print(f"\nExact Match : " f"{results['exact_match']}")

        print(f"F1 Score    : " f"{results['f1']:.4f}")

    else:

        print("\nEvaluation skipped.")

    # ---------------------------------------------------------
    # 16. Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TrustRAG pipeline completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
