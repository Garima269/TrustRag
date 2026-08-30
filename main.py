"""
Standard RAG Baseline

Pipeline:

Question
    ↓
Dense Retriever
    ↓
Top-k Documents
    ↓
Context Construction
    ↓
LLM Generation
    ↓
Answer
    ↓
Optional Evaluation
"""

from src.retrieval.retriever import Retriever
from src.generation.context_builder import ContextBuilder
from src.generation.generator import AnswerGenerator
from src.evaluation.evaluate import Evaluator


def main():

    print("=" * 60)
    print("TrustRAG - Standard RAG Baseline")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Initialize Retriever
    # ---------------------------------------------------------

    print("\nInitializing Retriever...")

    retriever = Retriever()

    # ---------------------------------------------------------
    # 2. Initialize Context Builder
    # ---------------------------------------------------------

    print("Initializing Context Builder...")

    context_builder = ContextBuilder()

    # ---------------------------------------------------------
    # 3. Get User Question
    # ---------------------------------------------------------

    question = input("\nEnter your question: ").strip()

    if not question:

        print("Error: Question cannot be empty.")

        return

    # ---------------------------------------------------------
    # 4. Retrieve Documents
    # ---------------------------------------------------------

    print("\nRetrieving documents...")

    retrieved_documents = retriever.search(question, top_k=5)

    print("\n" + "=" * 60)

    print("RETRIEVED DOCUMENTS")

    print("=" * 60)

    for document in retrieved_documents:

        print(f"\nRank : " f"{document['rank']}")

        print(f"Title : " f"{document['title']}")

        print(f"Retriever Score : " f"{document['retriever_score']:.4f}")

    # ---------------------------------------------------------
    # 5. Build Standard RAG Context
    # ---------------------------------------------------------

    print("\nBuilding retrieved context...")

    context = context_builder.build_context(retrieved_documents)

    print(f"Context length : " f"{len(context)} characters")

    # ---------------------------------------------------------
    # 6. Free Retriever Memory
    # ---------------------------------------------------------

    print("\nFreeing Retriever memory...")

    del retriever

    # ---------------------------------------------------------
    # 7. Initialize Generator
    # ---------------------------------------------------------

    print("\nInitializing Answer Generator...")

    generator = AnswerGenerator()

    # ---------------------------------------------------------
    # 8. Generate Answer
    # ---------------------------------------------------------

    print("\nGenerating answer...")

    answer = generator.generate_answer(question, context)

    print("\n" + "=" * 60)

    print("GENERATED ANSWER")

    print("=" * 60)

    print(f"\n{answer}")

    # ---------------------------------------------------------
    # 9. Optional Evaluation
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
    # 10. Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 60)

    print("Standard RAG pipeline completed.")

    print("=" * 60)


if __name__ == "__main__":
    main()
