"""
trust_estimator.py

Sentence-level Trust Estimation for TrustRAG.

Trust factors:
    - Semantic Relevance
    - Evidence Agreement
    - Retriever Confidence
    - Retriever Rank
    - Query Coverage

Output:
    Individual normalized trust factors
    + final sentence-level trust score
"""

from src.trust.semantic_relevance import SemanticRelevance
from src.trust.evidence_agreement import EvidenceAgreement
from src.trust.query_coverage import QueryCoverage
from src.trust.rank_score import RankScore


class TrustEstimator:
    """
    Computes a sentence-level trust score using
    multiple independent trust signals.
    """

    def __init__(
        self,
        semantic_weight=0.35,
        evidence_weight=0.25,
        retriever_weight=0.20,
        rank_weight=0.10,
        coverage_weight=0.10,
    ):

        print("Initializing Trust Estimator...\n")

        # -----------------------------------------------------
        # Trust components
        # -----------------------------------------------------

        self.semantic = SemanticRelevance()
        self.evidence = EvidenceAgreement()
        self.coverage = QueryCoverage()
        self.rank = RankScore()

        # -----------------------------------------------------
        # Weights
        # -----------------------------------------------------

        self.semantic_weight = semantic_weight
        self.evidence_weight = evidence_weight
        self.retriever_weight = retriever_weight
        self.rank_weight = rank_weight
        self.coverage_weight = coverage_weight

        total_weight = (
            semantic_weight
            + evidence_weight
            + retriever_weight
            + rank_weight
            + coverage_weight
        )

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Trust weights must sum to 1.0. " f"Current sum = {total_weight}"
            )

        print("Trust Estimator Ready!\n")

    # =========================================================
    # MAIN TRUST SCORE
    # =========================================================

    def compute_trust_score(
        self,
        question,
        sentence,
        retriever_score,
        max_retriever_score,
        rank,
        total_documents,
        evidence_score,
    ):
        """
        Compute the final sentence-level trust score.

        Parameters
        ----------
        question : str
            User query.

        sentence : str
            Retrieved sentence.

        retriever_score : float
            Original FAISS retrieval similarity.

        max_retriever_score : float
            Maximum retrieval similarity among retrieved documents.

        rank : int
            Document retrieval rank.

        total_documents : int
            Number of retrieved documents.

        evidence_score : float
            Evidence agreement score in [0,1].

        Returns
        -------
        dict
            Individual trust signals and final trust score.
        """

        # -----------------------------------------------------
        # 1. Semantic Relevance
        # -----------------------------------------------------

        semantic_score = self.semantic.compute_score(question, sentence)

        # -----------------------------------------------------
        # 2. Query Coverage
        # -----------------------------------------------------

        coverage_score = self.coverage.compute_score(question, sentence)

        # -----------------------------------------------------
        # 3. Retriever Confidence
        # -----------------------------------------------------

        retriever_confidence = self.rank.normalize_retriever_score(
            retriever_score, max_retriever_score
        )

        # -----------------------------------------------------
        # 4. Retrieval Rank
        # -----------------------------------------------------

        rank_score = self.rank.normalize_rank(rank, total_documents)

        # -----------------------------------------------------
        # 5. Make sure evidence is within [0,1]
        # -----------------------------------------------------

        evidence_score = max(0.0, min(1.0, float(evidence_score)))

        # -----------------------------------------------------
        # 6. Final Trust Score
        # -----------------------------------------------------
        #
        # Semantic Relevance    = 35%
        # Evidence Agreement    = 25%
        # Retriever Confidence  = 20%
        # Retrieval Rank        = 10%
        # Query Coverage        = 10%
        #
        # All components are expected to be normalized
        # to [0,1].
        # -----------------------------------------------------

        trust_score = (
            self.semantic_weight * semantic_score
            + self.evidence_weight * evidence_score
            + self.retriever_weight * retriever_confidence
            + self.rank_weight * rank_score
            + self.coverage_weight * coverage_score
        )

        # -----------------------------------------------------
        # 7. Safety clamp
        # -----------------------------------------------------

        trust_score = max(0.0, min(1.0, trust_score))

        # -----------------------------------------------------
        # 8. Return results
        # -----------------------------------------------------

        return {
            # Original FAISS score
            "original_retriever_score": round(float(retriever_score), 4),
            # Individual trust components
            "semantic_score": round(float(semantic_score), 4),
            "evidence_score": round(float(evidence_score), 4),
            "retriever_confidence": round(float(retriever_confidence), 4),
            "rank_score": round(float(rank_score), 4),
            "coverage_score": round(float(coverage_score), 4),
            # Final score
            "trust_score": round(float(trust_score), 4),
        }


# =============================================================
# Standalone Test
# =============================================================

if __name__ == "__main__":

    estimator = TrustEstimator()

    result = estimator.compute_trust_score(
        question="Who invented the electric battery?",
        sentence="Alessandro Volta invented the electric battery.",
        retriever_score=0.7718,
        max_retriever_score=0.7718,
        rank=1,
        total_documents=5,
        evidence_score=0.5389,
    )

    print("\n" + "=" * 60)
    print("SENTENCE TRUST SCORE")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key} : {value}")

    print("=" * 60)
