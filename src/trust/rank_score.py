class RankScore:

    def __init__(self):

        print("Rank Score Module Ready!\n")

    def normalize_retriever_score(self, retriever_score, max_score):

        if max_score == 0:
            return 0.0

        return round(retriever_score / max_score, 4)

    def normalize_rank(self, rank, total_documents):

        if total_documents <= 1:
            return 1.0

        score = (total_documents - rank + 1) / total_documents

        return round(score, 4)


if __name__ == "__main__":

    rank_module = RankScore()

    retriever_score = 0.7718
    highest_score = 0.7718

    normalized_similarity = rank_module.normalize_retriever_score(
        retriever_score, highest_score
    )

    rank = 2
    total_documents = 5

    normalized_rank = rank_module.normalize_rank(rank, total_documents)

    print("Normalized Retriever Score\n")
    print(normalized_similarity)

    print("\nNormalized Rank Score\n")
    print(normalized_rank)
