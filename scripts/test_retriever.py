from src.retrieval.retriever import Retriever

retriever = Retriever()

question = "Who developed ChatGPT?"

results = retriever.search(question, top_k=3)

print("=" * 60)
print("Question:")
print(question)

print("\nRetrieved Passages\n")

for i, result in enumerate(results, start=1):

    print(f"{i}. {result['title']}")
    print(f"Rank : {result['rank']}")
    print(f"Retriever Score : {result['retriever_score']:.4f}")
    print(result["text"])
    print("-" * 60)
