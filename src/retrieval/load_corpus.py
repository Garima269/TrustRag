from datasets import load_dataset
import pandas as pd
from pathlib import Path


OUTPUT_PATH = Path("data/raw/wiki_real.csv")

NUM_DOCUMENTS = 10000


print("=" * 60)
print("Loading Real Wikipedia Corpus")
print("=" * 60)


print("\nDownloading Wikipedia dataset...")

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)


print("Reading Wikipedia passages...")

documents = []

for idx, item in enumerate(dataset):

    documents.append({
        "id": idx,
        "title": item["title"],
        "text": item["text"]
    })

    if len(documents) >= NUM_DOCUMENTS:
        break


print(f"\nDocuments collected: {len(documents)}")


df = pd.DataFrame(documents)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nWikipedia corpus saved successfully!")
print(f"Path: {OUTPUT_PATH}")
print(f"Documents: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst document:")
print(df.iloc[0])