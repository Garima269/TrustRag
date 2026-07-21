import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

print("=" * 60)
print("Building Embeddings")
print("=" * 60)

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
CORPUS_PATH = "data/raw/wiki_real.csv"
df = pd.read_csv(CORPUS_PATH)

texts = df["text"].tolist()

print(f"Documents: {len(texts)}")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

os.makedirs("data/indexes", exist_ok=True)

np.save("data/indexes/wiki_embeddings.npy", embeddings)

print("\nDone!")

print("Shape:", embeddings.shape)
