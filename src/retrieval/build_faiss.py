import faiss
import numpy as np
from pathlib import Path


EMBEDDINGS_PATH = "data/indexes/wiki_embeddings.npy"
INDEX_PATH = "data/indexes/wiki.index"


print("=" * 60)
print("Building FAISS Index")
print("=" * 60)


print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDINGS_PATH
)

print(
    f"Embeddings shape: {embeddings.shape}"
)


print("\nCreating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)


print("Adding embeddings to index...")

index.add(
    embeddings.astype("float32")
)


print(
    f"Total vectors in index: {index.ntotal}"
)


Path(INDEX_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)


faiss.write_index(
    index,
    INDEX_PATH
)


print("\nFAISS index created successfully!")

print(
    f"Index saved to: {INDEX_PATH}"
)