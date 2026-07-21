from src.utils.logger import get_logger
from src.utils.constants import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    DEFAULT_TOP_K
)

logger = get_logger(__name__)

logger.info("Testing TrustRAG utilities")

print("\nConstants:")
print("Embedding Model:", EMBEDDING_MODEL)
print("Embedding Dimension:", EMBEDDING_DIM)
print("Default Top-K:", DEFAULT_TOP_K)

logger.info("Utility test completed successfully")