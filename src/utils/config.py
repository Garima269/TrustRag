"""
Central configuration file for the TrustRAG project.
"""

# --------------------------------------------------
# Retrieval Model
# --------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --------------------------------------------------
# Generator Model
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# --------------------------------------------------
# Retrieval Parameters
# --------------------------------------------------

TOP_K = 5

# --------------------------------------------------
# Generation Parameters
# --------------------------------------------------

MAX_NEW_TOKENS = 150

TEMPERATURE = 0.2

DO_SAMPLE = False