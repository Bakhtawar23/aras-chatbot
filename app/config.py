# app/config.py
# ─────────────────────────────────────────────────────────────
# All configuration lives here.
# We read values from environment variables (your .env file).
# If a variable is missing, we use the default value shown.
#
# HOW TO USE:
#   from app.config import config
#   print(config.openai_api_key)
# ─────────────────────────────────────────────────────────────

import os
from dataclasses import dataclass

# python-dotenv loads your .env file into environment variables
from dotenv import load_dotenv

# Load the .env file from the project root
load_dotenv()


@dataclass
class Config:
    # ── OpenAI ────────────────────────────────────────────────
    # Your OpenAI API key — NEVER hardcode this, always use .env
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Which GPT model to use for answering questions
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")  # mini = cheaper for learning

    # Which model to use for converting text → numbers (embeddings)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # How many numbers represent each chunk (1536 is the size for text-embedding-3-small)
    embedding_dimensions: int = 1536

    # ── Qdrant (Vector Database) ──────────────────────────────
    # Where Qdrant is running (localhost if you run `docker run qdrant/qdrant`)
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    # The "table" name inside Qdrant where we store ARAS document chunks
    collection_name: str = os.getenv("COLLECTION_NAME", "aras_documents")

    # ── Chunking ──────────────────────────────────────────────
    # How many characters per chunk (smaller = more precise, larger = more context)
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))

    # How many characters overlap between consecutive chunks
    # WHY OVERLAP? So that sentences that cross chunk boundaries aren't lost
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # ── Retrieval ─────────────────────────────────────────────
    # How many chunks to retrieve when answering a question
    top_k: int = int(os.getenv("TOP_K", "5"))

    # ── App identity ──────────────────────────────────────────
    app_name: str = "ARAS ChatBot"


# Create one global instance — import this everywhere
config = Config()
