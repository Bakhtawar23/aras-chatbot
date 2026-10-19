# app/embedder.py
# ─────────────────────────────────────────────────────────────
# PIPELINE STEP 3: Convert text into embeddings (vectors)
#
# INPUT:  a string of text
# OUTPUT: a list of 1536 floating point numbers
#
# WHAT IS AN EMBEDDING?
#   An embedding is a mathematical representation of text meaning.
#   Similar text → similar numbers → close together in vector space.
#
#   Example (conceptually):
#     "ItemType"       → [0.12, -0.45, 0.89, ...]   ← 1536 numbers
#     "Item Type"      → [0.11, -0.43, 0.91, ...]   ← very similar numbers!
#     "Life Cycle Map" → [0.89,  0.22, -0.34, ...]  ← very different numbers
#
# WHY DOES THIS MATTER?
#   When the user asks "How do I create an ItemType in Aras?",
#   we embed that question and search for chunks with SIMILAR numbers.
#   This finds semantically similar text, not just exact keyword matches —
#   so "define an ItemType" and "create an ItemType" are both found.
# ─────────────────────────────────────────────────────────────

from openai import OpenAI

from app.config import config

# Create one OpenAI client that we reuse for every call
# (avoids creating a new connection each time)
_client = OpenAI(api_key=config.openai_api_key)


def embed_text(text: str) -> list[float]:
    """
    Convert a single piece of text into an embedding vector.

    Args:
        text: Any string (a chunk, a question, etc.)

    Returns:
        A list of 1536 floats representing the text's meaning
    """
    # Clean up newlines — OpenAI recommends this for better quality
    clean_text = text.replace("\n", " ")

    # Call the OpenAI Embeddings API
    response = _client.embeddings.create(
        model=config.embedding_model,
        input=clean_text,
    )

    # The API returns a response object — we just need the numbers
    # response.data[0].embedding is a list like [0.12, -0.45, ...]
    return response.data[0].embedding


def embed_many(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple texts into embeddings in a single API call.
    This is more efficient than calling embed_text() in a loop.

    Args:
        texts: A list of strings

    Returns:
        A list of embeddings (each embedding is a list of 1536 floats)
    """
    # Clean all texts at once
    clean_texts = [t.replace("\n", " ") for t in texts]

    response = _client.embeddings.create(
        model=config.embedding_model,
        input=clean_texts,
    )

    # response.data is a list of embedding objects, one per input text
    # We extract just the numbers from each
    return [item.embedding for item in response.data]
