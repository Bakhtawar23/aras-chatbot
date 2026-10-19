# app/vector_store.py
# ─────────────────────────────────────────────────────────────
# PIPELINE STEP 4: Store and search ARAS document chunks in Qdrant
#
# WHAT IS QDRANT?
#   Qdrant is a vector database — it stores embeddings (lists of numbers)
#   and lets you search for the CLOSEST ones to a query vector.
#
# THINK OF IT LIKE THIS:
#   Normal database: "Find rows WHERE name = 'ItemType'"  (exact match)
#   Vector database: "Find rows SIMILAR TO this embedding" (nearest neighbor)
#
# KEY CONCEPTS:
#   Collection = a "table" in Qdrant (we use one collection for all ARAS docs)
#   Point      = one row = { id, vector (1536 numbers), payload (metadata) }
#   Search     = give me the top-K points whose vectors are closest to X
# ─────────────────────────────────────────────────────────────

import uuid  # for generating unique IDs

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,       # how to measure similarity (we use COSINE)
    PointStruct,    # the object we store: id + vector + payload
    VectorParams,   # config for the vector (size, distance metric)
)

from app.config import config

# Local mode: Qdrant runs inside our Python process and saves data to a folder.
# No Docker or separate server needed. Data persists in ./qdrant_data between restarts.
_client = QdrantClient(path="./qdrant_data")


def setup_collection() -> None:
    """
    Create the Qdrant collection if it doesn't already exist.

    A collection is like a table — it defines how vectors are stored.
    We only need to do this once (at app startup).
    """
    # Check if the collection already exists
    existing = [c.name for c in _client.get_collections().collections]

    if config.collection_name not in existing:
        _client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=config.embedding_dimensions,  # each vector has 1536 numbers
                distance=Distance.COSINE,           # COSINE similarity = measures angle between vectors
                                                    # values range from -1 (opposite) to 1 (identical)
            ),
        )
        print(f"✅ Created Qdrant collection: '{config.collection_name}'")
    else:
        print(f"✅ Qdrant collection '{config.collection_name}' already exists")


def save_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Save chunk text + embeddings into Qdrant.

    Args:
        chunks:     List of chunk dicts (from chunker.py)
                    Each dict: { text, document_name, chunk_index }
        embeddings: List of embedding vectors (from embedder.py)
                    Same length as chunks — embeddings[i] belongs to chunks[i]

    Returns:
        Number of chunks saved
    """
    # Build a list of "points" — each point is one row in Qdrant
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),   # unique ID for this chunk
                vector=embedding,       # the 1536 numbers representing this chunk's meaning
                payload={               # metadata stored alongside the vector
                    "text":          chunk["text"],
                    "document_name": chunk["document_name"],
                    "chunk_index":   chunk["chunk_index"],
                }
            )
        )

    # Upsert = insert or update (safe to call multiple times)
    _client.upsert(collection_name=config.collection_name, points=points)

    return len(points)


def search(query_embedding: list[float], top_k: int = None) -> list[dict]:
    """
    Find the chunks most similar to the query embedding.

    HOW IT WORKS:
        Qdrant compares the query vector against every stored vector
        using cosine similarity, and returns the top_k closest ones.

    Args:
        query_embedding: The embedding of the user's question
        top_k:           How many results to return (defaults to config.top_k)

    Returns:
        List of result dicts, each containing:
        {
            "text":          "...relevant ARAS doc text...",
            "document_name": "Workflow_Configuration.pdf",
            "chunk_index":   2,
            "score":         0.87  ← similarity score (higher = more relevant)
        }
    """
    k = top_k or config.top_k

    results = _client.search(
        collection_name=config.collection_name,
        query_vector=query_embedding,
        limit=k,
    )

    # Each result has: .id, .score, .payload
    # We flatten everything into a simple dict for easy use
    return [
        {
            "text":          result.payload["text"],
            "document_name": result.payload["document_name"],
            "chunk_index":   result.payload["chunk_index"],
            "score":         round(result.score, 4),  # e.g. 0.8732
        }
        for result in results
    ]


def list_documents() -> list[str]:
    """
    Return a list of unique document names stored in Qdrant.
    Used by the frontend to show what's been indexed.
    """
    # Scroll = read all points (no search, just browse)
    all_points, _ = _client.scroll(
        collection_name=config.collection_name,
        limit=10_000,       # read up to 10k points
        with_payload=True,
        with_vectors=False, # we don't need the actual numbers here
    )

    # Collect unique document names
    names = set()
    for point in all_points:
        names.add(point.payload.get("document_name", "unknown"))

    return sorted(names)


def delete_document(document_name: str) -> None:
    """
    Delete all chunks belonging to a specific document.

    Args:
        document_name: The filename to delete (e.g. "Workflow_Configuration.pdf")
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    _client.delete(
        collection_name=config.collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_name",                 # look at the payload field
                    match=MatchValue(value=document_name) # that equals this value
                )
            ]
        ),
    )


def count_chunks() -> int:
    """Return the total number of chunks stored (used for a quick stats display)."""
    info = _client.get_collection(config.collection_name)
    return info.points_count
