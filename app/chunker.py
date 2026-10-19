# app/chunker.py
# ─────────────────────────────────────────────────────────────
# PIPELINE STEP 2: Split text into small overlapping chunks
#
# INPUT:  one long string (the full ARAS document text)
# OUTPUT: list of smaller strings (chunks)
#
# WHY DO WE CHUNK?
#   GPT has a token limit — we can't send a 100-page ARAS configuration
#   guide all at once. We split it into pieces, store all pieces, and
#   only retrieve the RELEVANT pieces when answering a question.
#
# WHY OVERLAP?
#   Imagine the answer is: "Set the 'is_relationship' property to 'true'"
#   But "'is_relationship' property to" is at the end of chunk 1 and
#   "'true'" is at the start of chunk 2. Overlap ensures that sentence
#   isn't split across chunks and lost.
# ─────────────────────────────────────────────────────────────

from app.config import config


def split_into_chunks(text: str, document_name: str) -> list[dict]:
    """
    Split a long text into overlapping chunks with metadata attached.

    Args:
        text:          The full document text
        document_name: The filename (stored as metadata for citations later)

    Returns:
        A list of chunk dictionaries, each looking like:
        {
            "text":          "...the actual chunk text...",
            "document_name": "ItemType_Configuration_Guide.pdf",
            "chunk_index":   3   ← which chunk number this is (0-based)
        }
    """
    chunks = []
    chunk_index = 0

    # We slide a window of size `chunk_size` across the text,
    # moving forward by (chunk_size - chunk_overlap) each step.
    #
    # Example with chunk_size=10, chunk_overlap=3, text="ABCDEFGHIJKLMNOP":
    #   Chunk 0: ABCDEFGHIJ         (positions 0–9)
    #   Chunk 1: HIJKLMNOP          (positions 7–16) ← overlaps 3 chars with chunk 0
    #   Chunk 2: NOPQRSTUVW         (positions 14–23) ← overlaps 3 chars with chunk 1

    step = config.chunk_size - config.chunk_overlap  # how far to move the window each time

    start = 0
    while start < len(text):
        end = start + config.chunk_size

        # Extract this chunk of text
        chunk_text = text[start:end].strip()

        # Skip chunks that are empty or too short to be useful
        if len(chunk_text) > 50:
            chunks.append({
                "text": chunk_text,
                "document_name": document_name,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Move the window forward
        start += step

    return chunks
