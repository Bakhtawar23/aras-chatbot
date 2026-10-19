# app/rag_pipeline.py
# ─────────────────────────────────────────────────────────────
# THE BRAIN — ties all 5 steps together
#
# This file has two main functions:
#
# 1. ingest(file_bytes, filename)
#    Runs steps 1–4: extract → chunk → embed → store
#    Call this when a user uploads an ARAS document.
#
# 2. ask(question, chat_history)
#    Runs steps 5–8: embed question → search → build prompt → ask GPT
#    Call this when a user sends a chat message.
# ─────────────────────────────────────────────────────────────

from openai import OpenAI

from app.config import config
from app.extractor import extract_text       # Step 1
from app.chunker import split_into_chunks    # Step 2
from app.embedder import embed_many, embed_text  # Step 3
from app import vector_store                 # Step 4 (search + save)

# The OpenAI client for chat completions (generating the answer)
_client = OpenAI(api_key=config.openai_api_key)

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# This is the instruction we give GPT before every conversation.
# It tells GPT HOW to behave — tuned specifically for ARAS Innovator
# configuration and development questions.
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ARAS ChatBot, an assistant that helps engineers with
ARAS Innovator configuration and development questions.

Rules:
1. Use ONLY the information in the context below, which comes from the
   uploaded ARAS documentation (configuration guides, development guides,
   method references, etc.). Do not use outside knowledge.
2. If the answer is not in the context, say exactly:
   "I could not find that in the uploaded ARAS documents."
3. When explaining configuration steps (ItemTypes, Forms, Workflows, Life
   Cycle Maps, Relationships, Methods, etc.), keep the original step order
   if the source document lists steps.
4. Always cite which document you got the answer from.
5. Be concise and technically precise — this is for engineers, not end users.

Format your answer like this:
Answer: [your answer here]
Source: [document name]
"""


def ingest(file_bytes: bytes, filename: str) -> dict:
    """
    Process an uploaded ARAS document and store it in the vector database.

    STEP 1: extract_text()      → raw text string
    STEP 2: split_into_chunks() → list of chunk dicts
    STEP 3: embed_many()        → list of embedding vectors
    STEP 4: save_chunks()       → stored in Qdrant

    Returns a summary dict with stats about what was indexed.
    """
    print(f"\n📄 Ingesting: {filename}")

    # ── STEP 1: Extract text ──────────────────────────────────
    print("  Step 1/4: Extracting text...")
    text = extract_text(file_bytes, filename)
    print(f"  → Extracted {len(text)} characters")

    if not text.strip():
        raise ValueError("No text found in this file. Is it a scanned image PDF?")

    # ── STEP 2: Split into chunks ─────────────────────────────
    print("  Step 2/4: Splitting into chunks...")
    chunks = split_into_chunks(text, filename)
    print(f"  → Created {len(chunks)} chunks")

    # ── STEP 3: Generate embeddings ───────────────────────────
    print("  Step 3/4: Generating embeddings (calling OpenAI)...")
    texts_only = [chunk["text"] for chunk in chunks]  # extract just the text strings
    embeddings = embed_many(texts_only)
    print(f"  → Generated {len(embeddings)} embeddings (each is a {len(embeddings[0])}-number vector)")

    # ── STEP 4: Store in Qdrant ───────────────────────────────
    print("  Step 4/4: Saving to Qdrant...")
    saved_count = vector_store.save_chunks(chunks, embeddings)
    print(f"  → Saved {saved_count} chunks to Qdrant ✅")

    return {
        "filename": filename,
        "characters_extracted": len(text),
        "chunks_created": len(chunks),
        "chunks_saved": saved_count,
    }


def ask(question: str, chat_history: list[dict] = None) -> dict:
    """
    Answer a user's ARAS-related question using the RAG approach.

    STEP 5: embed the question
    STEP 6: search Qdrant for relevant chunks
    STEP 7: build a prompt with context + question
    STEP 8: call GPT and return the answer

    Args:
        question:     The user's question string
                       (e.g. "How do I configure a Life Cycle Map?")
        chat_history: Previous messages for multi-turn conversation
                      Format: [{"role": "user", "content": "..."}, ...]

    Returns:
        {
            "answer":  "Answer: To configure a Life Cycle Map...",
            "sources": [{"document_name": "Workflow_Guide.pdf", "score": 0.89, ...}]
        }
    """
    if chat_history is None:
        chat_history = []

    print(f"\n💬 Question: {question}")

    # ── STEP 5: Embed the question ────────────────────────────
    print("  Step 1/4: Embedding the question...")
    question_embedding = embed_text(question)
    # Now question_embedding is a list of 1536 numbers

    # ── STEP 6: Search Qdrant ─────────────────────────────────
    print(f"  Step 2/4: Searching for top {config.top_k} relevant chunks...")
    results = vector_store.search(question_embedding)
    print(f"  → Found {len(results)} chunks (best score: {results[0]['score'] if results else 'N/A'})")

    if not results:
        return {
            "answer": "I could not find that in the uploaded ARAS documents.",
            "sources": [],
        }

    # ── STEP 7: Build the prompt ──────────────────────────────
    print("  Step 3/4: Building prompt with retrieved context...")

    # Format the retrieved chunks as numbered context blocks
    # This is what GPT will read before answering
    context_blocks = []
    for i, result in enumerate(results, start=1):
        context_blocks.append(
            f"[Context {i}] From '{result['document_name']}' (relevance: {result['score']}):\n"
            f"{result['text']}"
        )
    context = "\n\n".join(context_blocks)

    # ── STEP 8: Call GPT ──────────────────────────────────────
    print("  Step 4/4: Calling GPT for the answer...")

    # Build the messages array that GPT expects:
    # - System message: our ARAS-specific instructions (SYSTEM_PROMPT)
    # - Previous conversation history (for multi-turn memory)
    # - Current user question with the retrieved context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *chat_history,  # unpack previous messages (if any)
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    response = _client.chat.completions.create(
        model=config.llm_model,
        messages=messages,
        temperature=0,  # 0 = deterministic, no creativity (we want factual answers)
    )

    # Extract the answer text from the response object
    answer = response.choices[0].message.content

    print(f"  → Answer generated ✅")

    return {
        "answer": answer,
        "sources": results,  # include the source chunks for citation display
    }
