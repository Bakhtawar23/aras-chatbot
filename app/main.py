# app/main.py
# ─────────────────────────────────────────────────────────────
# THE ENTRY POINT — FastAPI web server
#
# This file exposes our rag_pipeline.py functions as HTTP endpoints
# so the Streamlit frontend (or Postman, curl, etc.) can talk to it.
#
# Run it with:
#   uvicorn app.main:app --reload --port 8000
#
# Then open http://localhost:8000/docs to see the interactive API docs.
# ─────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import vector_store
from app.config import config
from app.rag_pipeline import ask, ingest


# ── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the server starts.
    We use it to make sure the Qdrant collection exists before
    any requests come in.
    """
    print(f"🚀 Starting {config.app_name}...")
    vector_store.setup_collection()
    yield
    print("👋 Shutting down.")


app = FastAPI(
    title=config.app_name,
    description="RAG chatbot for ARAS Innovator configuration and development docs.",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for learning/demo purposes; restrict in real production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response schemas ──────────────────────────────────
# Pydantic models describe the JSON shape we expect / return.
# FastAPI uses these to validate input and auto-generate docs.

class ChatRequest(BaseModel):
    question: str
    chat_history: list[dict] = []   # e.g. [{"role": "user", "content": "..."}]


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    Simple endpoint to verify the server (and its dependencies) are alive.
    Useful for the frontend to show a green/red status dot.
    """
    return {"status": "ok", "app": config.app_name}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload one ARAS document (PDF/DOCX/TXT).
    Runs the full ingestion pipeline: extract → chunk → embed → store.
    """
    file_bytes = await file.read()

    try:
        result = ingest(file_bytes, file.filename)
    except ValueError as e:
        # e.g. unsupported file type, or empty document
        raise HTTPException(status_code=400, detail=str(e))

    return result


@app.get("/documents")
def get_documents():
    """
    List all ARAS documents currently indexed in Qdrant.
    """
    documents = vector_store.list_documents()
    total_chunks = vector_store.count_chunks()
    return {
        "documents": documents,
        "document_count": len(documents),
        "total_chunks": total_chunks,
    }


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    """
    Delete an ARAS document and all its chunks from Qdrant.
    """
    vector_store.delete_document(document_name)
    return {"message": f"Deleted '{document_name}'"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question about ARAS configuration/development.
    Runs the full retrieval pipeline: embed → search → prompt → GPT answer.
    """
    result = ask(request.question, request.chat_history)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
