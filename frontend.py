# frontend.py
# ─────────────────────────────────────────────────────────────
# THE UI — Streamlit chat interface for ARAS ChatBot
#
# This talks to our FastAPI backend (app/main.py) over HTTP.
# It does NOT talk to OpenAI or Qdrant directly — the backend does that.
#
# Run it with:
#   streamlit run frontend.py
#
# Make sure the backend is already running on http://localhost:8000
# ─────────────────────────────────────────────────────────────

import requests
import streamlit as st

# The backend API's base URL
API_URL = "http://localhost:8000"

# ── Page setup ─────────────────────────────────────────────────
st.set_page_config(page_title="ARAS ChatBot", page_icon="🛠️", layout="wide")

st.title("🛠️ ARAS ChatBot")
st.caption("Ask questions about ARAS Innovator configuration and development — answered from your uploaded docs.")


# ── Session state ──────────────────────────────────────────────
# Streamlit re-runs this whole script on every interaction, so we
# use session_state to remember things between reruns (like chat history).
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": "user"/"assistant", "content": "..."}


# ── Helper functions (thin wrappers around the API) ─────────────

def api_health() -> bool:
    """Check if the backend is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def api_upload(file) -> dict:
    """Upload a file to the backend for ingestion."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    r = requests.post(f"{API_URL}/documents/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def api_list_documents() -> dict:
    """Get the list of indexed documents."""
    r = requests.get(f"{API_URL}/documents", timeout=10)
    r.raise_for_status()
    return r.json()


def api_delete_document(name: str) -> None:
    """Delete a document by name."""
    r = requests.delete(f"{API_URL}/documents/{name}", timeout=10)
    r.raise_for_status()


def api_chat(question: str, history: list[dict]) -> dict:
    """Send a question + chat history to the backend and get an answer."""
    payload = {"question": question, "chat_history": history}
    r = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Sidebar: status + document management ───────────────────────

with st.sidebar:
    st.header("⚙️ Status")

    if api_health():
        st.success("Backend connected ✅")
    else:
        st.error("Backend not reachable. Start it with:\n`uvicorn app.main:app --reload`")

    st.divider()

    st.header("📁 Upload ARAS Documents")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file and st.button("Upload & Index", use_container_width=True):
        with st.spinner(f"Processing '{uploaded_file.name}'... (extract → chunk → embed → store)"):
            try:
                result = api_upload(uploaded_file)
                st.success(
                    f"✅ Indexed '{result['filename']}'\n\n"
                    f"- Characters extracted: {result['characters_extracted']}\n"
                    f"- Chunks created: {result['chunks_created']}"
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")

    st.divider()

    st.header("📚 Indexed Documents")
    try:
        doc_data = api_list_documents()
        st.caption(f"{doc_data['document_count']} document(s) · {doc_data['total_chunks']} chunks total")

        for doc_name in doc_data["documents"]:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc_name}")
            if col2.button("🗑️", key=f"delete_{doc_name}"):
                api_delete_document(doc_name)
                st.rerun()

    except requests.exceptions.RequestException:
        st.caption("Could not load document list.")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────

# Display all previous messages in the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # If this is an assistant message with sources, show them
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📖 Sources used"):
                for src in message["sources"]:
                    st.markdown(
                        f"- **{src['document_name']}** "
                        f"(chunk #{src['chunk_index']}, relevance score: {src['score']})"
                    )

# Chat input box (always pinned to the bottom by Streamlit)
user_question = st.chat_input("Ask something about ARAS configuration or development...")

if user_question:
    # 1. Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Build chat history in the format the API expects (role/content only, no sources)
    history_for_api = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude the question we just added
    ]

    # 3. Call the backend and show the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching ARAS docs and thinking..."):
            try:
                result = api_chat(user_question, history_for_api)
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("📖 Sources used"):
                        for src in result["sources"]:
                            st.markdown(
                                f"- **{src['document_name']}** "
                                f"(chunk #{src['chunk_index']}, relevance score: {src['score']})"
                            )

                # Save assistant reply (with sources) to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except requests.exceptions.RequestException as e:
                error_msg = f"Error reaching backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
