# ARAS ChatBot

A simple, well-commented RAG (Retrieval Augmented Generation) chatbot that
answers questions about **ARAS Innovator configuration and development**
using your own uploaded documentation — instead of relying on GPT's
general training knowledge.


---

## What it does

1. You upload ARAS docs (configuration guides, development guides, method
   references — PDF, DOCX, or TXT).
2. The app extracts the text, splits it into chunks, converts each chunk
   into an embedding (numbers), and stores it in Qdrant (a vector database).
3. You ask a question like "How do I configure a Life Cycle Map in Aras?"
4. The app finds the most relevant chunks from your documents and gives
   them to GPT-4o-mini, which answers using ONLY that context — and cites
   the source document.

   ![Uploading image.png…]()


---

## Project Structure

```
aras-chatbot/
├── app/
│   ├── config.py        ← All settings (API keys, model names, chunk size)
│   ├── extractor.py      ← Reads PDF/DOCX/TXT → returns plain text
│   ├── chunker.py        ← Splits text → returns list of small chunks
│   ├── embedder.py       ← Sends text to OpenAI → returns list of numbers
│   ├── vector_store.py   ← Saves/searches chunks in Qdrant
│   ├── rag_pipeline.py   ← The brain: ties every step together
│   └── main.py           ← FastAPI server (the entry point)
│
├── frontend.py           ← Streamlit chat UI
├── requirements.txt
├── .env.example
└── HOW_IT_WORKS.txt       ← Plain-English explanation of the whole flow
```

---

## Setup

### 1. Start Qdrant (the vector database)

The easiest way is Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```
(If you don't have Docker, you can run Qdrant natively — see
https://qdrant.tech/documentation/quick-start/)

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```
Open `.env` and set:
```
OPENAI_API_KEY=sk-...your real key...
```

### 4. Start the backend (FastAPI)

```bash
uvicorn app.main:app --reload --port 8000
```
Visit **http://localhost:8000/docs** to see and test all API endpoints interactively.

### 5. Start the frontend (Streamlit) — in a new terminal

```bash
streamlit run frontend.py
```
Visit **http://localhost:8501** — upload your ARAS docs in the sidebar, then start chatting.

---

## API Endpoints (for reference)

| Method | Path                 | What it does                          |
|--------|----------------------|-----------------------------------------|
| GET    | `/health`             | Check if the server is running         |
| POST   | `/documents/upload`   | Upload + index one document            |
| GET    | `/documents`          | List all indexed documents             |
| DELETE | `/documents/{name}`   | Delete a document and its chunks       |
| POST   | `/chat`               | Ask a question, get a grounded answer  |

---

## What documents should I upload?

Any ARAS Innovator documentation you have, for example:
- ItemType / Relationship configuration guides
- Form and Grid design references
- Workflow & Life Cycle Map configuration guides
- Server-side / client-side Method development guides
- Best-practice / customization playbooks

The more specific and well-written the source document, the better the
chatbot's answers will be — it can only answer from what you give it.

---

