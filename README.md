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

  <img width="1919" height="879" alt="image" src="https://github.com/user-attachments/assets/490dd5fe-7b4a-4eb2-8e2c-49bfbec6742d" />



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
└── HOW_IT_WORKS.txt     
```

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

