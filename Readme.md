# HackRF Conversational RAG

Lightweight FastAPI service that exposes a conversational retrieval‑augmented generation workflow for HackRF documents.

---

## Project layout

- Service entrypoint: main.py – wires FastAPI routes to the RAG pipeline.
- RAG utilities: rag.py – loads documents, builds the Chroma vector store, and defines the conversational chain.
- SQLite session/message store: database.py – persists chat history for each session.
- API models: serializers.py – Pydantic schemas for every endpoint.
- Response formatting helper: utils.py.
- Dependencies: requirements.txt.
- Data sources live in documents and the persisted embeddings in chroma_db.

---

## Installation

````bash
python -m venv .venv
. .venv/Scripts/activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt
````

Initialize the vector store once (rerun when documents change):

````bash
python -c "from rag import create_vector_store; create_vector_store('./documents')"
````

---

## Running the API

````bash
uvicorn main:app --reload
````

Open http://localhost:8000/docs for interactive OpenAPI docs.
