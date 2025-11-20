import uuid
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from database import SessionStore
from serializers import (
    CreateSessionResponse,
    ChatRequest,
    ChatResponse,
    SessionHistoryResponse,
    SessionHistoryMessage,
    ListSessionsResponse,
    DeleteSessionResponse,
)
from rag import (
    load_vector_store,
    get_retriever,
    get_conversational_rag,
    chat,
)
from utils import format_history

app = FastAPI(title="HackRF Conversational RAG")

session_store = SessionStore("sessions.db")
vector_store = load_vector_store()
retriever = get_retriever(vector_store)
rag_chain = get_conversational_rag(retriever)



@app.post("/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    session_id = session_store.create_session()
    return CreateSessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat_with_rag(req: ChatRequest) -> ChatResponse:
    try:
        history = session_store.get_history_messages(req.session_id)
        history_len = len(history)

        result = chat(rag_chain, req.query, history)
        session_store.append_messages(req.session_id, history[history_len:])

    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return ChatResponse(
        session_id=req.session_id,
        answer=result["answer"],
        chat_history=format_history(history),
    )


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    try:
        history = session_store.get_history_messages(session_id)

        messages = []
        for msg in history:
            role = "system"
            if isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            messages.append(SessionHistoryMessage(role=role, content=msg.content))

    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionHistoryResponse(session_id=session_id, history=messages)


@app.get("/sessions", response_model=ListSessionsResponse)
def list_sessions() -> ListSessionsResponse:
    return ListSessionsResponse(sessions=session_store.list_sessions())


@app.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str) -> DeleteSessionResponse:
    try:
        session_store.delete_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return DeleteSessionResponse(session_id=session_id, deleted=True)
