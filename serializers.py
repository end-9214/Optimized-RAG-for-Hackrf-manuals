from typing import List, Optional
from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="Unique session identifier")


class ChatRequest(BaseModel):
    session_id: str = Field(description="Existing session id")
    query: str = Field(description="User input message")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    chat_history: List[str]


class SessionHistoryMessage(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[SessionHistoryMessage]


class ListSessionsResponse(BaseModel):
    sessions: List[str]


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool