from __future__ import annotations

import os
from threading import RLock
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from agent import run_agent
from visualizations import VisualizationSpec


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    conversation_id: Optional[str] = Field(default=None, alias="conversationId", max_length=80)


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str = Field(alias="conversationId")
    visualizations: list[VisualizationSpec] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


app = FastAPI(title="Mylo Mathematics Tutor API", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_conversations: dict[str, list[dict[str, str]]] = {}
_conversation_lock = RLock()
_MAX_CONVERSATIONS = 256
_MAX_TURNS = 12


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse, response_model_by_alias=True)
def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    with _conversation_lock:
        context = list(_conversations.get(conversation_id, []))

    result = run_agent(request.message, request.level, context)
    spec = result.get("visualization_spec", {"type": "none", "title": "", "config": {}})
    visualizations = [] if spec.get("type") == "none" else [spec]
    turns = context + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": result.get("final_answer", "")},
    ]
    with _conversation_lock:
        if conversation_id not in _conversations and len(_conversations) >= _MAX_CONVERSATIONS:
            _conversations.pop(next(iter(_conversations)))
        _conversations[conversation_id] = turns[-_MAX_TURNS:]

    return ChatResponse(
        answer=result.get("final_answer", ""),
        conversation_id=conversation_id,
        visualizations=visualizations,
        steps=result.get("solution_steps", []),
        metadata={
            "intent": result.get("intent", "CLARIFY"),
            "language": result.get("language", "en"),
            "verification": result.get("verification_status", "unverified"),
            "visualizationValidation": result.get("visualization_validation", "valid"),
            "conversationMemory": True,
        },
    )