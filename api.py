from __future__ import annotations

import os
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from agent import run_agent
import chat_history
from llm_factory import create_llm_runtime
from visualizations import VisualizationSpec


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    level: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    replace_latest: bool = Field(default=False, alias="replaceLatest")
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

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "llm": create_llm_runtime().status}


@app.post("/api/chat", response_model=ChatResponse, response_model_by_alias=True)
def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    context = chat_history.recent_context(conversation_id)
    if request.replace_latest and len(context) >= 2 and context[-2]["content"] == request.message:
        context = context[:-2]

    result = run_agent(request.message, request.level, context)
    spec = result.get("visualization_spec", {"type": "none", "title": "", "config": {}})
    visualizations = [] if spec.get("type") == "none" else [spec]
    response = ChatResponse(
        answer=result.get("final_answer", ""),
        conversation_id=conversation_id,
        visualizations=visualizations,
        steps=result.get("solution_steps", []),
        metadata={
            "llm": result.get("llm_status", {}),
            "explanationLevel": result.get("user_level", request.level),
            "intent": result.get("intent", "CLARIFY"),
            "language": result.get("language", "en"),
            "verification": result.get("verification_status", "unverified"),
            "visualizationValidation": result.get("visualization_validation", "valid"),
            "conversationMemory": True,
        },
    )
    chat_history.save_exchange(conversation_id, request.message, response.model_dump(by_alias=True), request.level, replace_latest=request.replace_latest)
    return response


@app.get("/api/conversations")
def conversations() -> list[dict[str, Any]]:
    return chat_history.list_conversations()


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str) -> dict[str, Any]:
    saved = chat_history.get_conversation(conversation_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return saved
