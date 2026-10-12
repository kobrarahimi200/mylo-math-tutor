import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent
from api import app


def test_chat_endpoint_uses_langgraph_and_returns_visualization(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "Solve 2x + 5 = 15", "level": "beginner"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["conversationId"]
    assert body["visualizations"][0]["type"] == "step_by_step"
    assert body["steps"]
    assert body["metadata"]["conversationMemory"] is True


def test_chat_keeps_conversation_scoped_followup_context(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    client = TestClient(app)
    first = client.post("/api/chat", json={"message": "What is a derivative?", "level": "beginner"}).json()
    second = client.post("/api/chat", json={
        "message": "Why?",
        "level": "beginner",
        "conversationId": first["conversationId"],
    })
    assert second.status_code == 200
    assert second.json()["metadata"]["intent"] == "CONCEPT"


def test_chat_rejects_invalid_request():
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "", "level": "beginner"})
    assert response.status_code == 422