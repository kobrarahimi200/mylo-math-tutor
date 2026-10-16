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


def test_chat_handles_formula_only_latex_prompt(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    client = TestClient(app)
    response = client.post("/api/chat", json={
        "message": "$$\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$",
        "level": "beginner",
    })

    assert response.status_code == 200
    assert response.json()["answer"]
    assert response.json()["metadata"]["intent"] == "CONCEPT"


def test_chat_answers_formula_followup_using_conversation_context(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    client = TestClient(app)
    first = client.post("/api/chat", json={
        "message": "Explain the quadratic formula: $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$",
        "level": "beginner",
    }).json()

    followup = client.post("/api/chat", json={
        "message": "What does the discriminant tell me?",
        "level": "beginner",
        "conversationId": first["conversationId"],
    })

    assert followup.status_code == 200
    assert "b² − 4ac" in followup.json()["answer"]
    assert "two distinct real roots" in followup.json()["answer"]

    coefficients = client.post("/api/chat", json={
        "message": "What do a, b, and c mean?",
        "level": "beginner",
        "conversationId": first["conversationId"],
    })
    assert coefficients.status_code == 200
    assert "a multiplies x²" in coefficients.json()["answer"]
    assert "b = −5" in coefficients.json()["answer"]

    plus_minus = client.post("/api/chat", json={
        "message": "Why are there two answers?",
        "level": "beginner",
        "conversationId": first["conversationId"],
    })
    assert plus_minus.status_code == 200
    assert "calculate the expression twice" in plus_minus.json()["answer"]


def test_chat_rejects_invalid_request():
    client = TestClient(app)
    response = client.post("/api/chat", json={"message": "", "level": "beginner"})
    assert response.status_code == 422

def test_selected_formula_example_followup_is_complete(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    client = TestClient(app)
    first = client.post("/api/chat", json={
        "message": "Explain the geometric series: $$S_n = a(1-r^n)/(1-r)$$",
    }).json()
    followup = client.post("/api/chat", json={
        "message": "Give me an example regarding this formula",
        "conversationId": first["conversationId"],
    })
    assert followup.status_code == 200
    assert "Final result: 80" in followup.json()["answer"]
    assert "2(−80)/(−2) = 80" in followup.json()["answer"]


def test_custom_latex_solve_request_does_not_crash(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    response = TestClient(app).post("/api/chat", json={
        "message": r"Solve this equation: $$\frac{x+1}{x-1}=2$$",
    })
    assert response.status_code == 200
    assert "offline" in response.json()["answer"]
