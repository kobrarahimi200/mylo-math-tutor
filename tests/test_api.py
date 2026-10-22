import sys
from pathlib import Path

import pytest
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
    assert "3" in response.json()["answer"]
    assert response.json()["metadata"]["verification"] == "verified"


# Read the actual picker payloads so additions to the UI enter this regression test.
def _picker_formulas():
    import json
    import re
    source = (ROOT / "frontend/src/App.tsx").read_text()
    return [(label, json.loads(latex)) for label, latex in re.findall(
        r'label: "([^"]+)", preview: "[^"]+", latex: ("(?:[^"\\]|\\.)*")', source,
    )]


@pytest.mark.parametrize("label,latex", _picker_formulas())
def test_every_picker_formula_returns_complete_answer_without_inference(monkeypatch, label, latex):
    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("Preset answers must not require model inference")
    monkeypatch.setattr(agent, "safe_llm_response", unexpected_model_call)
    response = TestClient(app).post("/api/chat", json={
        "message": f"Explain the {label.lower()}:\n\n$$\n{latex}\n$$",
        "level": "beginner",
    })
    assert response.status_code == 200
    assert "Final result:" in response.json()["answer"]


@pytest.mark.parametrize("formula,expected", [
    (r"Solve $$\frac{x+1}{x-1}=2$$", "3"),
    (r"Solve \(x^{2}-5x+6=0\)", "2, 3"),
    (r"Solve \[\sqrt{x}=3\]", "9"),
    (r"Solve $\frac{x}{\frac{1}{2}}=6$", "3"),
    (r"\frac{d}{dx}x^{3}", "3"),
    (r"\int x^{2}\,dx", "C"),
])
def test_common_formula_variations_are_solved_offline(formula, expected):
    response = TestClient(app).post("/api/chat", json={"message": formula})
    assert response.status_code == 200
    assert expected in response.json()["answer"]
    assert response.json()["metadata"]["verification"] == "verified"


@pytest.mark.parametrize("formula", [
    r"Explain the quadratic formula: $$x^2 - 9x + 20 = 0$$",
    r"Evaluate $$\int_0^1\frac{\log(1-x)\log(x)}{x}\,dx$$",
    r"Find the eigenvalues of $$\begin{pmatrix}1&2\\3&4\end{pmatrix}$$",
    r"Prove $$\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}$$",
])
def test_custom_complex_formulas_reach_general_tutor_intact(monkeypatch, formula):
    from types import SimpleNamespace
    prompts = []

    class Model:
        def invoke(self, prompt):
            prompts.append("\n".join(message.content for message in prompt))
            return SimpleNamespace(content="Complete custom worked solution.")

    monkeypatch.setattr(agent, "get_llm", lambda: Model())
    response = TestClient(app).post("/api/chat", json={"message": formula})
    assert response.status_code == 200
    assert "Complete custom worked solution" in response.json()["answer"]
    assert formula in prompts[0]
    assert response.json()["metadata"]["verification"] != "verified"


def test_picker_and_builtin_formula_catalogs_match():
    assert {label.lower(): latex for label, latex in _picker_formulas()} == agent.PRESET_LATEX
    assert set(agent.PRESET_LATEX) == set(agent.FORMULA_EXAMPLES)
