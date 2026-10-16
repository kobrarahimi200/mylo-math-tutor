from types import SimpleNamespace

import httpx
import pytest

import llm_factory as factory
import agent
from api import app
from fastapi.testclient import TestClient


def ollama_models(monkeypatch, names):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(factory.httpx, "get", lambda *a, **k: httpx.Response(200, json={"models": [{"name": n} for n in names]}, request=httpx.Request("GET", a[0])))


def test_ollama_default_needs_no_key(monkeypatch):
    ollama_models(monkeypatch, ["qwen3:4b"])
    runtime = factory.create_llm_runtime()
    assert isinstance(runtime.model, factory.ChatOllama)
    assert runtime.model.base_url == "http://localhost:11434"
    assert runtime.model.model == "qwen3:4b"
    assert runtime.status["provider"] == "ollama"


def test_missing_model(monkeypatch):
    ollama_models(monkeypatch, [])
    runtime = factory.create_llm_runtime()
    assert runtime.model is None
    assert "ollama pull qwen3:4b" in runtime.status["error"]


def test_server_unavailable_still_solves_arithmetic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    def unavailable(*a, **k):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(factory.httpx, "get", unavailable)
    response = TestClient(app).post("/api/chat", json={"message": "2 + 2"}).json()
    assert "4" in response["answer"]
    assert response["metadata"]["llm"]["provider"] == "offline"
    assert "ollama serve" in response["metadata"]["llm"]["error"]


def test_openai_remains_optional(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OPENAI_API_KEY" in factory.create_llm_runtime().status["error"]
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(factory.create_llm_runtime().model, factory.ChatOpenAI)


def test_graph_uses_configured_model_and_tracks_status(monkeypatch):
    ollama_models(monkeypatch, ["qwen3:4b"])
    calls = []
    class Model:
        def invoke(self, prompt):
            calls.append(prompt)
            return SimpleNamespace(content="A derivative measures change.")
    monkeypatch.setattr(factory, "ChatOllama", lambda **kwargs: Model())
    result = agent.run_agent("What is a derivative?")
    assert result["llm_status"]["provider"] == "ollama"
    assert result["llm_status"]["used"] is True
    assert calls
    assert result["current_node"] == "select_visualization"


def test_inference_failure_is_visible(monkeypatch):
    ollama_models(monkeypatch, ["qwen3:4b"])
    class Model:
        def invoke(self, prompt):
            raise TimeoutError()
    monkeypatch.setattr(factory, "ChatOllama", lambda **kwargs: Model())
    result = agent.run_agent("Explain the quadratic formula")
    assert result["llm_status"]["provider"] == "offline"
    assert "inference failed" in result["llm_status"]["error"]
    assert "Final result" in result["final_answer"]


@pytest.mark.parametrize("question", ["∫ 5x² dx", "integrate 5x^2 dx", r"\int 5x^2 \, dx"])
def test_integral_computes_and_verifies_without_model(question):
    result = agent.run_agent(question)
    assert result["intent"] == "SOLVE"
    assert result["calculation_result"] == "5*x**3/3 + C"
    assert result["verification_status"] == "verified"


def test_equation_and_integral_are_distinct():
    result = agent.run_agent("5x² = 676")
    assert result["verification_status"] == "verified"
    assert "11.627" in result["calculation_result"]
    assert "+ C" not in result["calculation_result"]
    assert result["visualization_type"] == "function_plot"
    integral = TestClient(app).post("/api/chat", json={"message": "∫ 5x² dx"}).json()
    assert integral["visualizations"][0]["type"] == "integral_plot"
    assert integral["visualizations"][0]["config"]["expression"] == "5*x^2"


def test_derivative_is_independently_verified():
    result = agent.run_agent("Differentiate x^3 + 2*x")
    assert result["calculation_result"] == "3*x**2 + 2"
    assert result["verification_status"] == "verified"


def test_wrong_antiderivative_fails():
    from calculus import verify_calculus
    assert not verify_calculus("integral", "5*x^2", "5*x**3")


def test_local_reasoning_is_not_rendered(monkeypatch):
    class Model:
        model = "qwen3:4b"
        def invoke(self, messages):
            assert messages[0].type == "system"
            assert messages[-1].content.endswith("/no_think")
            return SimpleNamespace(content="private scratchpad</think>Final result: 4")
    monkeypatch.setattr(agent, "get_llm", lambda: Model())
    assert agent.safe_llm_response("Explain", "2+2", context=[{"role": "user", "content": "f(x) = {x}"}]) == "Final result: 4"
