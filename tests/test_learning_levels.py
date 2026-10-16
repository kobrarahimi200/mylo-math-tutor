from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import agent
from api import app
from learning_levels import LEVEL_GUIDANCE


@pytest.mark.parametrize("question", ["Solve 2x + 5 = 15", "5x² = 676", "∫ 5x² dx", "Differentiate x^3 + 2*x", "Explain the quadratic formula", "Explain the binomial theorem"])
def test_offline_explanations_differ_by_level_without_changing_the_result(question):
    answers = [agent.run_agent(question, level) for level in LEVEL_GUIDANCE]
    assert len({result["final_answer"] for result in answers}) == 3
    assert len({result["calculation_result"] for result in answers}) == 1
    assert len({result["verification_status"] for result in answers}) == 1
    assert "formal" in answers[2]["final_answer"].lower()


def test_beginner_explains_each_operation_advanced_states_restrictions():
    beginner = agent.run_agent("Solve 2x + 5 = 15", "beginner")
    advanced = agent.run_agent("Solve 2x + 5 = 15", "advanced")
    assert "Division undoes multiplication" in beginner["final_answer"]
    assert "nonzero divisor" in advanced["final_answer"]
    assert "Division undoes multiplication" not in advanced["final_answer"]


@pytest.mark.parametrize("question", ["What is a derivative?", "Solve 2x + 5 = 15", "Differentiate x^3"])
def test_model_receives_detailed_policy_for_each_level(monkeypatch, question):
    prompts = []
    class Model:
        def invoke(self, messages):
            prompts.append(messages[-1].content)
            return SimpleNamespace(content="A complete explanation.")
    monkeypatch.setattr(agent, "get_llm", lambda: Model())
    for level, policy in LEVEL_GUIDANCE.items():
        agent.run_agent(question, level)
        assert policy in prompts[-1]
        assert "overrides the style of earlier conversation" in prompts[-1]


def test_api_reports_level_and_preserves_math_across_level_switch_in_conversation():
    client = TestClient(app)
    first = client.post("/api/chat", json={"message": "Solve 2x + 5 = 15", "level": "beginner"}).json()
    second = client.post("/api/chat", json={"message": "Solve 2x + 5 = 15", "level": "advanced", "conversationId": first["conversationId"]}).json()
    assert first["metadata"]["explanationLevel"] == "beginner"
    assert second["metadata"]["explanationLevel"] == "advanced"
    assert first["answer"] != second["answer"]
    assert first["metadata"]["verification"] == second["metadata"]["verification"] == "verified"
