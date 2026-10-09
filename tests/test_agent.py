import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent
import cli


@pytest.fixture
def simple_state():
    return {
        "question": "Solve 2x + 5 = 15",
        "user_level": "beginner",
        "language": "en",
        "intent": "SOLVE",
        "math_concept": "algebra",
        "strategy": "Isolate x.",
        "solution_steps": ["Step 1", "Step 2"],
        "current_step": 0,
        "has_more_steps": True,
        "calculation_result": "5",
        "verification_status": "unverified",
        "explanation": "",
        "clarity_score": 0,
        "retry_count": 0,
        "max_retries": 3,
        "errors": [],
        "final_answer": "",
        "current_node": "",
        "selected_branch": "",
        "debug_trace": [],
        "is_debug": False,
    }


def test_concept_question_routing():
    result = agent.run_agent("What is a derivative?", user_level="beginner")
    assert result["intent"] == "CONCEPT"
    assert "derivative" in result["final_answer"].lower() or "rate of change" in result["final_answer"].lower()


def test_solve_question_routing():
    result = agent.run_agent("Solve 2x + 5 = 15", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert "5" in result["final_answer"] or "Result: 5" in result["final_answer"]


def test_clarification_routing():
    result = agent.run_agent("Solve this equation.", user_level="beginner")
    assert result["intent"] == "CLARIFY"
    assert "equation" in result["final_answer"].lower() or "question" in result["final_answer"].lower()


def test_beginner_vs_advanced_explanation():
    beginner_state = {"question": "What is a derivative?", "user_level": "beginner", "language": "en"}
    advanced_state = {"question": "What is a derivative?", "user_level": "advanced", "language": "en"}
    beginner = agent.explain_concept(beginner_state)
    advanced = agent.explain_concept(advanced_state)
    assert "speed" in beginner["explanation"].lower() or "rate of change" in beginner["explanation"].lower()
    assert "formal" in advanced["explanation"].lower() or "notation" in advanced["explanation"].lower()


def test_state_updates_between_nodes():
    graph = agent.build_graph()
    state = agent.create_initial_state("Solve 2x + 5 = 15", "beginner")
    result = graph.invoke(state)
    assert result["current_node"] == "finalize_answer"
    assert result["debug_trace"]
    assert result["selected_branch"] in {"FINAL", "SOLVE", "CLARIFY"}


def test_multi_step_mathematical_solution():
    result = agent.run_agent("Solve 2x + 5 = 15", user_level="beginner")
    assert len(result["solution_steps"]) >= 2


def test_successful_sympy_verification():
    state = {"question": "2x + 5 = 15", "calculation_result": "5", "verification_status": "unverified"}
    result = agent.verify_result(state)
    assert result["verification_status"] == "verified"


def test_failed_verification_and_retry():
    state = {"question": "2x + 5 = 15", "calculation_result": "4", "retry_count": 0, "max_retries": 3}
    after_verify = agent.verify_result(state)
    assert after_verify["verification_status"] == "incorrect"
    after_retry = agent.retry_strategy(after_verify)
    assert after_retry["retry_count"] == 1
    assert after_retry["selected_branch"] == "RETRY"


def test_retry_limit_enforcement():
    state = {"retry_count": 3, "max_retries": 3, "verification_status": "incorrect"}
    result = agent.retry_strategy(state)
    assert result["selected_branch"] == "FINAL"


def test_cli_commands_and_exit_behavior():
    session = {"user_level": "beginner", "debug": False}
    assert cli.handle_command("/level advanced", session) is False
    assert session["user_level"] == "advanced"
    assert cli.handle_command("/exit", session) is True


def test_persian_language_support():
    result = agent.run_agent("انتگرال را خیلی ساده توضیح بده", user_level="beginner")
    assert result["language"] == "fa"
    assert "انتگرال" in result["final_answer"] or "مشتق" in result["final_answer"] or "پاسخ" in result["final_answer"]


def test_direct_equation_without_solve_keyword_is_solved():
    result = agent.run_agent("2x + 5 = 15", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert "5" in result["final_answer"]


def test_persian_concept_prompt_routes_to_concept():
    result = agent.run_agent("انتگرال را خیلی ساده توضیح بده", user_level="beginner")
    assert result["intent"] == "CONCEPT"
    assert "انتگرال" in result["final_answer"]


def test_multiplication_question_explains_multiplication():
    result = agent.run_agent("what is multiply?", user_level="beginner")
    assert result["intent"] == "CONCEPT"
    assert "multiply" in result["final_answer"].lower() or "multiplication" in result["final_answer"].lower()


def test_missing_api_key_and_llm_error_handling(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.safe_llm_response("Explain this idea.", "What is a derivative?")
    assert isinstance(result, str)
    assert result


def test_solution_verification_for_exact_value(simple_state):
    result = agent.verify_result(simple_state)
    assert result["verification_status"] == "verified"
