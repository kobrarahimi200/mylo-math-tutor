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
    assert result["current_node"] == "select_visualization"
    assert result["debug_trace"]
    assert result["selected_branch"] in {"FINAL", "SOLVE", "CLARIFY", "VISUALIZATION"}


def test_multi_step_mathematical_solution():
    result = agent.run_agent("Solve 2x + 5 = 15", user_level="beginner")
    assert len(result["solution_steps"]) >= 2


def test_solved_equation_selects_step_visualization():
    result = agent.run_agent("Solve 2x + 5 = 15", user_level="beginner")
    assert result["visualization_type"] == "step_by_step"
    assert result["visualization_spec"]["type"] == "step_by_step"
    assert result["visualization_spec"]["config"]["steps"] == result["solution_steps"]


def test_simple_arithmetic_does_not_select_a_chart():
    result = agent.run_agent("What is 2 + 2?", user_level="beginner")
    assert result["visualization_type"] == "none"
    assert "4" in result["final_answer"]


def test_bare_arithmetic_expression_is_evaluated():
    result = agent.run_agent("2 + 2", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert result["calculation_result"] == "4"
    assert result["verification_status"] == "verified"


@pytest.mark.parametrize(
    "question",
    ["343-4 =", "Please calculate: 343 minus 4 =", "What is 343 minus 4?"],
)
def test_arithmetic_is_extracted_from_question_wording(question):
    result = agent.run_agent(question, user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert result["calculation_result"] == "339"
    assert result["verification_status"] == "verified"


def test_arithmetic_is_found_inside_longer_question_wording():
    result = agent.run_agent("what is the output if this 2304 -232 =", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert result["calculation_result"] == "2072"
    assert result["verification_status"] == "verified"


def test_modulo_is_extracted_from_question_wording():
    result = agent.run_agent("give me the answer of 12312 % 2342 =", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert result["calculation_result"] == "602"
    assert result["verification_status"] == "verified"


def test_worded_arithmetic_is_evaluated():
    result = agent.run_agent("What is 3 times 4?", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert result["calculation_result"] == "12"
    assert result["verification_status"] == "verified"


@pytest.mark.parametrize(
    ("question", "expected"),
    [("What is 10 minus 3?", "7"), ("What is 12 divided by 4?", "3")],
)
def test_worded_subtraction_and_division_are_evaluated(question, expected):
    result = agent.run_agent(question, user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert result["calculation_result"] == expected
    assert result["verification_status"] == "verified"


def test_expression_with_blank_right_side_combines_like_terms():
    result = agent.run_agent("this is 43x -30x =", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert "13x" in result["final_answer"]
    assert result["verification_status"] == "verified"


def test_definite_log_integral_is_solved_with_zeta_value():
    question = r"""Evaluate the following definite integral:
\[
I=\int_0^1\frac{\ln(x)\ln(1-x)}{x}\,dx
\]
Please solve step by step and verify numerically."""
    result = agent.run_agent(question, user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert "zeta(3)" in result["final_answer"].lower() or "ζ(3)" in result["final_answer"]
    assert "1.202056903" in result["final_answer"]
    assert result["verification_status"] == "verified"


def test_resolve_unicode_squared_equation_is_solved_and_verified():
    result = agent.run_agent("resolve integral 5x² = 222", user_level="beginner")
    assert result["intent"] == "SOLVE"
    assert "sqrt(1110)" in result["exact_calculation_result"]
    assert "6.6633" in result["calculation_result"]
    assert result["verification_status"] == "verified"


def test_irrational_roots_are_displayed_as_decimals_but_verified_exactly():
    result = agent.run_agent("Solve 5x² = 676", user_level="beginner")
    assert "sqrt" not in result["final_answer"].lower()
    assert "11.627" in result["final_answer"]
    assert "sqrt" in result["exact_calculation_result"]
    assert result["verification_status"] == "verified"


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
    assert "rate of change" not in result["final_answer"].lower()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is division?", "equal groups"),
        ("What is a fraction?", "part of a whole"),
        ("What is a prime number?", "exactly two"),
        ("What is probability?", "chance"),
        ("What is a square root?", "multiplied by itself"),
        ("What does 25 percent mean?", "out of 100"),
        ("What is the average?", "sum of the numbers"),
        ("What is slope?", "change in y"),
        ("What is algebra?", "unknown"),
        ("What is a function?", "one output"),
    ],
)
def test_common_math_concepts_have_useful_offline_answers(monkeypatch, question, expected):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.run_agent(question, user_level="beginner")
    assert result["intent"] == "CONCEPT"
    assert expected in result["final_answer"].lower()


def test_quadratic_formula_preset_gets_relevant_offline_explanation(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    question = r"Explain the quadratic formula: $$ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} $$"
    result = agent.run_agent(question, user_level="beginner")

    assert result["intent"] == "CONCEPT"
    assert "solves ax² + bx + c = 0" in result["final_answer"]
    assert "square root is a number" not in result["final_answer"]


def test_equation_explanation_does_not_use_generic_solver_fallback(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.run_agent("Explain what an equation means.", user_level="beginner")

    assert "two mathematical expressions are equal" in result["final_answer"]
    assert "isolate the variable" not in result["final_answer"]


def test_formula_followup_uses_conversation_context_when_llm_is_unavailable(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    context = [{
        "role": "user",
        "content": r"Explain the quadratic formula: $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$",
    }]

    response = agent.safe_llm_response(
        "Explain this mathematical idea.",
        "What does the discriminant tell me?",
        context=context,
    )

    assert "b² − 4ac" in response
    assert "two distinct real roots" in response


def test_derivative_of_expression_is_not_treated_as_equation(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.run_agent("Find the derivative of x^2", user_level="beginner")
    assert result["intent"] == "CONCEPT"
    assert "2*x" in result["final_answer"] or "2x" in result["final_answer"]
    assert result["visualization_type"] == "derivative_plot"


def test_missing_api_key_and_llm_error_handling(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.safe_llm_response("Explain this idea.", "What is a derivative?")
    assert isinstance(result, str)
    assert result


def test_solution_verification_for_exact_value(simple_state):
    result = agent.verify_result(simple_state)
    assert result["verification_status"] == "verified"


@pytest.mark.parametrize("label", __import__("formula_examples").FORMULA_EXAMPLES)
def test_every_preset_has_completed_offline_example(monkeypatch, label):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.run_agent(f"Give me an example of the {label}: $$f(x) = x$$")
    assert result["intent"] == "CONCEPT"
    assert "Final result:" in result["final_answer"]
    assert "1." in result["final_answer"]
    assert "This version" not in result["final_answer"]


def test_formula_example_followup_and_topic_switch(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    context = [{"role": "user", "content": "Explain the quadratic formula"}]
    example = agent.run_agent("Give me an example regarding this formula", conversation_context=context)
    assert "x = 2 or 3" in example["final_answer"]
    assert example["visualization_spec"]["config"]["expression"] == "x^2 - 5*x + 6"
    switched = agent.run_agent("Explain the circle area", conversation_context=context)
    assert "12.566" in switched["final_answer"]
    assert "Discriminant" not in switched["final_answer"]


def test_quadratic_solution_shows_calculations(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.run_agent("Solve x^2 - 7x + 12 = 0")
    assert "= 1" in result["solution_steps"][1]
    assert "3, 4" in result["final_answer"]
    assert "left − right = 0" in result["final_answer"]
    assert result["verification_status"] == "verified"


def test_verification_requires_all_roots_and_rejects_extra_roots():
    for candidate in ("2", "2, 3, 99"):
        state = {"question": "x^2 - 5x + 6 = 0", "calculation_result": candidate}
        assert agent.verify_result(state)["verification_status"] == "incorrect"


def test_supported_preset_derivative_plot_matches_example(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.run_agent("Explain the chain rule")
    assert result["visualization_spec"]["config"]["expression"] == "(3*x+1)^2"
    assert "18*x" in result["visualization_spec"]["config"]["derivative_expression"]


def test_llm_explanation_preserves_full_solution_and_requests_completion(monkeypatch):
    from types import SimpleNamespace
    prompts = []
    completed = "Let u = x. Integrate by parts. Final result: (x − 1)e^x + C. Check by differentiation."

    class Model:
        def invoke(self, prompt):
            prompts.append(str(prompt))
            return SimpleNamespace(content=completed)

    monkeypatch.setattr(agent, "get_llm", lambda: Model())
    result = agent.run_agent(r"Explain this: $$\int x e^x dx$$", "beginner")
    assert result["final_answer"] == "Answer:\n" + completed
    assert "every meaningful calculation" in prompts[0]
    assert "final result with units" in prompts[0]


def test_triangle_side_label_is_not_subtraction(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda: None)
    result = agent.run_agent("Explain the Pythagorean theorem with a 3–4–5 right triangle and a diagram")
    assert result["intent"] == "CONCEPT"
    assert result["visualization_type"] == "geometry"
    assert result["visualization_spec"]["config"]["points"] == [(0, 0), (3, 0), (3, 4)]
    assert "c = 5 cm" in result["final_answer"]
