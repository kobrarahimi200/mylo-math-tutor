import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualizations import VISUALIZATION_ADAPTER, choose_visualization


def test_derivative_question_selects_original_and_derivative():
    spec = choose_visualization({"question": "Explain derivatives using y = x^2", "intent": "CONCEPT"})
    assert spec["type"] == "derivative_plot"
    assert spec["config"]["expression"] == "x^2"
    assert spec["config"]["derivative_expression"] == "2*x"


def test_parameterized_function_selects_parameter_explorer():
    spec = choose_visualization({"question": "Show how a changes y = ax^2", "intent": "CONCEPT"})
    assert spec["type"] == "parameter_explorer"
    assert spec["config"]["parameters"][0]["name"] == "a"


def test_circle_question_selects_interactive_geometry():
    spec = choose_visualization({"question": "Explain area of a circle", "intent": "CONCEPT"})
    assert spec["type"] == "geometry"
    assert spec["config"]["shape"] == "circle"


def test_visualization_rejects_untrusted_expression():
    with pytest.raises(ValidationError):
        VISUALIZATION_ADAPTER.validate_python({
            "type": "function_plot",
            "title": "Invalid",
            "config": {"expression": "__import__('os').system('echo bad')"},
        })


def test_visualization_rejects_malformed_allowlisted_expression():
    with pytest.raises(ValidationError):
        VISUALIZATION_ADAPTER.validate_python({
            "type": "function_plot",
            "title": "Invalid",
            "config": {"expression": "x*/2"},
        })


def test_unknown_visualization_type_falls_back_to_none():
    spec = choose_visualization({"question": "What is 2 + 2?", "intent": "CONCEPT"})
    assert spec["type"] == "none"


def test_multiplication_concept_selects_a_concept_card():
    spec = choose_visualization({"question": "What is multiplication?", "intent": "CONCEPT", "explanation": "Repeated addition."})
    assert spec["type"] == "concept_card"


def test_show_parameter_prompt_selects_explorer_not_equation_solver():
    import agent

    result = agent.run_agent("Show how a changes y = ax^2", user_level="beginner")
    assert result["intent"] == "CONCEPT"
    assert result["visualization_type"] == "parameter_explorer"


def test_function_plot_request_is_not_misrouted_to_solver():
    import agent

    result = agent.run_agent("Plot y = sin(x)", user_level="beginner")
    assert result["intent"] == "CONCEPT"
    assert result["visualization_type"] == "function_plot"
    assert result["visualization_spec"]["config"]["expression"] == "sin(x)"
    assert "isolate the variable" not in result["final_answer"].lower()


def test_polynomial_equation_selects_graph_of_both_sides():
    import agent

    result = agent.run_agent("Solve 5x² = 676", user_level="beginner")
    assert result["visualization_type"] == "function_plot"
    assert result["visualization_spec"]["config"]["expression"] == "5*x^2"
    assert result["visualization_spec"]["config"]["second_expression"] == "676"


def test_unsupported_visualization_type_is_rejected():
    with pytest.raises(ValidationError):
        VISUALIZATION_ADAPTER.validate_python({"type": "custom_code", "title": "No", "config": {}})