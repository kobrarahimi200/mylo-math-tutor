from __future__ import annotations

import re
from typing import Annotated, Literal, Optional, Union

import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlotParameter(StrictModel):
    name: Literal["a", "b", "c"]
    label: str
    minimum: float = -5
    maximum: float = 5
    step: float = 0.1
    default: float = 1


def validate_expression(expression: str) -> str:
    normalized = expression.strip().replace("^", "^")
    if not normalized or len(normalized) > 120:
        raise ValueError("Expression is empty or too long")
    if not re.fullmatch(r"[A-Za-z0-9_\s+\-*/^().,]+", normalized):
        raise ValueError("Expression contains unsupported characters")
    tokens = re.findall(r"[A-Za-z_]\w*|(?:\d+(?:\.\d*)?|\.\d+)|[+\-*/^().,]", normalized)
    if "".join(tokens).replace(" ", "") != re.sub(r"\s+", "", normalized):
        raise ValueError("Expression could not be tokenized")
    allowed_names = {"x", "a", "b", "c", "sin", "cos", "tan", "sqrt", "abs", "exp", "log", "pi"}
    names = re.findall(r"[A-Za-z_]\w*", normalized)
    if any(name not in allowed_names for name in names):
        raise ValueError("Expression contains an unsupported symbol or function")
    symbols = {name: sp.Symbol(name) for name in ("x", "a", "b", "c")}
    symbols.update({
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "exp": sp.exp,
        "log": sp.log,
        "pi": sp.pi,
    })
    try:
        parsed = sp.sympify(normalized.replace("^", "**"), locals=symbols, evaluate=True)
        if not isinstance(parsed, sp.Expr) or any(symbol.name not in {"x", "a", "b", "c"} for symbol in parsed.free_symbols):
            raise ValueError("Expression contains an unsupported symbol")
    except Exception as error:
        raise ValueError("Expression is not valid mathematics") from error
    return normalized


class FunctionPlotConfig(StrictModel):
    expression: str
    second_expression: Optional[str] = None
    second_label: Optional[str] = None
    domain: tuple[float, float] = (-10, 10)
    parameters: list[PlotParameter] = Field(default_factory=list)

    @field_validator("expression")
    @classmethod
    def expression_is_allowlisted(cls, value: str) -> str:
        return validate_expression(value)

    @field_validator("second_expression")
    @classmethod
    def second_expression_is_allowlisted(cls, value: Optional[str]) -> Optional[str]:
        return validate_expression(value) if value is not None else value


class DerivativePlotConfig(FunctionPlotConfig):
    derivative_expression: str

    @field_validator("derivative_expression")
    @classmethod
    def derivative_is_allowlisted(cls, value: str) -> str:
        return validate_expression(value)


class IntegralPlotConfig(FunctionPlotConfig):
    bounds: tuple[float, float] = (0, 2)


class GeometryConfig(StrictModel):
    shape: Literal["circle", "triangle", "angle", "coordinates"]
    radius: float = Field(default=2, gt=0, le=100)
    points: list[tuple[float, float]] = Field(default_factory=list, max_length=6)


class StepByStepConfig(StrictModel):
    steps: list[str] = Field(default_factory=list, max_length=30)
    result: str = ""
    verification: str = "unverified"


class ConceptCardConfig(StrictModel):
    explanation: str
    formula: str = ""
    example: str = ""


class EmptyConfig(StrictModel):
    pass


class FunctionPlotSpec(StrictModel):
    type: Literal["function_plot"]
    title: str
    config: FunctionPlotConfig


class DerivativePlotSpec(StrictModel):
    type: Literal["derivative_plot"]
    title: str
    config: DerivativePlotConfig


class IntegralPlotSpec(StrictModel):
    type: Literal["integral_plot"]
    title: str
    config: IntegralPlotConfig


class GeometrySpec(StrictModel):
    type: Literal["geometry"]
    title: str
    config: GeometryConfig


class StepByStepSpec(StrictModel):
    type: Literal["step_by_step"]
    title: str
    config: StepByStepConfig


class ConceptCardSpec(StrictModel):
    type: Literal["concept_card"]
    title: str
    config: ConceptCardConfig


class ParameterExplorerSpec(StrictModel):
    type: Literal["parameter_explorer"]
    title: str
    config: FunctionPlotConfig


class NoneSpec(StrictModel):
    type: Literal["none"] = "none"
    title: str = ""
    config: EmptyConfig = Field(default_factory=EmptyConfig)


VisualizationSpec = Annotated[
    Union[
        FunctionPlotSpec,
        DerivativePlotSpec,
        IntegralPlotSpec,
        GeometrySpec,
        StepByStepSpec,
        ConceptCardSpec,
        ParameterExplorerSpec,
        NoneSpec,
    ],
    Field(discriminator="type"),
]
VISUALIZATION_ADAPTER = TypeAdapter(VisualizationSpec)


def _expression_from_question(question: str, default: str = "x^2") -> str:
    normalized = question.replace("²", "^2").replace("³", "^3")
    match = re.search(r"(?:y|f\(x\))\s*=\s*([^?.!,;]+)", normalized, flags=re.IGNORECASE)
    candidate = match.group(1).strip() if match else default
    candidate = re.sub(r"(?<=\d)(?=x\b)|(?<=\d)(?=[abc]\b)|(?<=[abc])(?=x\b)", "*", candidate, flags=re.IGNORECASE)
    try:
        return validate_expression(candidate)
    except ValueError:
        return default


def _derivative(expression: str) -> str:
    try:
        parsed = sp.sympify(expression.replace("^", "**"), locals={"x": sp.Symbol("x")}, evaluate=True)
        result = sp.diff(parsed, sp.Symbol("x"))
        text = str(result).replace("**", "^")
        return validate_expression(text)
    except Exception:
        return "2*x"


def _none() -> dict:
    return {"type": "none", "title": "", "config": {}}


def _polynomial_equation_plot(question: str) -> Optional[dict]:
    normalized = question.lower().replace("²", "^2").replace("³", "^3")
    normalized = normalized.replace("×", "*").replace("−", "-").replace("–", "-")
    normalized = re.sub(r"(?i)^(?:(?:resolve|solve|find|calculate|evaluate|work out|the equation|equation|integral)\s+)+", "", normalized)
    match = re.search(r"([0-9a-z\s.+*/^()\-]+?)\s*=\s*([0-9a-z\s.+*/^()\-]+?)(?:[?.!,;]|$)", normalized)
    if not match:
        return None
    try:
        lhs_text = re.sub(r"(?<=\d)(?=[a-z])", "*", match.group(1).strip())
        rhs_text = re.sub(r"(?<=\d)(?=[a-z])", "*", match.group(2).strip())
        lhs = validate_expression(lhs_text)
        rhs = validate_expression(rhs_text)
        x = sp.Symbol("x")
        lhs_expr = sp.sympify(lhs.replace("^", "**"), locals={"x": x})
        rhs_expr = sp.sympify(rhs.replace("^", "**"), locals={"x": x})
        if not lhs_expr.is_polynomial(x) or not rhs_expr.is_polynomial(x):
            return None
        degree = max(sp.degree(lhs_expr, x), sp.degree(rhs_expr, x))
        if degree <= 1:
            return None
        return {"expression": lhs, "second_expression": rhs, "second_label": f"y = {rhs}"}
    except Exception:
        return None


def choose_visualization(state: dict) -> dict:
    question = state.get("question", "")
    q = question.lower()
    intent = state.get("intent", "")

    from calculus import calculus_request
    calculus = calculus_request(question)

    arithmetic = re.sub(r"(?i)^\s*(?:what is|calculate|evaluate)\s+", "", question).rstrip("?!. ")
    if re.fullmatch(r"[\d\s.+\-*/^()]+", arithmetic):
        candidate = _none()
    elif state.get("verification_status") == "verified" and calculus:
        kind, expression = calculus
        if kind == "integral":
            candidate = {"type": "integral_plot", "title": "Integrand (shading over [0, 2] is illustrative)", "config": {"expression": expression, "bounds": (0, 2)}}
        else:
            candidate = {"type": "derivative_plot", "title": "Function and derivative", "config": {"expression": expression, "derivative_expression": _derivative(expression), "domain": (-10, 10)}}
    elif intent == "SOLVE":
        equation_plot = _polynomial_equation_plot(question)
        if equation_plot:
            candidate = {
                "type": "function_plot",
                "title": "Graph both sides of the equation",
                "config": {**equation_plot, "domain": (-15, 15)},
            }
        else:
            candidate = {
                "type": "step_by_step",
                "title": "Worked solution",
                "config": {
                    "steps": state.get("solution_steps", []),
                    "result": state.get("calculation_result", ""),
                    "verification": state.get("verification_status", "unverified"),
                },
            }
    elif any(term in q for term in ("circle", "radius", "دایره")):
        candidate = {"type": "geometry", "title": "Explore a circle", "config": {"shape": "circle", "radius": 2}}
    elif "pythagorean" in q or any(term in q for term in ("triangle", "angle", "geometry", "مثلث", "زاویه")):
        candidate = {"type": "geometry", "title": "A 3–4–5 right triangle" if "pythagorean" in q or "3–4–5" in q else "Explore the geometry", "config": {"shape": "triangle", "points": [(0, 0), (3, 0), (3, 4)] if "pythagorean" in q or "3–4–5" in q else []}}
    elif any(term in q for term in ("derivative", "differentiate", "مشتق")):
        expression = _expression_from_question(question)
        candidate = {
            "type": "derivative_plot",
            "title": "Function and derivative",
            "config": {"expression": expression, "derivative_expression": _derivative(expression), "domain": (-10, 10)},
        }
    elif any(term in q for term in ("integral", "area under", "انتگرال")):
        candidate = {"type": "integral_plot", "title": "Area under the curve", "config": {"expression": _expression_from_question(question), "bounds": (0, 2)}}
    elif re.search(r"\by\s*=|\bf\(x\)\s*=|\b(plot|graph|function)\b", q):
        expression = _expression_from_question(question)
        parameter_names = [name for name in ("a", "b", "c") if re.search(rf"\b{name}\b", expression)]
        if parameter_names:
            params = [
                {"name": name, "label": name, "minimum": -5, "maximum": 5, "step": 0.1, "default": 1 if name == "a" else 0}
                for name in parameter_names
            ]
            candidate = {"type": "parameter_explorer", "title": "Adjust the function", "config": {"expression": expression, "domain": (-10, 10), "parameters": params}}
        else:
            candidate = {"type": "function_plot", "title": "Function plot", "config": {"expression": expression, "domain": (-10, 10)}}
    elif intent == "CONCEPT" and state.get("explanation", state.get("final_answer", "")):
        formula = ""
        if "probability" in q:
            formula = "P(A) = \\frac{\\text{favorable outcomes}}{\\text{total outcomes}}"
        elif "multiply" in q or "multiplication" in q:
            formula = "a \\times b"
        candidate = {
            "type": "concept_card",
            "title": "The key idea",
            "config": {"explanation": state.get("explanation", state.get("final_answer", "")), "formula": formula, "example": ""},
        }
    else:
        candidate = _none()

    try:
        validated = VISUALIZATION_ADAPTER.validate_python(candidate)
        return validated.model_dump(mode="json")
    except Exception:
        return _none()