"""Restricted calculus input parsing; verification never depends on model output."""
from __future__ import annotations

import re
import sympy as sp
from visualizations import validate_expression


def calculus_request(question: str) -> tuple[str, str] | None:
    normalized = question.replace("²", "^2").replace("³", "^3").replace("−", "-")
    normalized = normalized.replace("\\int", "∫").replace("\\,", " ").replace("$$", "").strip()
    # An explicit equality without an integral sign is an equation, even if
    # the user casually called it an integral.
    if "=" in normalized and "∫" not in normalized:
        return None
    integral = re.search(r"(?:∫|\bintegral(?:\s+of)?\b|\bintegrate\b)\s*(.+?)\s*d\s*x\b", normalized, re.I)
    derivative = re.search(r"(?:\bderivative\s+of\b|\bdifferentiate\b)\s*(.+?)(?:[?!;]|$)", normalized, re.I)
    match = integral or derivative
    if not match:
        return None
    expression = match.group(1).strip().rstrip(". ")
    expression = re.sub(r"(?<=\d)(?=[x(])", "*", expression)
    try:
        validate_expression(expression)
    except ValueError:
        return None
    return ("integral" if integral else "derivative", expression)


def compute_calculus(kind: str, expression: str) -> tuple[sp.Expr, sp.Expr]:
    x = sp.Symbol("x")
    parsed = sp.sympify(expression.replace("^", "**"), locals={"x": x})
    return parsed, sp.integrate(parsed, x) if kind == "integral" else sp.diff(parsed, x)


def verify_calculus(kind: str, expression: str, candidate: str) -> bool:
    x = sp.Symbol("x")
    original, _ = compute_calculus(kind, expression)
    value = sp.sympify(candidate, locals={"x": x})
    if kind == "integral":
        return sp.simplify(sp.diff(value, x) - original) == 0
    h = sp.Symbol("h", real=True)
    definition = sp.limit((original.subs(x, x + h) - original)/h, h, 0)
    return sp.simplify(value - definition) == 0
