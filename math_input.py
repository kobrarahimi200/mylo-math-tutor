"""Normalize a small, explicit subset of TeX for the local symbolic solver.

Unrecognized notation stays intact for the general tutor; it is never guessed.
"""
from __future__ import annotations

import re


def normalize_math_question(question: str) -> str:
    text = question.strip()
    # Preserve named explanations and prose for the model and preset resolver.
    if re.match(r"(?i)^(explain|describe|derive|prove|show|what|why|how|give)\b", text):
        return question
    text = re.sub(
        r"(?i)^(solve|evaluate|calculate|simplify|differentiate|integrate)\s+"
        r"(?:(?:this|the)\s+)?(?:equation|expression|formula)\s*:?\s*",
        r"\1 ", text,
    )
    text = re.sub(r"\$\$?|\\[\[\]() ]", "", text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\\(?:,|;|!|quad\b|qquad\b)", " ", text)
    text = re.sub(r"\\frac\s*\{d\}\s*\{d\s*x\}", "differentiate ", text)
    # Work from inner braces out, so nested fractions and roots are supported.
    for _ in range(20):
        previous = text
        text = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
        text = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", text)
        if text == previous:
            break
    text = re.sub(r"\\(sin|cos|tan|exp|log)\b", r"\1", text)
    text = re.sub(r"\\ln\b", "log", text)
    text = re.sub(r"\\pi\b", "pi", text)
    text = re.sub(r"\\(?:cdot|times)\b", "*", text)
    text = re.sub(r"\\div\b", "/", text)
    text = re.sub(r"\\int\b", "integral ", text)
    # Commands, bounds, systems, and ambiguous braces belong to the model.
    if re.search(r"[\\{}_&]", text):
        return question
    if not re.fullmatch(r"[\w\s.+*/^()=?!,−–×÷-]+", text):
        return question
    return text.strip()
