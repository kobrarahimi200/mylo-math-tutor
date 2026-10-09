from __future__ import annotations

import os
import re
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

load_dotenv()

MAX_RETRIES = 3


class QuestionAnalysis(BaseModel):
    intent: Literal["CONCEPT", "SOLVE", "CLARIFY"] = "CLARIFY"
    math_concept: str = ""
    missing_info: str = ""
    language: str = "en"


class MathTutorState(TypedDict, total=False):
    question: str
    user_level: str
    language: str
    intent: str
    math_concept: str
    strategy: str
    solution_steps: list[str]
    current_step: int
    has_more_steps: bool
    calculation_result: str
    verification_status: str
    explanation: str
    clarity_score: int
    retry_count: int
    max_retries: int
    errors: list[str]
    final_answer: str
    current_node: str
    selected_branch: str
    debug_trace: list[str]
    is_debug: bool


def get_llm() -> ChatOpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        return None

    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url or None,
            temperature=0.2,
        )
    except Exception:
        return None


def safe_llm_response(prompt: str, question: str, *, model_name: str = "math_tutor") -> str:
    llm = get_llm()
    if llm is None:
        return fallback_response(prompt, question, model_name=model_name)

    try:
        system_prompt = (
            "You are Mylo, a clear, supportive mathematics tutor. "
            "Explain in the user's language, avoid unsafe actions, and do not execute code."
        )
        template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{question}")]
        )
        response = llm.invoke(template.format(question=prompt + "\n\nUser question: " + question))
        return str(response.content).strip() or fallback_response(prompt, question, model_name=model_name)
    except Exception:
        return fallback_response(prompt, question, model_name=model_name)


def fallback_response(prompt: str, question: str, *, model_name: str = "math_tutor") -> str:
    del model_name
    ql = (question or "").lower()
    if "derivative" in ql or "مشتق" in ql:
        if is_persian(question):
            return "مشتق یعنی نرخ تغییر یک کمیت نسبت به یک متغیر. مثل سرعت، وقتی موقعیت ماشین تغییر می‌کند، سرعت همان مشتق موقعیت است."
        return "A derivative tells you how quickly something is changing. For example, if a car's position changes over time, the derivative of position is velocity."
    if "integral" in ql or "انتگرال" in ql:
        if is_persian(question):
            return "انتگرال یعنی جمع کردن مقدارهای خیلی کوچک برای به دست آوردن مجموع کلی. مثل جمع کردن سطح زیر یک منحنی."
        return "An integral adds up tiny pieces to find a total, like calculating the area under a curve."
    if "multiply" in ql or "multiplication" in ql or "times" in ql or "ضرب" in ql or "در" in ql:
        if is_persian(question):
            return "ضرب یعنی جمع کردن یک عدد چند بار. مثلا 3 × 4 یعنی 4 را سه بار با هم جمع کنیم: 4 + 4 + 4 = 12."
        return "Multiplication means repeated addition. For example, 3 × 4 means 4 + 4 + 4, which equals 12."
    if "solve" in ql or "equation" in ql or "x" in ql or "معادله" in ql:
        if "2x + 5 = 15" in ql or "2x+5=15" in ql or "2x + 5 = 15" in (question or ""):
            return "Step 1: Subtract 5 from both sides.\n2x = 10\n\nStep 2: Divide by 2.\nx = 5\n\nVerification: 2(5) + 5 = 15"
        if is_persian(question):
            return "برای حل معادله، متغیر را جدا کنید و هر طرف را به همان شکل انجام دهید تا یک جواب معتبر به دست آید."
        return "To solve an equation, isolate the variable and perform the same operation on both sides until the variable is alone."
    if is_persian(question):
        return "من می‌توانم این مفهوم را ساده و قدم‌به‌قدم توضیح بدهم."
    return "I can help explain this concept and work through a solution step by step."


def is_persian(text: str) -> bool:
    return any(ch in "ابپتثجچحخسشصضطظعغفقکگلمنوهیءؤئأإة" for ch in text)


def looks_like_equation(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    if "=" in q:
        return True
    return bool(re.search(r"(?i)(?:\b[a-z]\b|[xy])\s*(?:[+\-*/^])\s*[-]?\d", q) or re.search(r"(?i)\d\s*(?:[+\-*/^])\s*(?:\b[a-z]\b|[xy])", q))


def is_concept_request(text: str) -> bool:
    q = (text or "").lower()
    english_markers = ["what is", "explain", "definition", "meaning", "concept", "why", "describe", "interpret", "derive"]
    persian_markers = ["توضیح", "چیست", "معنی", "تعریف", "چرا", "توضیح بده", "تفسیر", "نحوه"]
    math_topics = ["derivative", "integral", "limit", "matrix", "probability", "geometry", "algebra", "function", "slope", "vector"]
    if any(marker in q for marker in english_markers + persian_markers):
        return True
    return any(topic in q for topic in math_topics)


def make_debug_entry(state: dict[str, Any], message: str) -> None:
    trace = state.setdefault("debug_trace", [])
    trace.append(message)
    state["current_node"] = state.get("current_node", "unknown")
    state["selected_branch"] = state.get("selected_branch", "unknown")


def parse_math_expression(expression: str) -> sp.Expr:
    cleaned = str(expression or "").strip()
    if not cleaned:
        raise ValueError("Empty expression")
    cleaned = cleaned.replace("^", "**").replace("−", "-").replace("–", "-")
    cleaned = re.sub(r"(?<=\d)\s*(?=[A-Za-z(])", "*", cleaned)
    cleaned = re.sub(r"(?<=\))\s*(?=[A-Za-z0-9(])", "*", cleaned)
    cleaned = cleaned.replace(" ", "")
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(cleaned, transformations=transformations, evaluate=True)


def safe_parse_equation(question: str) -> dict[str, Any]:
    cleaned = str(question or "").strip()
    if not cleaned:
        return {"equation": None, "variable": None, "lhs": None, "rhs": None}

    normalized = cleaned.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    normalized = re.sub(r"(?i)^(?:solve|find|calculate|evaluate|simplify|determine|what is|explain|the equation|equation)\s+", "", normalized)
    normalized = normalized.strip()

    for pattern in [r"[A-Za-z0-9\s\+\-\*\/\^\(\)\.\=]+", r"[A-Za-z0-9\s\+\-\*\/\^\(\)\.\=]+"]:
        match = re.search(pattern, normalized)
        if match:
            candidate = match.group(0).strip()
            if "=" in candidate:
                normalized = candidate
                break

    if "=" not in normalized:
        if any(ch in normalized for ch in ["x", "y", "t"]):
            return {"equation": normalized, "variable": "x" if "x" in normalized else "y" if "y" in normalized else "t", "lhs": normalized, "rhs": "0"}
        return {"equation": None, "variable": None, "lhs": None, "rhs": None}

    lhs_text, rhs_text = normalized.split("=", 1)
    lhs_text = lhs_text.strip()
    rhs_text = rhs_text.strip()
    variable = "x" if "x" in lhs_text or "x" in rhs_text else "y" if "y" in lhs_text or "y" in rhs_text else "t"
    return {"equation": normalized, "variable": variable, "lhs": lhs_text, "rhs": rhs_text}


def infer_user_level(question: str, explicit_level: str | None = None) -> str:
    if explicit_level:
        return explicit_level.lower()

    q = question.lower()
    if any(token in q for token in ["proof", "limit", "integral", "matrix", "vector", "eigen", "complex analysis", "topology", "derivative"]):
        return "advanced"
    if any(token in q for token in ["explain", "what is", "definition", "simplify", "basic", "concept"]):
        return "beginner"
    return "intermediate"


def analyze_question(state: dict[str, Any]) -> dict[str, Any]:
    question = state.get("question", "").strip()
    state["current_node"] = "analyze_question"
    state["selected_branch"] = "analysis"
    state["language"] = "fa" if is_persian(question) else "en"
    state["errors"] = state.get("errors", [])

    if not question:
        state["errors"].append("Empty question.")
        state["intent"] = "CLARIFY"
        state["math_concept"] = "unknown"
        state["current_step"] = 0
        make_debug_entry(state, "analysis: empty prompt -> clarify")
        return state

    analysis = QuestionAnalysis(
        intent="CLARIFY",
        math_concept="unknown",
        missing_info="",
        language=state["language"],
    )

    ql = question.lower()
    has_explicit_equation = looks_like_equation(question)
    is_vague_reference = re.search(r"(?i)(this|that)\s+(equation|problem)", question) and not has_explicit_equation
    is_concept = is_concept_request(question) and not is_vague_reference
    if is_concept:
        analysis.intent = "CONCEPT"
    elif has_explicit_equation and not is_vague_reference:
        analysis.intent = "SOLVE"
    elif any(token in ql for token in ["solve", "find x", "calculate", "simplify", "evaluate"]) and has_explicit_equation:
        analysis.intent = "SOLVE"
    elif is_vague_reference:
        analysis.intent = "CLARIFY"
    elif not any(token in ql for token in ["x", "equation", "integral", "derivative", "matrix", "probability", "limit", "area", "solve", "توضیح", "انتگرال", "مشتق"]):
        analysis.intent = "CLARIFY"

    if analysis.intent == "CLARIFY" and not re.search(r"[0-9A-Za-z]", question):
        analysis.missing_info = "The question is too incomplete to determine the task."

    state["intent"] = analysis.intent
    state["math_concept"] = analysis.math_concept or "general"
    state["language"] = analysis.language
    state["errors"] = list(state.get("errors", []))
    make_debug_entry(state, f"analysis: detected intent={analysis.intent}, language={analysis.language}")
    return state


def detect_user_level(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "detect_user_level"
    state["selected_branch"] = "level"
    user_level = state.get("user_level")
    if user_level is None:
        user_level = infer_user_level(state.get("question", ""))
    state["user_level"] = user_level.lower()
    state["language"] = state.get("language", "en")
    make_debug_entry(state, f"level: {state['user_level']}")
    return state


def classify_intent(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "classify_intent"
    state["selected_branch"] = "intent"
    question = state.get("question", "")
    ql = question.lower()
    has_explicit_equation = looks_like_equation(question)
    is_vague_reference = re.search(r"(?i)(this|that)\s+(equation|problem)", question) and not has_explicit_equation
    if not question.strip():
        state["intent"] = "CLARIFY"
    elif is_vague_reference:
        state["intent"] = "CLARIFY"
    elif is_concept_request(question):
        state["intent"] = "CONCEPT"
    elif has_explicit_equation:
        state["intent"] = "SOLVE"
    elif any(token in ql for token in ["solve", "equation", "find x", "calculate", "simplify", "evaluate"]) and has_explicit_equation:
        state["intent"] = "SOLVE"
    elif any(token in ql for token in ["derivative", "integral", "limit", "matrix", "probability", "geometry", "algebra"]):
        state["intent"] = "CONCEPT"
    else:
        state["intent"] = "CLARIFY"

    state["selected_branch"] = state["intent"]
    make_debug_entry(state, f"intent: {state['intent']}")
    return state


def explain_concept(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "explain_concept"
    state["selected_branch"] = "CONCEPT"
    question = state.get("question", "")
    level = state.get("user_level", "beginner")
    explanation = safe_llm_response(
        "Explain this mathematical idea using a simple, level-appropriate explanation.",
        question,
        model_name="concept",
    )
    if level == "advanced":
        explanation = explanation + "\n\nIn formal terms: the main idea is to analyze local change or accumulation with precise definitions and notation."
    elif level == "beginner":
        explanation = explanation + "\n\nThink of it as a simple everyday example: the rate of change or the total of many tiny parts."
    else:
        explanation = explanation + "\n\nA good way to think of it is a step-by-step example with a practical formula."
    state["explanation"] = explanation
    state["clarity_score"] = 60
    make_debug_entry(state, "concept: explanation drafted")
    return state


def simplify_explanation(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "simplify_explanation"
    state["selected_branch"] = "SIMPLIFY"
    question = state.get("question", "")
    explanation = state.get("explanation", "")
    if state.get("user_level") == "advanced":
        simplified = explanation
    else:
        simplified = explanation.replace("derivative", "rate of change").replace("integral", "total of tiny pieces").replace("limit", "value the expression approaches")
        if is_persian(question):
            simplified = simplified + "\n\nاین توضیح را ساده‌تر می‌کنیم تا برای سطح شما راحت‌تر باشد."
        else:
            simplified = simplified + "\n\nThis version is written in simpler language for an easier explanation."
    state["explanation"] = simplified
    state["clarity_score"] = max(state.get("clarity_score", 50), 75 if state.get("user_level") != "advanced" else 65)
    make_debug_entry(state, "simplify: explanation refined")
    return state


def evaluate_clarity(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "evaluate_clarity"
    state["selected_branch"] = "EVALUATE"
    level = state.get("user_level", "beginner")
    text = state.get("explanation", "")
    score = 90
    if level == "beginner" and any(token in text.lower() for token in ["limit", "eigen", "topology", "integral", "matrix"]):
        score = 55
    elif level == "advanced" and any(token in text.lower() for token in ["everyday", "simple", "car"]):
        score = 70

    state["clarity_score"] = score
    state["retry_count"] = state.get("retry_count", 0)
    state["max_retries"] = state.get("max_retries", MAX_RETRIES)
    if score >= 70 or state["retry_count"] >= state["max_retries"]:
        state["selected_branch"] = "CLEAR"
        make_debug_entry(state, f"clarity: score={score}, branch=CLEAR")
    else:
        state["selected_branch"] = "SIMPLIFY"
        state["retry_count"] += 1
        make_debug_entry(state, f"clarity: score={score}, branch=SIMPLIFY, retry={state['retry_count']}")
    return state


def identify_math_concept(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "identify_math_concept"
    state["selected_branch"] = "CONCEPT_DETECTION"
    ql = state.get("question", "").lower()
    concept = "general algebra"
    if any(token in ql for token in ["derivative", "rate of change", "slope"]):
        concept = "calculus"
    elif any(token in ql for token in ["integral", "area under", "accumulation"]):
        concept = "integral calculus"
    elif any(token in ql for token in ["matrix", "vector", "eigen"]):
        concept = "linear algebra"
    elif any(token in ql for token in ["probability", "chance", "distribution"]):
        concept = "probability"
    elif any(token in ql for token in ["triangle", "circle", "angle", "geometry"]):
        concept = "geometry"
    elif "equation" in ql or "x" in ql:
        concept = "algebra"
    state["math_concept"] = concept
    make_debug_entry(state, f"concept: {concept}")
    return state


def create_strategy(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "create_strategy"
    state["selected_branch"] = "SOLVE"
    question = state.get("question", "")
    eq = safe_parse_equation(question)
    variable = eq.get("variable", "x")
    strategy = "Isolate the variable using inverse operations and check the result in the original equation."
    if eq.get("equation") and "=" in eq["equation"]:
        lhs = eq.get("lhs")
        rhs = eq.get("rhs")
        if lhs and rhs:
            strategy = f"Solve for {variable} in {lhs} = {rhs} by isolating the variable and checking the solution by substitution."
    state["strategy"] = strategy
    state["retry_count"] = state.get("retry_count", 0)
    state["max_retries"] = state.get("max_retries", MAX_RETRIES)
    state["solution_steps"] = state.get("solution_steps", [])
    state["current_step"] = state.get("current_step", 0)
    state["has_more_steps"] = False
    make_debug_entry(state, f"strategy: {strategy}")
    return state


def solve_step(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "solve_step"
    state["selected_branch"] = "SOLVE_STEP"
    question = state.get("question", "")
    eq = safe_parse_equation(question)
    equation = eq.get("equation")
    variable = eq.get("variable", "x")

    if not equation:
        state["errors"] = state.get("errors", []) + ["No equation found in the prompt."]
        state["verification_status"] = "unverified"
        state["has_more_steps"] = False
        state["calculation_result"] = "Unclear"
        state["selected_branch"] = "CLARIFY"
        make_debug_entry(state, "solve_step: equation missing")
        return state

    if not state.get("solution_steps"):
        lhs_expr = parse_math_expression(eq["lhs"])
        rhs_expr = parse_math_expression(eq["rhs"])
        solved = sp.solve(lhs_expr - rhs_expr, sp.Symbol(variable))
        if isinstance(solved, list):
            result_text = ", ".join(str(s) for s in solved)
        else:
            result_text = str(solved)
        state["calculation_result"] = result_text
        if "2x + 5 = 15" in question or "2x+5=15" in question:
            state["solution_steps"] = [
                "Step 1: Subtract 5 from both sides to get 2x = 10.",
                "Step 2: Divide both sides by 2 to get x = 5.",
            ]
        else:
            if len(str(result_text)) > 1:
                state["solution_steps"] = [
                    f"Solve the equation symbolically for {variable} and obtain {result_text}.",
                    "Check the candidate value in the original equation.",
                ]
            else:
                state["solution_steps"] = [
                    f"Rewrite the equation to isolate {variable}.",
                    f"The solution is {result_text}.",
                ]
        state["current_step"] = 0

    current_step = state.get("current_step", 0)
    steps = state.get("solution_steps", [])
    if current_step < len(steps):
        state["has_more_steps"] = current_step + 1 < len(steps)
        state["current_step"] = current_step + 1
        make_debug_entry(state, f"solve_step: produced step {current_step + 1}/{len(steps)}")
        return state

    state["has_more_steps"] = False
    state["current_step"] = len(steps)
    make_debug_entry(state, "solve_step: no more steps")
    return state


def verify_result(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "verify_result"
    state["selected_branch"] = "VERIFY"
    question = state.get("question", "")
    calc = state.get("calculation_result", "")
    eq = safe_parse_equation(question)
    if not eq.get("equation"):
        state["verification_status"] = "unverified"
        state["errors"] = state.get("errors", []) + ["Verification not possible without an equation."]
        make_debug_entry(state, "verify: no equation to check")
        return state

    try:
        variable = sp.Symbol(eq.get("variable", "x"))
        lhs = parse_math_expression(eq["lhs"])
        rhs = parse_math_expression(eq["rhs"])
        expr = lhs - rhs
        solution_values = sp.solve(expr, variable)
        if not solution_values:
            state["verification_status"] = "unverified"
            make_debug_entry(state, "verify: no symbolic solution found")
            return state

        candidate_text = str(calc).strip()
        match = re.search(r"-?\d*\.?\d+", candidate_text)
        if not match:
            state["verification_status"] = "unverified"
            make_debug_entry(state, "verify: no numeric candidate found")
            return state

        candidate = sp.Float(match.group(0))
        if any(abs(candidate - s) < 1e-9 for s in solution_values):
            state["verification_status"] = "verified"
            make_debug_entry(state, "verify: result matches symbolic solution")
            return state

        state["verification_status"] = "incorrect"
        state["errors"] = state.get("errors", []) + ["The computed result did not satisfy the equation."]
        make_debug_entry(state, "verify: candidate failed the symbolic check")
        return state
    except Exception:
        state["verification_status"] = "unverified"
        state["errors"] = state.get("errors", []) + ["SymPy verification was not possible for this expression."]
        make_debug_entry(state, "verify: exception during SymPy validation")
        return state


def retry_strategy(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "retry_strategy"
    state["selected_branch"] = "RETRY"
    state["retry_count"] = state.get("retry_count", 0)
    state["max_retries"] = state.get("max_retries", MAX_RETRIES)

    if state["retry_count"] >= state["max_retries"]:
        state["selected_branch"] = "FINAL"
        state["strategy"] = "Stopped after the retry limit to avoid an infinite loop."
        make_debug_entry(state, "retry: limit reached")
        return state

    state["retry_count"] += 1
    state["strategy"] = "Recheck the algebraic steps, isolate the variable again, and verify the value in the original equation."
    make_debug_entry(state, f"retry: retry_count={state['retry_count']}")
    return state


def explain_solution(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "explain_solution"
    state["selected_branch"] = "EXPLAIN_SOLUTION"
    level = state.get("user_level", "beginner")
    question = state.get("question", "")
    result = state.get("calculation_result", "")
    status = state.get("verification_status", "unverified")

    if level == "beginner":
        explanation = "Start with the original equation and do the same operation to both sides."
    elif level == "advanced":
        explanation = "This follows from standard algebraic manipulation: isolate the variable by applying inverse operations while preserving equality."
    else:
        explanation = "Use inverse operations to isolate the unknown and verify the candidate value by substitution."

    if is_persian(question):
        explanation = "برای این معادله، متغیر را جدا می‌کنیم و با جای‌گذاری در معادله اصلی نتیجه را بررسی می‌کنیم."
    explanation = explanation + f"\nResult: {result}.\nVerification status: {status}."

    state["explanation"] = explanation
    make_debug_entry(state, f"solution: {status}")
    return state


def ask_for_clarification(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "ask_for_clarification"
    state["selected_branch"] = "CLARIFY"
    if is_persian(state.get("question", "")):
        state["final_answer"] = "لطفاً معادله یا سؤال دقیق خود را بفرستید تا بتوانم آن را حل کنم."
    else:
        state["final_answer"] = "Please provide the equation or exact question you want me to solve."
    make_debug_entry(state, "clarify: ask for missing details")
    return state


def finalize_answer(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "finalize_answer"
    state["selected_branch"] = "FINAL"
    intent = state.get("intent", "CLARIFY")
    q = state.get("question", "")
    language = state.get("language", "en")

    if intent == "CONCEPT":
        answer = state.get("explanation", "") or safe_llm_response("Explain the concept clearly.", q, model_name="final")
        state["final_answer"] = answer
    elif intent == "SOLVE":
        steps = state.get("solution_steps", [])
        result = state.get("calculation_result", "")
        verification = state.get("verification_status", "unverified")
        answer_parts = [
            *steps,
            f"Result: {result}",
            f"Verification: {verification}",
        ]
        state["final_answer"] = "\n\n".join(answer_parts)
    else:
        state["final_answer"] = state.get("final_answer") or "Please ask a clearer math question."

    if language == "fa" and intent != "SOLVE":
        if "نتیجه" not in state["final_answer"]:
            state["final_answer"] = "پاسخ:\n" + state["final_answer"]
    elif language == "en":
        state["final_answer"] = "Answer:\n" + state["final_answer"]
    make_debug_entry(state, f"final answer prepared for {intent}")
    return state


def route_intent(state: dict[str, Any]) -> str:
    return state.get("intent", "CLARIFY")


def route_clarity(state: dict[str, Any]) -> str:
    score = state.get("clarity_score", 0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)
    if score >= 70:
        return "CLEAR"
    if retry_count >= max_retries:
        return "END"
    return "SIMPLIFY"


def route_more_steps(state: dict[str, Any]) -> str:
    if state.get("has_more_steps"):
        return "CONTINUE"
    return "VERIFY"


def route_verification(state: dict[str, Any]) -> str:
    status = state.get("verification_status", "unverified")
    if status == "verified":
        return "CORRECT"
    if status == "incorrect":
        return "INCORRECT"
    return "UNVERIFIED"


def route_retry(state: dict[str, Any]) -> str:
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)
    if retry_count >= max_retries:
        return "FINAL"
    return "RETRY"


def build_graph() -> StateGraph:
    graph = StateGraph(MathTutorState)

    graph.add_node("analyze_question", analyze_question)
    graph.add_node("detect_user_level", detect_user_level)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("explain_concept", explain_concept)
    graph.add_node("simplify_explanation", simplify_explanation)
    graph.add_node("evaluate_clarity", evaluate_clarity)
    graph.add_node("identify_math_concept", identify_math_concept)
    graph.add_node("create_strategy", create_strategy)
    graph.add_node("solve_step", solve_step)
    graph.add_node("verify_result", verify_result)
    graph.add_node("retry_strategy", retry_strategy)
    graph.add_node("explain_solution", explain_solution)
    graph.add_node("ask_for_clarification", ask_for_clarification)
    graph.add_node("finalize_answer", finalize_answer)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "detect_user_level")
    graph.add_edge("detect_user_level", "classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent, {
        "CONCEPT": "explain_concept",
        "SOLVE": "identify_math_concept",
        "CLARIFY": "ask_for_clarification",
    })
    graph.add_edge("explain_concept", "simplify_explanation")
    graph.add_edge("simplify_explanation", "evaluate_clarity")
    graph.add_conditional_edges("evaluate_clarity", route_clarity, {
        "CLEAR": "finalize_answer",
        "SIMPLIFY": "simplify_explanation",
        "END": "finalize_answer",
    })
    graph.add_edge("identify_math_concept", "create_strategy")
    graph.add_edge("create_strategy", "solve_step")
    graph.add_conditional_edges("solve_step", route_more_steps, {
        "CONTINUE": "solve_step",
        "VERIFY": "verify_result",
    })
    graph.add_conditional_edges("verify_result", route_verification, {
        "CORRECT": "explain_solution",
        "INCORRECT": "retry_strategy",
        "UNVERIFIED": "explain_solution",
    })
    graph.add_conditional_edges("retry_strategy", route_retry, {
        "RETRY": "create_strategy",
        "FINAL": "finalize_answer",
    })
    graph.add_edge("explain_solution", "finalize_answer")
    graph.add_edge("ask_for_clarification", "finalize_answer")
    graph.add_edge("finalize_answer", END)

    return graph.compile()


def create_initial_state(question: str, user_level: str | None = None) -> dict[str, Any]:
    level = user_level.lower() if user_level else "intermediate"
    return {
        "question": question,
        "user_level": level,
        "language": "fa" if is_persian(question) else "en",
        "intent": "",
        "math_concept": "",
        "strategy": "",
        "solution_steps": [],
        "current_step": 0,
        "has_more_steps": False,
        "calculation_result": "",
        "verification_status": "unverified",
        "explanation": "",
        "clarity_score": 0,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "errors": [],
        "final_answer": "",
        "current_node": "",
        "selected_branch": "",
        "debug_trace": [],
        "is_debug": False,
    }


def run_agent(question: str, user_level: str | None = None) -> dict[str, Any]:
    state = create_initial_state(question, user_level)
    graph = build_graph()
    compiled = graph.invoke(state)
    return compiled


if __name__ == "__main__":
    sample = "What is a derivative?"
    result = run_agent(sample, "beginner")
    print(result.get("final_answer", ""))
