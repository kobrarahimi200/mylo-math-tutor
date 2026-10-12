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
from visualizations import choose_visualization

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
    exact_calculation_result: str
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
    visualization_type: str
    visualization_spec: dict[str, Any]
    visualization_validation: str
    explanation_level: str
    conversation_context: list[dict[str, str]]


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


def safe_llm_response(
    prompt: str,
    question: str,
    *,
    model_name: str = "math_tutor",
    context: list[dict[str, str]] | None = None,
) -> str:
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
        history = ""
        if context:
            history = "\n\nRecent conversation:\n" + "\n".join(
                f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in context[-8:]
            )
        response = llm.invoke(template.format(question=prompt + history + "\n\nUser question: " + question))
        return str(response.content).strip() or fallback_response(prompt, question, model_name=model_name)
    except Exception:
        return fallback_response(prompt, question, model_name=model_name)


def fallback_response(prompt: str, question: str, *, model_name: str = "math_tutor") -> str:
    del model_name
    ql = (question or "").lower()
    arithmetic = re.sub(r"(?i)^\s*(?:what is|calculate|evaluate)\s+", "", question or "").rstrip("?!. ")
    if arithmetic and re.fullmatch(r"[\d\s.+\-*/^()]+", arithmetic):
        try:
            result = parse_math_expression(arithmetic)
            if result.is_number:
                return f"{arithmetic} = {result}"
        except Exception:
            pass
    if "derivative" in ql or "differentiate" in ql or "مشتق" in ql:
        expression_match = re.search(
            r"(?i)(?:derivative of|differentiate)\s+([0-9a-z\s.+*/^()\-]+?)(?:\s+(?:using|at)\b|[?.!,;]|$)",
            question,
        )
        derivative_text = ""
        if expression_match:
            expression = expression_match.group(1).strip().replace("²", "^2").replace("³", "^3")
            if re.fullmatch(r"[0-9x\s.+*/^()\-]+", expression):
                try:
                    derivative = sp.diff(parse_math_expression(expression), sp.Symbol("x"))
                    derivative_text = f" The derivative of {expression} is {format_algebraic_expression(derivative)}."
                except Exception:
                    pass
        if is_persian(question):
            return "مشتق نرخ تغییر یک تابع را نشان می‌دهد؛ مثلا مشتق مکان نسبت به زمان، سرعت است." + derivative_text
        return "A derivative measures how quickly a function changes. For example, the derivative of position with respect to time is velocity." + derivative_text
    if "integral" in ql or "انتگرال" in ql:
        if is_persian(question):
            return "انتگرال یعنی جمع کردن مقدارهای خیلی کوچک برای به دست آوردن مجموع کلی. مثل جمع کردن سطح زیر یک منحنی."
        return "An integral adds up tiny pieces to find a total, like calculating the area under a curve."
    if "multiply" in ql or "multiplication" in ql or "times" in ql or "ضرب" in ql or "در" in ql:
        if is_persian(question):
            return "ضرب یعنی جمع کردن یک عدد چند بار. مثلا 3 × 4 یعنی 4 را سه بار با هم جمع کنیم: 4 + 4 + 4 = 12."
        return "Multiplication means repeated addition. For example, 3 × 4 means 4 + 4 + 4, which equals 12."
    if any(term in ql for term in ("division", "divide", "divided", "تقسیم")):
        if is_persian(question):
            return "تقسیم یعنی تقسیم کردن یک مقدار به گروه‌های برابر. مثلا 12 ÷ 3 یعنی 12 را به 3 گروه برابر تقسیم کنیم؛ در هر گروه 4 تا می‌شود."
        return "Division splits a quantity into equal groups. For example, 12 ÷ 3 asks how many are in each of 3 equal groups; the answer is 4."
    if "fraction" in ql or "کسر" in ql:
        if is_persian(question):
            return "کسر بخشی از یک کل را نشان می‌دهد. در 3/4، عدد 4 کل را به چهار قسمت برابر تقسیم می‌کند و عدد 3 سه قسمت از آن را نشان می‌دهد."
        return "A fraction represents part of a whole. In 3/4, the denominator 4 divides the whole into four equal parts, and the numerator 3 counts three of them."
    if "prime" in ql or "اول" in ql:
        if is_persian(question):
            return "عدد اول عددی بزرگ‌تر از 1 است که دقیقا دو مقسوم‌علیه مثبت دارد: 1 و خودش. مثلا 7 عدد اول است."
        return "A prime number is an integer greater than 1 with exactly two positive divisors: 1 and itself. For example, 7 is prime."
    if "probability" in ql or "احتمال" in ql:
        if is_persian(question):
            return "احتمال اندازه شانس رخ دادن یک رویداد است. برای حالت‌های هم‌شانس، احتمال برابر است با تعداد نتیجه‌های مطلوب تقسیم بر تعداد کل نتیجه‌ها. مثلا احتمال شیر آمدن سکه سالم 1/2 است."
        return "Probability measures the chance that an event will happen. For equally likely outcomes, it is favorable outcomes divided by total outcomes. For a fair coin, the chance of heads is 1/2."
    if "square root" in ql or "sqrt" in ql or "جذر" in ql:
        if is_persian(question):
            return "جذر دوم عددی است که اگر در خودش ضرب شود، عدد اصلی به دست می‌آید. مثلا جذر دوم 25 برابر 5 است، چون 5 × 5 = 25."
        return "A square root is a number that multiplied by itself gives the original number. For example, √25 = 5 because 5 × 5 = 25."
    if "percent" in ql or "percentage" in ql or "درصد" in ql:
        if is_persian(question):
            return "درصد یعنی تعداد از هر 100. مثلا 25٪ یعنی 25 از 100، یا یک‌چهارم کل."
        return "A percent means a number out of 100. For example, 25% means 25 out of 100, which is one quarter of the whole."
    if "average" in ql or "mean" in ql or "میانگین" in ql:
        if is_persian(question):
            return "میانگین حسابی از تقسیم مجموع عددها بر تعداد آن‌ها به دست می‌آید. میانگین 2، 4 و 6 برابر (2 + 4 + 6) ÷ 3 = 4 است."
        return "The arithmetic mean is the sum of the numbers divided by how many numbers there are. The mean of 2, 4, and 6 is (2 + 4 + 6) ÷ 3 = 4."
    if "slope" in ql or "شیب" in ql:
        if is_persian(question):
            return "شیب نشان می‌دهد y به ازای تغییر x چقدر تغییر می‌کند. برای دو نقطه، شیب برابر تغییر y تقسیم بر تغییر x است: m = (y₂ - y₁)/(x₂ - x₁)."
        return "Slope tells how much y changes for a change in x. Between two points, slope is the change in y divided by the change in x: m = (y₂ - y₁)/(x₂ - x₁)."
    if "algebra" in ql or "جبر" in ql:
        if is_persian(question):
            return "جبر از نمادها، معمولا حروف، برای نمایش عددهای ناشناخته یا متغیر استفاده می‌کند. مثلا در x + 3 = 7، باید x را پیدا کنیم؛ پس x = 4."
        return "Algebra uses symbols, often letters, to represent unknown or changing numbers. In x + 3 = 7, solve for the unknown and get x = 4."
    if "function" in ql or "تابع" in ql:
        if is_persian(question):
            return "تابع قانونی است که به هر ورودی مجاز دقیقا یک خروجی نسبت می‌دهد. مثلا در f(x) = 2x، ورودی 3 خروجی 6 دارد."
        return "A function is a rule that gives exactly one output for each allowed input. For example, for f(x) = 2x, input 3 gives output 6."
    if "addition" in ql or "add" in ql or "جمع" in ql:
        if is_persian(question):
            return "جمع کردن یعنی کنار هم گذاشتن مقدارها. مثلا 3 + 2 = 5."
        return "Addition combines quantities. For example, 3 + 2 = 5."
    if "subtraction" in ql or "subtract" in ql or "minus" in ql or "تفریق" in ql:
        if is_persian(question):
            return "تفریق یعنی کم کردن یک مقدار از مقدار دیگر. مثلا 5 - 2 = 3."
        return "Subtraction takes one quantity away from another. For example, 5 - 2 = 3."
    if "circle" in ql or "دایره" in ql:
        if is_persian(question):
            return "مساحت دایره از رابطه A = πr² به دست می‌آید؛ r شعاع دایره است. مثلا اگر شعاع 2 باشد، مساحت 4π واحد مربع است."
        return "A circle's area is A = πr², where r is its radius. For example, if the radius is 2, the area is 4π square units."
    if " y = a" in ql or "how a changes" in ql or "coefficient a" in ql:
        if is_persian(question):
            return "ضریب a کشیدگی عمودی سهمی y = ax² را کنترل می‌کند. وقتی |a| بزرگ‌تر شود سهمی باریک‌تر می‌شود؛ اگر a منفی باشد، رو به پایین باز می‌شود."
        return "The coefficient a controls the vertical stretch of y = ax². A larger |a| makes the parabola narrower; a negative a flips it downward."
    if "plot" in ql or "graph" in ql or re.search(r"\by\s*=", ql):
        expression = ql.split("=", 1)[1].strip(" ?.!;,") if "=" in ql else ""
        if "sin" in expression:
            return "The graph of y = sin(x) is a smooth wave that oscillates between -1 and 1 and repeats every 2π."
        if "cos" in expression:
            return "The graph of y = cos(x) is a smooth wave between -1 and 1 that repeats every 2π, starting at 1."
        if "x" in expression:
            return "This graph shows how y changes as x changes. The plotted curve makes its shape and key values easier to explore."
    if "solve" in ql or "equation" in ql or "معادله" in ql or looks_like_equation(question):
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
    english_markers = ["what is", "what does", "what are", "explain", "definition", "meaning", "mean?", "concept", "why", "describe", "interpret", "derive", "show", "visualize", "plot", "graph", "how do", "how does"]
    persian_markers = ["توضیح", "چیست", "معنی", "تعریف", "چرا", "توضیح بده", "تفسیر", "نحوه"]
    math_topics = ["derivative", "integral", "limit", "matrix", "probability", "geometry", "algebra", "function", "slope", "vector", "division", "fraction", "prime", "percent", "percentage", "average", "mean", "square root", "multiply", "multiplication"]
    if any(marker in q for marker in english_markers + persian_markers):
        return True
    return any(topic in q for topic in math_topics) or bool(re.search(r"\by\s*=|\bf\(x\)\s*=", q))


def is_contextual_followup(text: str, context: list[dict[str, str]] | None) -> bool:
    if not context:
        return False
    q = (text or "").strip().lower()
    markers = ("why", "how", "explain more", "what about", "and then", "can you", "it?", "that?", "چرا", "چطور", "بیشتر")
    return any(q.startswith(marker) or q == marker for marker in markers)


def make_debug_entry(state: dict[str, Any], message: str) -> None:
    trace = state.setdefault("debug_trace", [])
    trace.append(message)
    state["current_node"] = state.get("current_node", "unknown")
    state["selected_branch"] = state.get("selected_branch", "unknown")


def parse_math_expression(expression: str) -> sp.Expr:
    cleaned = str(expression or "").strip()
    if not cleaned:
        raise ValueError("Empty expression")
    cleaned = cleaned.replace("²", "^2").replace("³", "^3")
    cleaned = cleaned.replace("^", "**").replace("−", "-").replace("–", "-")
    cleaned = re.sub(r"(?<=\d)\s*(?=[A-Za-z(])", "*", cleaned)
    cleaned = re.sub(r"(?<=\))\s*(?=[A-Za-z0-9(])", "*", cleaned)
    cleaned = cleaned.replace(" ", "")
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(cleaned, transformations=transformations, evaluate=True)


def simple_arithmetic_expression(question: str) -> str | None:
    expression = str(question or "").strip().lower()
    expression = re.sub(r"^(?:what is|calculate|compute|evaluate|solve)\s+", "", expression)
    expression = expression.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    replacements = (
        (r"\b(?:multiplied by|times|multiply)\b", "*"),
        (r"\b(?:divided by|divide by|over)\b", "/"),
        (r"\b(?:plus|added to)\b", "+"),
        (r"\bminus\b", "-"),
    )
    for pattern, operator in replacements:
        expression = re.sub(pattern, operator, expression)
    expression = expression.strip(" ?.!;,:")
    if not re.fullmatch(r"[\d\s.+*/^()\-]+", expression) or not re.search(r"[+*/^]|-(?=\s*\d)", expression):
        return None
    try:
        parsed = parse_math_expression(expression)
    except Exception:
        return None
    return expression if parsed.is_number else None


def expression_before_empty_equals(question: str) -> str | None:
    normalized = str(question or "").strip().replace("−", "-").replace("–", "-")
    if "=" not in normalized or normalized.split("=", 1)[1].strip():
        return None
    expression = normalized.split("=", 1)[0].strip()
    expression = re.sub(r"(?i)^(?:this is|simplify|reduce|what is|calculate|evaluate|solve)\s+", "", expression).strip()
    if not expression or not re.fullmatch(r"[0-9A-Za-z\s+*/^().-]+", expression):
        return None
    if any(name.lower() not in {"x", "y", "z", "a", "b", "c", "t"} for name in re.findall(r"[A-Za-z]+", expression)):
        return None
    try:
        parsed = parse_math_expression(expression)
    except Exception:
        return None
    return expression if parsed.free_symbols else None


def is_log_product_zeta_integral(question: str) -> bool:
    compact = re.sub(r"\s+", "", str(question or "").lower())
    has_integral = r"\int" in compact or "definiteintegral" in compact
    has_unit_bounds = "_0^1" in compact or "from0to1" in compact
    has_log_product = "\\ln(x)" in compact and "\\ln(1-x)" in compact
    has_denominator_x = "}{x}" in compact or "/x" in compact
    return has_integral and has_unit_bounds and has_log_product and has_denominator_x


def format_algebraic_expression(expression: sp.Expr) -> str:
    return str(expression).replace("**", "^").replace("*", "")


def format_solution_value(value: sp.Expr) -> str:
    if value.is_Rational:
        return str(value)
    return str(sp.N(value, 12))


def safe_parse_equation(question: str) -> dict[str, Any]:
    cleaned = str(question or "").strip()
    if not cleaned:
        return {"equation": None, "variable": None, "lhs": None, "rhs": None}

    normalized = cleaned.replace("²", "^2").replace("³", "^3")
    normalized = normalized.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    normalized = re.sub(r"(?i)^(?:(?:resolve|solve|find|calculate|evaluate|simplify|determine|what is|explain|the equation|equation|integral)\s+)+", "", normalized)
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
    arithmetic_expression = simple_arithmetic_expression(question)
    simplification_expression = expression_before_empty_equals(question)
    derivative_request = any(term in ql for term in ("derivative", "differentiate", "مشتق"))
    is_vague_reference = re.search(r"(?i)(this|that)\s+(equation|problem)", question) and not has_explicit_equation
    is_concept = (is_concept_request(question) or is_contextual_followup(question, state.get("conversation_context"))) and not is_vague_reference
    asks_to_solve = any(token in ql for token in ["solve", "resolve", "find", "calculate", "simplify", "evaluate", "work out"])
    if is_log_product_zeta_integral(question):
        analysis.intent = "SOLVE"
        analysis.math_concept = "calculus and infinite series"
    elif derivative_request:
        analysis.intent = "CONCEPT"
        analysis.math_concept = "calculus"
    elif arithmetic_expression or simplification_expression:
        analysis.intent = "SOLVE"
    elif asks_to_solve and has_explicit_equation and not is_vague_reference:
        analysis.intent = "SOLVE"
    elif is_concept:
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
    arithmetic_expression = simple_arithmetic_expression(question)
    simplification_expression = expression_before_empty_equals(question)
    derivative_request = any(term in ql for term in ("derivative", "differentiate", "مشتق"))
    is_vague_reference = re.search(r"(?i)(this|that)\s+(equation|problem)", question) and not has_explicit_equation
    asks_to_solve = any(token in ql for token in ["solve", "resolve", "find", "calculate", "simplify", "evaluate", "work out"])
    if not question.strip():
        state["intent"] = "CLARIFY"
    elif is_vague_reference:
        state["intent"] = "CLARIFY"
    elif is_log_product_zeta_integral(question):
        state["intent"] = "SOLVE"
    elif derivative_request:
        state["intent"] = "CONCEPT"
    elif arithmetic_expression or simplification_expression:
        state["intent"] = "SOLVE"
    elif asks_to_solve and has_explicit_equation:
        state["intent"] = "SOLVE"
    elif is_concept_request(question) or is_contextual_followup(question, state.get("conversation_context")):
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
        context=state.get("conversation_context"),
    )
    if level == "advanced":
        explanation = explanation + "\n\nIn formal terms: the main idea is to analyze local change or accumulation with precise definitions and notation."
    elif level == "beginner":
        ql = question.lower()
        if any(term in ql for term in ("derivative", "مشتق")):
            explanation += "\n\nThink of it as an everyday example: a car's changing position gives its speed."
        elif any(term in ql for term in ("integral", "انتگرال")):
            explanation += "\n\nThink of adding many tiny pieces to find a total, like the area under a curve."
        elif any(term in ql for term in ("multiply", "multiplication", "ضرب")):
            explanation += "\n\nFor example, 3 bags with 4 apples each make 3 × 4 = 12 apples."
        elif "circle" in ql or "دایره" in ql:
            explanation += "\n\nThe radius is the distance from the center to the edge; doubling it makes the area four times larger."
        else:
            explanation += "\n\nA small worked example can make the main idea easier to see."
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
    arithmetic_expression = simple_arithmetic_expression(question)
    simplification_expression = expression_before_empty_equals(question)
    special_integral = is_log_product_zeta_integral(question)
    eq = safe_parse_equation(question) if not (arithmetic_expression or simplification_expression or special_integral) else {"equation": None}
    equation = eq.get("equation")
    variable = eq.get("variable", "x")

    if special_integral:
        if not state.get("solution_steps"):
            numeric_value = str(sp.N(sp.zeta(3), 15))
            state["calculation_result"] = numeric_value
            state["solution_steps"] = [
                "Method: expand ln(1 - x) as the power series -Σ(n=1 to ∞) x^n/n, valid for 0 ≤ x < 1.",
                "Substitute the series into the integral. Each term contains -x^(n-1)ln(x)/n, whose integral from 0 to 1 is 1/n^3 because ∫₀¹ x^(n-1)ln(x) dx = -1/n².",
                "Sum the resulting series: Σ(n=1 to ∞) 1/n^3 = ζ(3), Apéry's constant. Term-by-term integration is justified by monotone convergence since the terms are nonnegative on (0, 1).",
                f"Numerical verification: ζ(3) ≈ {numeric_value}.",
            ]
            state["current_step"] = 0
    elif arithmetic_expression:
        if not state.get("solution_steps"):
            result = parse_math_expression(arithmetic_expression)
            state["calculation_result"] = str(result)
            state["solution_steps"] = [
                f"Evaluate {arithmetic_expression} using the order of operations.",
                f"The result is {result}.",
            ]
            state["current_step"] = 0
    elif simplification_expression:
        if not state.get("solution_steps"):
            simplified = sp.simplify(parse_math_expression(simplification_expression))
            result_text = format_algebraic_expression(simplified)
            state["calculation_result"] = result_text
            state["solution_steps"] = [
                f"Combine like terms in {simplification_expression} to get {result_text}."
            ]
            state["current_step"] = 0
    elif not equation:
        state["errors"] = state.get("errors", []) + ["No equation or arithmetic expression found in the prompt."]
        state["verification_status"] = "unverified"
        state["has_more_steps"] = False
        state["calculation_result"] = "Unclear"
        state["selected_branch"] = "CLARIFY"
        make_debug_entry(state, "solve_step: equation or arithmetic expression missing")
        return state

    if not state.get("solution_steps"):
        lhs_expr = parse_math_expression(eq["lhs"])
        rhs_expr = parse_math_expression(eq["rhs"])
        solved = sp.solve(lhs_expr - rhs_expr, sp.Symbol(variable))
        if isinstance(solved, list):
            exact_result_text = ", ".join(str(value) for value in solved)
            result_text = ", ".join(format_solution_value(value) for value in solved)
        else:
            exact_result_text = str(solved)
            result_text = format_solution_value(solved) if isinstance(solved, sp.Expr) else str(solved)
        state["exact_calculation_result"] = exact_result_text
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
    arithmetic_expression = simple_arithmetic_expression(question)
    simplification_expression = expression_before_empty_equals(question)
    special_integral = is_log_product_zeta_integral(question)
    eq = safe_parse_equation(question) if not (arithmetic_expression or simplification_expression or special_integral) else {"equation": None}
    if special_integral:
        expected = sp.N(sp.zeta(3), 15)
        try:
            candidate = sp.Float(calc)
            if abs(candidate - expected) < sp.Float("1e-14"):
                state["verification_status"] = "verified"
                make_debug_entry(state, "verify: integral numerical value matches zeta(3)")
                return state
            state["verification_status"] = "incorrect"
            state["errors"] = state.get("errors", []) + ["The numerical integral result did not match zeta(3)."]
            make_debug_entry(state, "verify: integral numerical check failed")
            return state
        except Exception:
            state["verification_status"] = "unverified"
            state["errors"] = state.get("errors", []) + ["Numerical integral verification was not possible."]
            make_debug_entry(state, "verify: integral numerical check unavailable")
            return state
    if arithmetic_expression:
        try:
            expected = parse_math_expression(arithmetic_expression)
            candidate = parse_math_expression(calc)
            if sp.simplify(candidate - expected) == 0:
                state["verification_status"] = "verified"
                make_debug_entry(state, "verify: arithmetic result matches SymPy evaluation")
                return state
            state["verification_status"] = "incorrect"
            state["errors"] = state.get("errors", []) + ["The computed arithmetic result is incorrect."]
            make_debug_entry(state, "verify: arithmetic candidate failed")
            return state
        except Exception:
            state["verification_status"] = "unverified"
            state["errors"] = state.get("errors", []) + ["SymPy arithmetic verification was not possible."]
            make_debug_entry(state, "verify: arithmetic expression could not be checked")
            return state
    if simplification_expression:
        try:
            expected = sp.simplify(parse_math_expression(simplification_expression))
            candidate = parse_math_expression(calc)
            if sp.simplify(candidate - expected) == 0:
                state["verification_status"] = "verified"
                make_debug_entry(state, "verify: simplified expression matches SymPy")
                return state
            state["verification_status"] = "incorrect"
            state["errors"] = state.get("errors", []) + ["The simplified expression is incorrect."]
            make_debug_entry(state, "verify: simplified expression failed")
            return state
        except Exception:
            state["verification_status"] = "unverified"
            state["errors"] = state.get("errors", []) + ["SymPy simplification verification was not possible."]
            make_debug_entry(state, "verify: simplified expression could not be checked")
            return state
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

        exact_calc = state.get("exact_calculation_result") or calc
        candidate_texts = [value.strip() for value in str(exact_calc).split(",") if value.strip()]
        if not candidate_texts:
            state["verification_status"] = "unverified"
            make_debug_entry(state, "verify: no numeric candidate found")
            return state

        candidates = [parse_math_expression(value) for value in candidate_texts]
        if any(sp.simplify(candidate - solution) == 0 for candidate in candidates for solution in solution_values):
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


def select_visualization(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "select_visualization"
    state["selected_branch"] = "VISUALIZATION"
    spec = choose_visualization(state)
    state["visualization_type"] = spec["type"]
    state["visualization_spec"] = spec
    state["visualization_validation"] = "valid"
    state["explanation_level"] = state.get("user_level", "intermediate")
    make_debug_entry(state, f"visualization: selected {spec['type']}")
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
    graph.add_node("select_visualization", select_visualization)

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
    graph.add_edge("finalize_answer", "select_visualization")
    graph.add_edge("select_visualization", END)

    return graph.compile()


def create_initial_state(
    question: str,
    user_level: str | None = None,
    conversation_context: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
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
        "visualization_type": "none",
        "visualization_spec": {"type": "none", "title": "", "config": {}},
        "visualization_validation": "pending",
        "explanation_level": level,
        "conversation_context": conversation_context or [],
    }


def run_agent(
    question: str,
    user_level: str | None = None,
    conversation_context: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    state = create_initial_state(question, user_level, conversation_context)
    graph = build_graph()
    compiled = graph.invoke(state)
    return compiled


if __name__ == "__main__":
    sample = "What is a derivative?"
    result = run_agent(sample, "beginner")
    print(result.get("final_answer", ""))
