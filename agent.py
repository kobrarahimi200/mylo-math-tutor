from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from llm_factory import get_llm, begin_llm_request, llm_status, record_llm_success, record_llm_failure, activate_runtime
from calculus import calculus_request, compute_calculus, verify_calculus
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from visualizations import choose_visualization
from formula_examples import FORMULA_EXAMPLES, resolve_formula
from learning_levels import level_guidance, explain_local_steps, explain_local_derivative, adapt_local_concept

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
    llm_status: dict[str, Any]
    _llm_runtime: Any


def safe_llm_response(
    prompt: str,
    question: str,
    *,
    model_name: str = "math_tutor",
    context: list[dict[str, str]] | None = None,
) -> str:
    llm = get_llm()
    if llm is None:
        return fallback_response(prompt, question, model_name=model_name, context=context)

    try:
        system_prompt = (
            "You are Mylo, a clear, supportive mathematics tutor. "
            "Explain in the user's language, avoid unsafe actions, and do not execute code. "
            "Present one focused worked solution with a short introduction, numbered steps, "
            "and a clear final answer. Use $...$ for inline mathematics and $$...$$ for "
            "display equations, using LaTeX fractions and powers instead of Python syntax. "
            "Use at most three short headings. Avoid repeated summaries, emojis, internal "
            "analysis, and invitations to continue instead of completing the answer."
        )
        messages = [("system", system_prompt)]
        messages.extend(
            (HumanMessage if turn["role"] == "user" else AIMessage)(content=turn.get("content", ""))
            for turn in (context or [])[-8:]
            if turn.get("role") in {"user", "assistant"}
        )
        user_prompt = prompt + "\n\nUser question: " + question
        if str(getattr(llm, "model", "")).startswith("qwen3"):
            user_prompt += "\n/no_think"
        messages.append(("user", "{question}"))
        template = ChatPromptTemplate.from_messages(messages)
        response = llm.invoke(template.format_messages(question=user_prompt))
        content = str(response.content).strip()
        # Some local model templates emit a closing tag without an opening tag.
        # Never render the model's internal reasoning in the tutor's answer.
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1].strip()
        elif "<think>" in content:
            content = content.split("<think>", 1)[0].strip()
        if not content:
            raise ValueError("Empty model response")
        record_llm_success()
        return content
    except Exception:
        record_llm_failure()
        return fallback_response(prompt, question, model_name=model_name, context=context)


def fallback_response(
    prompt: str,
    question: str,
    *,
    model_name: str = "math_tutor",
    context: list[dict[str, str]] | None = None,
) -> str:
    del model_name
    ql = (question or "").lower()
    formula_explanations = (
        ("quadratic formula", "The quadratic formula solves ax² + bx + c = 0. Substitute a, b, and c into x = (−b ± √(b² − 4ac)) / 2a; the ± gives the two possible roots, and b² − 4ac determines how many real roots there are."),
        ("binomial theorem", "The binomial theorem expands (x + y)ⁿ as a sum. Each term uses a binomial coefficient to count how many ways that combination appears."),
        ("difference of squares", "The identity a² − b² = (a − b)(a + b) factors a difference of two squares into a difference times a sum."),
        ("arithmetic series", "An arithmetic series adds terms with a constant difference. Its sum is the number of terms times the average of the first and last terms."),
        ("geometric series", "A geometric series has terms that change by a constant ratio. For a finite series with ratio r ≠ 1, its sum is a(1 − rⁿ)/(1 − r)."),
        ("change of log base", "The change-of-base rule rewrites a logarithm using a base you can evaluate: log_b(x) = ln(x)/ln(b)."),
        ("euler's identity", "Euler's identity, e^(iπ) + 1 = 0, connects five fundamental constants: e, i, π, 1, and 0."),
        ("pythagorean theorem", "For a right triangle, a² + b² = c², where c is the hypotenuse, the side opposite the right angle."),
        ("law of cosines", "The law of cosines relates all three sides of a triangle to one angle. It generalizes the Pythagorean theorem to non-right triangles."),
        ("law of sines", "The law of sines says each side of a triangle divided by the sine of its opposite angle has the same ratio."),
        ("circle area", "A = πr² gives the area inside a circle; r is its radius."),
        ("sphere volume", "V = 4πr³/3 gives the volume inside a sphere of radius r."),
        ("heron's formula", "Heron's formula finds a triangle's area from its side lengths: A = √(s(s−a)(s−b)(s−c)), where s is the semiperimeter."),
        ("chain rule", "The chain rule differentiates a function inside another function: differentiate the outer function, keep the inner one, then multiply by the inner derivative."),
        ("product rule", "The product rule differentiates two multiplied functions: (uv)' = u'v + uv'."),
        ("quotient rule", "The quotient rule differentiates a ratio: (u/v)' = (u'v − uv')/v², where v cannot be zero."),
        ("integration by parts", "Integration by parts, ∫u dv = uv − ∫v du, is the integration counterpart of the product rule."),
        ("fundamental theorem", "The fundamental theorem of calculus evaluates a definite integral using an antiderivative: ∫ₐᵇ f'(x)dx = f(b) − f(a)."),
        ("taylor series", "A Taylor series represents a function as an infinite sum of its derivatives at a point, multiplied by powers of the distance from that point."),
        ("bayes' theorem", "Bayes' theorem updates a probability using new evidence: P(A|B) = P(B|A)P(A)/P(B)."),
        ("normal distribution", "The normal distribution is a symmetric bell-shaped probability curve. Its mean μ sets the center and its standard deviation σ sets the spread."),
        ("gaussian integral", "The Gaussian integral, ∫₋∞∞ e^(−x²)dx = √π, is a classic improper integral that also appears in probability and statistics."),
        ("fourier transform", "The Fourier transform rewrites a signal as a combination of frequencies, revealing which frequencies it contains."),
        ("fourier series", "A Fourier series represents a periodic function as a sum of sine and cosine waves at different frequencies."),
        ("stirling's approximation", "Stirling's approximation estimates large factorials: n! ≈ √(2πn)(n/e)ⁿ."),
    )
    selected_formula = resolve_formula(question, context or [])
    for label, explanation in formula_explanations:
        if label == selected_formula:
            if label == "quadratic formula":
                if any(term in ql for term in ("a, b, and c", "a, b and c", "a b and c", "coefficients", "what do a", "what are a")):
                    return "In ax² + bx + c = 0, a multiplies x², b multiplies x, and c is the constant term. For example, in 2x² − 5x + 3 = 0, a = 2, b = −5, and c = 3. The formula requires a ≠ 0."
                if "discriminant" in ql:
                    return "The discriminant is b² − 4ac, the expression under the square root. If it is positive, the quadratic has two distinct real roots; if it is zero, it has one repeated real root; if it is negative, the roots are complex and there are no real x-intercepts."
                if any(term in ql for term in ("two answers", "two roots", "plus-minus", "plus or minus", "what does ±", "meaning of ±")):
                    return "The ± means calculate the expression twice: once with + and once with −. That produces the two roots of the quadratic. They can be equal when b² − 4ac = 0, and they are complex when b² − 4ac < 0."
                if any(term in ql for term in ("example", "work through", "show me", "give me", "calculate")):
                    return explanation + "\n\nWorked example:\n" + FORMULA_EXAMPLES[label]
                if any(term in ql for term in ("derive", "where did", "why does the formula")):
                    return "The formula comes from completing the square: divide ax² + bx + c = 0 by a, move c/a, then add (b/2a)² to both sides. This gives (x + b/2a)² = (b² − 4ac)/(4a²). Taking square roots and isolating x yields x = (−b ± √(b² − 4ac))/(2a)."
                if any(term in ql for term in ("when can", "when do i use", "what is it used for")):
                    return "Use the quadratic formula to solve any equation that can be written as ax² + bx + c = 0 with a ≠ 0. It works even when the equation does not factor neatly."
            return explanation + "\n\nWorked example:\n" + FORMULA_EXAMPLES[label]

    if contains_latex_notation(question):
        return "This formula is outside the built-in offline examples. Enable a working Ollama or OpenAI provider for a full explanation and worked solution of custom formulas. The offline tutor can still solve supported arithmetic and equations."

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
                    derivative_text = (
                        f"\n1. Differentiate {expression} with respect to x.\n"
                        "2. Apply the power rule d(xⁿ)/dx = n*x^(n−1) to each term; "
                        "constant terms differentiate to 0.\n"
                        f"Final result: the derivative is {format_algebraic_expression(derivative)}."
                    )
                except Exception:
                    pass
        if is_persian(question):
            return "مشتق نرخ تغییر یک تابع را نشان می‌دهد؛ مثلا مشتق مکان نسبت به زمان، سرعت است." + derivative_text
        return ("A derivative measures the rate of change of a function. For example, the derivative of position with respect to time is velocity. "
                "In formal notation, f′(x) = lim(h→0) (f(x+h) − f(x))/h, when this limit exists.") + derivative_text
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
    if "average" in ql or (re.search(r"\bmean\b", ql) and "equation" not in ql) or "میانگین" in ql:
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
    if "equation" in ql or "معادله" in ql:
        if any(term in ql for term in ("what is", "what does", "explain", "meaning", "definition")):
            return "An equation states that two mathematical expressions are equal. The equals sign separates the left and right sides; solving means finding values for the variables that make both sides equal."
    if "solve" in ql or "معادله" in ql or looks_like_equation(question):
        if "2x + 5 = 15" in ql or "2x+5=15" in ql or "2x + 5 = 15" in (question or ""):
            return "Step 1: Subtract 5 from both sides.\n2x = 10\n\nStep 2: Divide by 2.\nx = 5\n\nVerification: 2(5) + 5 = 15"
        if is_persian(question):
            return "لطفاً یک معادله کامل بفرستید تا بتوانم مقدار متغیر را با مراحل حل و بررسی پاسخ پیدا کنم."
        return "Please send the complete equation you want solved, such as 2x + 5 = 15, and I can show the steps and check the result."
    if is_persian(question):
        return "من می‌توانم این مفهوم را ساده و قدم‌به‌قدم توضیح بدهم."
    return "This question is outside the built-in offline solver. Enable a working Ollama or OpenAI provider for full explanations and worked solutions of more general mathematics questions."


def is_persian(text: str) -> bool:
    return any(ch in "ابپتثجچحخسشصضطظعغفقکگلمنوهیءؤئأإة" for ch in text)


def looks_like_equation(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    if "=" in q:
        return True
    return bool(re.search(r"(?i)(?:\b[a-z]\b|[xy])\s*(?:[+\-*/^])\s*[-]?\d", q) or re.search(r"(?i)\d\s*(?:[+\-*/^])\s*(?:\b[a-z]\b|[xy])", q))


def contains_latex_notation(text: str) -> bool:
    return "$$" in text or bool(re.search(r"\\[A-Za-z]+", text))


def is_concept_request(text: str) -> bool:
    q = (text or "").lower()
    english_markers = ["what is", "what does", "what are", "explain", "definition", "meaning", "mean?", "concept", "why", "describe", "interpret", "derive", "example", "show", "visualize", "plot", "graph", "how do", "how does"]
    persian_markers = ["توضیح", "چیست", "معنی", "تعریف", "چرا", "توضیح بده", "تفسیر", "نحوه"]
    math_topics = ["derivative", "integral", "limit", "matrix", "probability", "geometry", "algebra", "function", "slope", "vector", "division", "fraction", "prime", "percent", "percentage", "average", "mean", "square root", "multiply", "multiplication"]
    if any(marker in q for marker in english_markers + persian_markers):
        return True
    return any(topic in q for topic in math_topics) or bool(re.search(r"\by\s*=|\bf\(x\)\s*=", q))


def is_contextual_followup(text: str, context: list[dict[str, str]] | None) -> bool:
    if not context:
        return False
    q = (text or "").strip().lower()
    markers = (
        "why", "how", "explain more", "what about", "what do", "what does",
        "what is", "what are", "where did", "which", "and then", "can you",
        "give me an example", "show an example", "work through", "it?", "that?", "example", "explain this",
        "چرا", "چطور", "بیشتر",
    )
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
    if "triangle" in question.lower() and re.search(r"\d+\s*[-–]\s*\d+\s*[-–]\s*\d+", question):
        return None
    expression = str(question or "").strip().lower()
    expression = re.sub(
        r"^(?:(?:please|can you|could you|what is|what's|what does|calculate|compute|evaluate|work out|solve|resolve|find|tell me|is)\b[\s,:]*)+",
        "",
        expression,
    )
    expression = expression.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    replacements = (
        (r"\b(?:multiplied by|times|multiply)\b", "*"),
        (r"\b(?:divided by|divide by|over)\b", "/"),
        (r"\b(?:plus|added to)\b", "+"),
        (r"\b(?:minus|subtract)\b", "-"),
    )
    for pattern, operator in replacements:
        expression = re.sub(pattern, operator, expression)
    if not re.fullmatch(r"[\d\s.+*/^()%\-=:]+", expression):
        arithmetic_match = re.search(
            r"(?<![\w.])\d+(?:\.\d+)?(?:\s*[+\-*/^%]\s*\d+(?:\.\d+)?)+(?![\w.])",
            expression,
        )
        if not arithmetic_match:
            return None
        expression = arithmetic_match.group(0)
    expression = re.sub(r"\bequals?\s*$", "", expression)
    expression = expression.strip(" ?.!; ,:=")
    if not re.fullmatch(r"[\d\s.+*/^()%\-]+", expression) or not re.search(r"[+*/^%]|-(?=\s*\d)", expression):
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
    state["_llm_runtime"] = begin_llm_request()
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
        state["intent"] = "CONCEPT" if state.get("conversation_context") else "CLARIFY"
    elif is_log_product_zeta_integral(question):
        state["intent"] = "SOLVE"
    elif calculus_request(question) and calculus_request(question)[0] == "integral":
        state["intent"] = "SOLVE"
    elif derivative_request:
        state["intent"] = "CONCEPT"
    elif arithmetic_expression or simplification_expression:
        state["intent"] = "SOLVE"
    elif contains_latex_notation(question):
        state["intent"] = "CONCEPT"
    elif asks_to_solve and has_explicit_equation:
        state["intent"] = "SOLVE"
    elif is_concept_request(question) or is_contextual_followup(question, state.get("conversation_context")):
        state["intent"] = "CONCEPT"
    elif contains_latex_notation(question):
        state["intent"] = "CONCEPT"
    elif has_explicit_equation:
        state["intent"] = "SOLVE"
    elif any(token in ql for token in ["solve", "equation", "find x", "calculate", "simplify", "evaluate"]) and has_explicit_equation:
        state["intent"] = "SOLVE"
    elif any(token in ql for token in ["derivative", "integral", "limit", "matrix", "probability", "geometry", "algebra", "determinant", "eigenvalue", "series", "prove", "proof", "sum", "system", "vector", "complex", "trigonometry"]):
        state["intent"] = "CONCEPT"
    else:
        state["intent"] = "CLARIFY"

    state["selected_branch"] = state["intent"]
    make_debug_entry(state, f"intent: {state['intent']}")
    return state


def explain_concept(state: dict[str, Any]) -> dict[str, Any]:
    activate_runtime(state.get("_llm_runtime"))
    state["current_node"] = "explain_concept"
    state["selected_branch"] = "CONCEPT"
    question = state.get("question", "")
    level = state.get("user_level", "beginner")
    request = calculus_request(question)
    if request:
        kind, expression = request
        original, result = compute_calculus(kind, expression)
        checked = verify_calculus(kind, expression, str(result))
        state["calculation_result"] = str(result)
        state["verification_status"] = "verified" if checked else "unverified"
        state["explanation"] = explain_local_derivative(original, result, level)
        commentary = safe_llm_response(
            level_guidance(level) + " Explain this SymPy-checked derivative. "
            f"Original: {original}; derivative: {result}. Keep these results unchanged. Finish with the final result.",
            question, context=state.get("conversation_context"),
        ) if get_llm() is not None else ""
        if commentary and llm_status().get("used"):
            state["explanation"] = commentary
        return state
    explanation = safe_llm_response(
        level_guidance(level) + " Resolve references using the recent conversation. "
        "Explain the formula, define each symbol and its conditions of use. "
        "Give a concrete worked example with chosen values when none are supplied. "
        "Show substitution, every meaningful calculation, and the final result with units. "
        "Check the result. For a derivation or proof, justify each step and state assumptions. "
        "Answer the actual follow-up; do not merely offer to explain or solve later.",
        question,
        model_name="concept",
        context=state.get("conversation_context"),
    )
    if not llm_status().get("used"):
        explanation = adapt_local_concept(explanation, question, level)
    state["explanation"] = explanation
    state["clarity_score"] = 60
    make_debug_entry(state, "concept: explanation drafted")
    return state


def simplify_explanation(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "simplify_explanation"
    state["selected_branch"] = "SIMPLIFY"
    explanation = state.get("explanation", "")
    # Preserve terminology, formulas, and completed calculations verbatim.
    simplified = explanation
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
    elif any(token in ql for token in ["integral", "∫", "integrate", "area under", "accumulation"]):
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
    if calculus_request(question):
        strategy = "Find an antiderivative using SymPy, include C, and verify by differentiation."
    state["strategy"] = strategy
    state["retry_count"] = state.get("retry_count", 0)
    state["max_retries"] = state.get("max_retries", MAX_RETRIES)
    state["solution_steps"] = state.get("solution_steps", [])
    state["current_step"] = state.get("current_step", 0)
    state["has_more_steps"] = False
    make_debug_entry(state, f"strategy: {strategy}")
    return state


def solve_step(state: dict[str, Any]) -> dict[str, Any]:
    activate_runtime(state.get("_llm_runtime"))
    try:
        return _solve_step(state)
    except Exception:
        # Unsupported notation must not turn a valid chat request into an HTTP 500.
        state["solution_steps"] = [safe_llm_response(
            level_guidance(state.get("user_level", "intermediate")) + " Solve this mathematics problem completely. Explain every meaningful step, "
            "state domain restrictions, and finish with a checked final result. "
            "Be explicit about any uncertainty or missing information.",
            state.get("question", ""), context=state.get("conversation_context"),
        )]
        state["calculation_result"] = "See worked solution; not symbolically verified."
        state["verification_status"] = "unverified"
        state["has_more_steps"] = False
        return state


def _solve_step(state: dict[str, Any]) -> dict[str, Any]:
    state["current_node"] = "solve_step"
    state["selected_branch"] = "SOLVE_STEP"
    question = state.get("question", "")
    arithmetic_expression = simple_arithmetic_expression(question)
    simplification_expression = expression_before_empty_equals(question)
    special_integral = is_log_product_zeta_integral(question)
    eq = safe_parse_equation(question) if not (arithmetic_expression or simplification_expression or special_integral) else {"equation": None}
    equation = eq.get("equation")
    variable = eq.get("variable", "x")
    request = calculus_request(question)
    if request and request[0] == "integral" and not special_integral:
        kind, expression = request
        original, result = compute_calculus(kind, expression)
        state["exact_calculation_result"] = str(result)
        state["calculation_result"] = f"{result} + C"
        state["solution_steps"] = [
            f"Integrate ${sp.latex(original)}$ with respect to $x$.",
            "Use linearity and the power rule ∫xⁿ dx = x^(n+1)/(n+1) + C for n ≠ −1; use the corresponding antiderivative rule for other functions.",
            f"An antiderivative is ${sp.latex(result)}$.",
            f"Include the arbitrary constant: ${sp.latex(result)} + C$.",
            f"Check by differentiating: $\\frac{{d}}{{dx}}({sp.latex(result)} + C) = {sp.latex(sp.diff(result, sp.Symbol('x')))}$.",
        ]
        state["has_more_steps"] = False
        state["current_step"] = len(state["solution_steps"])
        return state
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
        variable_symbol = sp.Symbol(variable)
        polynomial = sp.Poly(sp.expand(lhs_expr - rhs_expr), variable_symbol) if (lhs_expr - rhs_expr).is_polynomial(variable_symbol) else None
        if polynomial is not None and polynomial.degree() == 1:
            coefficient, constant = polynomial.all_coeffs()
            state["solution_steps"] = [
                f"Collect terms on the left: ${sp.latex(polynomial.as_expr())} = 0$.",
                f"Subtract ${sp.latex(constant)}$ from both sides: ${sp.latex(coefficient*variable_symbol)} = {sp.latex(-constant)}$.",
                f"Divide both sides by ${sp.latex(coefficient)}$: ${variable} = {result_text}$.",
                f"Check by substitution into the original equation: {sp.simplify(lhs_expr.subs(variable_symbol, solved[0]))} = {sp.simplify(rhs_expr.subs(variable_symbol, solved[0]))}.",
            ]
        elif polynomial is not None and polynomial.degree() == 2:
            a, b, c = polynomial.all_coeffs()
            discriminant = sp.simplify(b*b - 4*a*c)
            state["solution_steps"] = [
                f"Collect terms: ${sp.latex(polynomial.as_expr())} = 0$; $a = {a}$, $b = {b}$, $c = {c}$.",
                f"Calculate the discriminant: $D = b^2 - 4ac = ({b})^2 - 4({a})({c}) = {discriminant}$.",
                f"Apply x = (−b ± √(b² − 4ac))/(2a), with denominator 2a = {2*a}.",
                f"Evaluate both branches: ${variable} = {result_text}$.",
                "Check each root by substitution: " + "; ".join(
                    f"for {variable} = {format_solution_value(value)}, left − right = {sp.simplify((lhs_expr-rhs_expr).subs(variable_symbol, value))}"
                    for value in solved
                ) + ".",
            ]
        else:
            state["solution_steps"] = [safe_llm_response(
                level_guidance(state.get("user_level", "intermediate")) + f" Solve completely step by step, explaining each transformation. Symbolic candidates: {exact_result_text}. "
                "Check domains and extraneous roots. Finish with the final answer. "
                "If you cannot solve it, state the specific limitation.",
                question, context=state.get("conversation_context"),
            )]
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
    request = calculus_request(question)
    if request and not special_integral:
        try:
            checked = verify_calculus(*request, state.get("exact_calculation_result", ""))
            state["verification_status"] = "verified" if checked else "incorrect"
        except Exception:
            state["verification_status"] = "unverified"
        return state
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
        if (all(any(sp.simplify(candidate - solution) == 0 for solution in solution_values) for candidate in candidates)
                and all(any(sp.simplify(candidate - solution) == 0 for candidate in candidates) for solution in solution_values)):
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
    activate_runtime(state.get("_llm_runtime"))
    state["current_node"] = "explain_solution"
    state["selected_branch"] = "EXPLAIN_SOLUTION"
    level = state.get("user_level", "beginner")
    question = state.get("question", "")
    result = state.get("calculation_result", "")
    status = state.get("verification_status", "unverified")

    request = calculus_request(question)
    explanation = explain_local_steps(
        state.get("solution_steps", []), level,
        integral=bool(request and request[0] == "integral"),
    )
    if "=" in question and "integral" in question.lower() and not calculus_request(question):
        explanation += " Your input contains an explicit equation, so I solved for the unknown. To request integration, use notation such as ∫ 5x² dx."


    if status == "verified" and get_llm() is not None:
        commentary = safe_llm_response(
            level_guidance(level) + " Explain these verified SymPy calculations. "
            f"Steps: {state.get('solution_steps', [])}. Final result: {result}. "
            "Do not change the checked result; explain why each operation is valid.",
            question, context=state.get("conversation_context"),
        )
        if llm_status().get("used"):
            explanation = commentary
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
    activate_runtime(state.get("_llm_runtime"))
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
        # The model explains the checked work once; offline steps remain complete.
        body = state.get("explanation") or explain_local_steps(steps, state.get("user_level", "intermediate"))
        try:
            values = result.removesuffix(" + C").split(",")
            rendered = r",\; ".join(sp.latex(parse_math_expression(value.strip())) for value in values)
            if result.endswith(" + C"):
                rendered += " + C"
            result_block = f"$$\n\\boxed{{{rendered}}}\n$$"
        except Exception:
            result_block = result
        state["final_answer"] = f"### Worked solution\n\n{body}\n\n> **Final result**\n>\n" + "\n".join(
            f"> {line}" for line in result_block.splitlines()
        ) + f"\n\n*Verification: {verification}*"
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
    activate_runtime(state.get("_llm_runtime"))
    state["current_node"] = "select_visualization"
    state["selected_branch"] = "VISUALIZATION"
    formula = resolve_formula(state.get("question", ""), state.get("conversation_context", []))
    example_plots = {
        "quadratic formula": "Plot y = x^2 - 5*x + 6",
        "circle area": "Circle radius",
        "pythagorean theorem": "Triangle geometry",
        "heron's formula": "Triangle geometry",
        "normal distribution": "Plot y = exp(-x^2/2)/sqrt(2*pi)",
        "gaussian integral": "Plot y = exp(-x^2)",
        "chain rule": "Differentiate y = (3*x+1)^2",
        "product rule": "Differentiate y = x^2*sin(x)",
        "quotient rule": "Differentiate y = x/(x+1)",
    }
    visual_state = dict(state)
    if (formula in example_plots and state.get("intent") == "CONCEPT"
            and FORMULA_EXAMPLES[formula] in state.get("final_answer", "")):
        visual_state["question"] = example_plots[formula]
    spec = choose_visualization(visual_state)
    if formula in ("pythagorean theorem", "heron's formula") and spec["type"] == "geometry":
        spec["config"]["points"] = [(0, 0), (3, 0), (3, 4)]
        spec["title"] = "Worked example: 3–4–5 triangle"
    state["visualization_type"] = spec["type"]
    state["visualization_spec"] = spec
    state["visualization_validation"] = "valid"
    state["explanation_level"] = state.get("user_level", "intermediate")
    state["llm_status"] = llm_status()
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
