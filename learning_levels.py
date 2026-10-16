"""Explicit teaching policies shared by model prompts and local explanations."""
from __future__ import annotations

import re
import sympy as sp

LEVEL_GUIDANCE = {
    "beginner": (
        "Assume no prior knowledge. Define each symbol before using it. Break every "
        "operation into small numbered steps and explain why it is allowed. Show "
        "substitution and arithmetic explicitly. Use plain language and a concrete "
        "example; introduce technical terms gently. Finish with a result and a simple check."
    ),
    "intermediate": (
        "Assume basic algebra and standard mathematical notation. Identify the method, "
        "show the essential transformations and calculations, and verify the answer. "
        "Skip definitions of familiar operations. Explain the key reasoning without "
        "a beginner analogy or unnecessary repetition."
    ),
    "advanced": (
        "Assume fluent algebra and calculus. Give a concise formal argument using "
        "precise notation. State relevant domains, hypotheses and exceptional cases. "
        "Justify nontrivial transformations or convergence claims, while compressing "
        "routine arithmetic. Include the final result and its verification. Do not "
        "repeat elementary definitions or beginner analogies."
    ),
}


def level_guidance(level: str) -> str:
    return f"Teaching level: {level.upper()}. {LEVEL_GUIDANCE.get(level, LEVEL_GUIDANCE['intermediate'])} The current level overrides the style of earlier conversation turns."


def explain_local_steps(steps: list[str], level: str, *, integral: bool = False) -> str:
    if level == "beginner":
        intro = (
            "Integration works backwards from a derivative: we want a function whose derivative is the given expression."
            if integral else
            "We will work one operation at a time. In an equation, both sides must stay equal, so apply the same operation to both sides."
        )
        notes = {
            "collect terms": "This puts the expression in a standard form so we can see its coefficients.",
            "subtract": "Subtracting the same amount from both sides keeps them equal.",
            "divide both": "Division undoes multiplication; divide both sides by the same nonzero number.",
            "discriminant": "The discriminant is the number under the square root. Positive means two real roots, zero means one repeated root, and negative means no real roots.",
            "both branches": "The plus-or-minus sign means doing the calculation twice, once with each sign.",
            "power rule": "For a power of x, add 1 to the exponent and divide by that new exponent. Keep any constant multiplier.",
            "arbitrary constant": "C represents any constant, because differentiating a constant gives zero.",
            "substitution": "Putting the answer back into the original equation checks that it really works.",
            "differentiating": "Taking the derivative should give back exactly the original integrand.",
            "order of operations": "Evaluate brackets and powers first, then multiplication and division, then addition and subtraction.",
            "like terms": "Only terms with the same variable and power can be combined; add or subtract their coefficients.",
        }
        expanded = []
        for index, step in enumerate(steps, 1):
            note = next((text for keyword, text in notes.items() if keyword in step.lower()), "")
            expanded.append(f"{index}. {step}" + (f"\n\n   {note}" if note else ""))
        return intro + "\n\n" + "\n\n".join(expanded)
    if level == "advanced":
        assumptions = (
            "Find a primitive on each interval of the integrand's domain. The additive constant is independent on disconnected intervals; differentiation verifies the primitive."
            if integral else
            "Preserve the domain through each transformation. Division requires a nonzero divisor; candidate roots must satisfy the original equation, particularly if denominators or radicals occur."
        )
        return "**Formal method.** " + assumptions + "\n\n" + "\n\n".join(steps)
    return "**Method.** " + ("Apply antiderivative rules, include C, and differentiate to check." if integral else "Use the appropriate algebraic or arithmetic rules, then check the result.") + "\n\n" + "\n\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def explain_local_derivative(original: sp.Expr, derivative: sp.Expr, level: str) -> str:
    f, df = sp.latex(original).replace(" x", "x"), sp.latex(derivative).replace(" x", "x")
    if level == "advanced":
        return (
            "Use the formal definition on the domain where the limit exists:\n\n"
            rf"$$f'(x)=\lim_{{h\to0}}\frac{{f(x+h)-f(x)}}{{h}}={df}.$$" + "\n\n"
            "The computed derivative was independently checked against this limit."
        )
    if level == "beginner":
        rule = "For each power of x, bring the exponent down as a multiplier, then subtract 1 from the exponent. A constant becomes zero." if original.is_polynomial(sp.Symbol("x")) else "Identify the function in each term and apply its derivative rule; use the chain rule when one function is inside another."
        steps = [f"Start with $f(x)={f}$.", rule]
        x = sp.Symbol("x")
        if original.is_polynomial(x):
            for term in sp.Add.make_args(sp.expand(original)):
                steps.append(f"Differentiate ${sp.latex(term)}$: ${sp.latex(sp.diff(term, x))}$.")
        steps.append(f"Combine the terms. Final result: $f'(x)={df}$.")
        return "A derivative measures the rate of change of a function.\n\n" + "\n\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)) + "\n\nThe result is checked using the derivative's limit definition."
    return f"Apply the sum, constant-multiple, and appropriate differentiation rules to $f(x)={f}$.\n\nFinal result: $f'(x)={df}$.\n\nVerified against the limit definition."


# Additional mathematical depth for offline concept explanations.
CONCEPT_NOTES = {
    "quadratic formula": (
        "In ax² + bx + c = 0, a is the number multiplying x², b multiplies x, and c is the constant. The sign ± means calculate once with plus and once with minus.",
        "For a ≠ 0, completing the square gives (x + b/(2a))² = (b² − 4ac)/(4a²). The discriminant classifies real-root multiplicity; over ℂ every quadratic has two roots counting multiplicity."
    ),
    "derivative": (
        "Think of a car: its speed tells you how quickly its position changes. The derivative gives that rate at one instant.",
        "Formally, differentiability at x requires the finite two-sided limit lim(h→0)(f(x+h)−f(x))/h. Differentiability implies continuity, but continuity alone is insufficient: |x| is continuous and not differentiable at 0."
    ),
    "integral": (
        "A definite integral adds tiny signed areas; an indefinite integral finds functions that differentiate back to the original expression. The + C covers their possible constants.",
        "For continuous f on an interval, any two primitives differ by a constant. The fundamental theorem links primitives to definite integration; improper integrals additionally require convergence at singularities or infinite endpoints."
    ),
    "pythagorean": (
        "A right angle is 90°. The hypotenuse is the longest side, opposite that angle. Squaring a length means multiplying it by itself.",
        "For perpendicular vectors u and v in Euclidean space, ⟨u,v⟩ = 0, so ‖u+v‖² = ‖u‖² + ‖v‖² by expanding the inner product. The perpendicularity assumption is essential."
    ),
    "circle": (
        "The radius runs from the center to the edge. Area counts square units inside the boundary, so the radius is squared.",
        "In polar coordinates the area is ∫₀²π∫₀ʳ ρ dρ dθ = πr² for r ≥ 0. The factor ρ is the coordinate transformation's Jacobian."
    ),
    "fraction": (
        "The bottom number says how many equal pieces make a whole; the top number counts how many pieces you have.",
        "For nonzero denominators, a/b = c/d iff ad = bc. Rational representations are equivalence classes; cancellation requires a common nonzero factor, not a common summand."
    ),
    "probability": (
        "Probabilities range from 0 (impossible) to 1 (certain). Counting favorable outcomes works when all outcomes have equal chances.",
        "A probability measure satisfies nonnegativity, normalization P(Ω)=1, and countable additivity on pairwise disjoint events. Conditional probabilities require a conditioning event of positive probability."
    ),
}


CONCEPT_NOTES.update({
    "binomial": ("A binomial has two terms. The exponent counts how many identical brackets are multiplied together; the coefficients count repeated choices.", "For n ∈ ℕ₀ the expansion follows by selecting k of the n factors to contribute y. Noninteger exponents require the generalized binomial series and convergence conditions."),
    "difference of squares": ("Identify the two quantities being squared. Multiply a difference by a sum; the two middle terms cancel.", "This polynomial identity holds over any commutative ring. In characteristic 2, the two factors coincide because subtraction equals addition."),
    "arithmetic series": ("A term is one number in the list. A constant difference means adding the same amount to reach the next term.", "Pairing aₖ with aₙ₊₁₋ₖ gives a₁+aₙ. Summing these n equal pairs counts the original sum twice, giving 2Sₙ = n(a₁+aₙ)."),
    "geometric series": ("The ratio is the multiplier between consecutive terms. The first term and number of terms tell you where the sum starts and stops.", "Subtract rSₙ from Sₙ to obtain (1−r)Sₙ = a(1−rⁿ). The finite formula requires r ≠ 1; an infinite geometric sum converges for |r| < 1."),
    "log base": ("A logarithm asks which exponent produces a given number. Changing the base lets you calculate that exponent with another logarithm.", "For real logarithms require x > 0, b > 0 and b ≠ 1. Taking ln of bʸ=x yields y ln b=ln x, justifying the change of base."),
    "euler": ("Here i is a number with i² = −1. The angle π radians is half a turn around a circle.", "Euler's formula follows by separating the absolutely convergent power series of exp(iθ) into even and odd terms, yielding cos θ+i sin θ."),
    "law of cosines": ("The included angle lies between the two known sides. The side opposite that angle is the one you calculate.", "Expanding ‖u−v‖² = ‖u‖²+‖v‖²−2⟨u,v⟩ and using ⟨u,v⟩=ab cos C yields the law of cosines."),
    "law of sines": ("Match each side to the angle directly opposite it before substituting numbers.", "In a nondegenerate Euclidean triangle, the common side-to-sine ratio is 2R, where R is the circumradius. Inverse sine may yield two admissible triangles in the SSA case."),
    "sphere": ("A sphere is a solid ball. Its radius goes from the center to its surface, and its volume is measured in cubic units.", "Integrating disk cross-sections gives V = ∫₋ʳʳ π(r²−z²) dz = 4πr³/3 for r ≥ 0."),
    "heron": ("The semiperimeter is half the sum of the three side lengths. Calculate it first, then substitute each side.", "The radicand is nonnegative for valid triangle side lengths. Strict triangle inequalities give positive area; equality corresponds to a degenerate triangle of area zero."),
    "chain rule": ("Find the inside function first. Differentiate the outside while keeping the inside, then multiply by the inside's derivative.", "If g is differentiable at x and f at g(x), then D(f∘g)(x)=Df(g(x))Dg(x). In several variables this is matrix multiplication of Jacobians."),
    "product rule": ("Differentiate one factor at a time while leaving the other unchanged, then add the two contributions.", "Insert and subtract u(x+h)v(x) in the difference quotient of uv. The two resulting limits give u′v+uv′ using differentiability and continuity."),
    "quotient rule": ("The bottom expression cannot be zero. Differentiate both pieces separately before substituting into the rule.", "Write u/v = uv⁻¹ and apply the product and chain rules. The result holds at points of differentiability where v ≠ 0."),
    "integration by parts": ("Choose which factor will be u and which will be integrated to give v. A useful choice makes the remaining integral simpler.", "Integrating the product identity d(uv)=u dv+v du yields the formula. For absolutely continuous u and v on [a,b], the definite version includes the boundary term [uv]ₐᵇ."),
    "fundamental theorem": ("Find a function whose derivative is the integrand. Evaluate it at the upper endpoint and subtract its value at the lower endpoint.", "For continuous f on [a,b], F(x)=∫ₐˣ f(t)dt satisfies F′=f in the interior. Any primitive G therefore gives ∫ₐᵇ f=G(b)−G(a)."),
    "taylor": ("The center is the point near which you build the approximation. Each extra term uses another derivative to improve the local match.", "An infinite derivative sequence alone does not guarantee equality to the Taylor series. Taylor's theorem requires control of the remainder; analyticity supplies local equality within the radius of convergence."),
    "bayes": ("Start with the original chance, then update it using how likely the observed evidence is under each possibility.", "For P(B)>0, equate P(A∩B)=P(A|B)P(B)=P(B|A)P(A). The denominator must include all relevant cases via the law of total probability."),
    "normal distribution": ("The mean is the center of the bell. The standard deviation controls its spread. Curve height is density; interval area is probability.", "Standardization Z=(X−μ)/σ gives a standard normal when σ>0. The density integrates to one; an individual point has probability zero."),
    "gaussian": ("Squaring the integral lets us view the calculation as an area problem in two dimensions. Polar coordinates then make it easier.", "Tonelli's theorem permits the nonnegative double integral. The polar transformation contributes Jacobian r; its radial integral is 1/2, giving I²=π."),
    "fourier transform": ("The transform tells you which frequencies make up a signal. Frequency zero measures its total signed area under this convention.", "For f ∈ L¹(ℝ), the displayed transform exists absolutely for every real ω. Inversion needs additional hypotheses, and factors of 2π depend on the chosen convention."),
    "fourier series": ("A periodic function repeats. Sines and cosines are repeating building blocks, each with a coefficient telling us how much to use.", "Under standard Dirichlet conditions the series converges to the midpoint of the one-sided limits at a jump. Pointwise convergence alone does not imply uniform convergence."),
    "stirling": ("The factorial n! multiplies all positive integers from 1 to n. Stirling's formula estimates that large product.", "For positive integers n, n! ∼ √(2πn)(n/e)ⁿ as n→∞. Taking logs is numerically preferable for large n; the first relative correction is 1/(12n)."),
    "function": ("Think of a rule that takes an allowed input and assigns it exactly one output. Different inputs may have the same output.", "A function f:X→Y assigns each x∈X exactly one element of Y. Domain, codomain and image are distinct; invertibility requires bijectivity onto the stated codomain."),
    "slope": ("Slope compares the change in height with the change across: rise divided by run.", "A secant slope is Δy/Δx for Δx≠0. A tangent slope is its limit where the derivative exists; vertical lines do not have a finite slope in y-as-a-function-of-x coordinates."),
})


CONCEPT_NOTES.update({
    "multiply": ("Multiplying a whole number counts equal groups; for example, three groups of four contain twelve items.", "Multiplication distributes over addition: a(b+c)=ab+ac. Repeated addition describes nonnegative integer factors; multiplication extends consistently to rational and real numbers."),
    "multiplication": ("Multiplying a whole number counts equal groups; for example, three groups of four contain twelve items.", "Multiplication distributes over addition. Zero is absorbing, one is the identity, and nonzero real numbers have unique multiplicative inverses."),
    "division": ("Division asks how to share into equal groups or how many equal groups fit into a quantity.", "For b≠0, a/b is the unique x satisfying bx=a. Division by zero is undefined in a field because zero has no multiplicative inverse."),
    "prime": ("Check whether the number has any positive factors other than one and itself. The number one is not prime.", "An integer p>1 is prime iff its only positive divisors are 1 and p. To test an integer n>1 by trial division, it suffices to check prime divisors at most √n."),
    "percent": ("Percent means out of one hundred. Divide by 100 to turn a percentage into a fraction or decimal.", "A p-percent change multiplies a value by 1+p/100; successive percentage changes multiply their factors rather than simply adding their rates."),
    "average": ("Add all the values first, then divide by how many values there are.", "For the arithmetic mean x̄, Σ(xᵢ−x̄)=0. It uniquely minimizes Σ(xᵢ−c)² over real c when at least one observation is present."),
    "mean": ("Add all the values first, then divide by how many values there are.", "The arithmetic mean minimizes the sum of squared deviations. It is sensitive to extreme observations; weighting changes both the numerator and its normalization."),
    "square root": ("A square root undoes squaring. For positive numbers, both a positive and negative number can have that square.", "For real a≥0, √a denotes the nonnegative principal root. The equation x²=a has roots ±√a for a>0, one repeated root at a=0, and no real roots for a<0."),
    "addition": ("Addition combines quantities. You can count on from the first number to add the second.", "Real addition is associative and commutative, with identity 0 and inverse −a for each a. These properties justify regrouping and cancellation."),
    "subtraction": ("Subtraction finds what remains or the difference between two quantities.", "Subtraction is addition of the additive inverse: a−b=a+(−b). It is neither commutative nor associative in general."),
    "algebra": ("A letter stands for an unknown value. We use the same operation on both sides to keep an equation balanced.", "Equivalent transformations preserve a solution set only on their domain of validity. Multiplication by a possibly zero expression or squaring both sides may introduce extraneous solutions."),
})


def adapt_local_concept(answer: str, question: str, level: str) -> str:
    if "outside the built-in" in answer:
        return answer
    text = (question + " " + answer).lower()
    note = next((notes for keyword, notes in CONCEPT_NOTES.items() if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", question.lower())), None)
    if note is None:
        note = next((notes for keyword, notes in CONCEPT_NOTES.items() if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text)), None)
    if level == "beginner" and note:
        if "derivative" in question.lower():
            answer = answer.split(" In formal notation,", 1)[0]
        return note[0] + "\n\n" + answer
    if level == "advanced" and note:
        return answer + "\n\n**Formal extension.** " + note[1]
    return answer
