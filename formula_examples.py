from __future__ import annotations

"""Complete, deterministic worked examples for the formula picker."""

FORMULA_EXAMPLES = {
    "quadratic formula": "1. Solve x² − 5x + 6 = 0; a = 1, b = −5, c = 6 (a ≠ 0).\n2. Discriminant: b² − 4ac = 25 − 24 = 1.\n3. Substitute: x = (5 ± √1)/2 = (5 ± 1)/2.\n4. Calculate both branches: x = 6/2 = 3 or x = 4/2 = 2.\nFinal result: x = 2 or 3. Check: 4 − 10 + 6 = 0 and 9 − 15 + 6 = 0. The graph crosses the x-axis at both roots.",
    "binomial theorem": "1. Expand (x + 2)³; n = 3, y = 2, with nonnegative integer n.\n2. The binomial coefficients are 1, 3, 3, 1.\n3. Substitute: x³ + 3x²(2) + 3x(2²) + 2³.\nFinal result: x³ + 6x² + 12x + 8. Check at x = 1: 27 = 1 + 6 + 12 + 8.",
    "difference of squares": "1. Factor 9x² − 16; identify a = 3x and b = 4.\n2. Substitute into a² − b² = (a − b)(a + b).\nFinal result: (3x − 4)(3x + 4). Check by expansion: 9x² + 12x − 12x − 16 = 9x² − 16.",
    "arithmetic series": "1. Add 3 + 7 + 11 + 15 + 19. There are n = 5 terms, first term a₁ = 3, last term aₙ = 19; the common difference is 4.\n2. Substitute: S₅ = 5(3 + 19)/2.\n3. Calculate: 5 × 22/2 = 55.\nFinal result: 55. Check by direct addition: 3 + 7 + 11 + 15 + 19 = 55.",
    "geometric series": "1. Add 2 + 6 + 18 + 54; first term a = 2, ratio r = 3, number of terms n = 4.\n2. Substitute: S₄ = 2(1 − 3⁴)/(1 − 3).\n3. Calculate: 2(−80)/(−2) = 80.\nFinal result: 80. Check: 2 + 6 + 18 + 54 = 80. If r = 1, use Sₙ = na instead.",
    "change of log base": "1. Find log₂(8); the argument must be positive and the base positive and unequal to 1.\n2. Substitute: ln(8)/ln(2).\n3. Since ln(8) = ln(2³) = 3ln(2), cancel ln(2).\nFinal result: 3. Check: 2³ = 8.",
    "euler's identity": "1. Start with Euler's formula e^(iθ) = cos θ + i sin θ, where i² = −1 and θ is in radians.\n2. Set θ = π: cos π = −1 and sin π = 0.\n3. Thus e^(iπ) = −1 + i(0) = −1.\nFinal result: e^(iπ) + 1 = 0. Geometrically, a half-turn on the unit circle ends at (−1, 0).",
    "pythagorean theorem": "1. In a right triangle, let the perpendicular legs be a = 3 cm and b = 4 cm; c is the hypotenuse.\n2. Substitute: c² = 3² + 4² = 9 + 16 = 25.\n3. A length is positive, so c = √25 = 5.\nFinal result: c = 5 cm. Check: 3² + 4² = 5².",
    "law of cosines": "1. Let a = 3 cm, b = 4 cm, and their included angle C = 60°. Side c is opposite C.\n2. Substitute: c² = 3² + 4² − 2(3)(4)cos 60°.\n3. Use cos 60° = 1/2: c² = 25 − 12 = 13.\nFinal result: c = √13 ≈ 3.606 cm. Check: 1 < √13 < 7, so the triangle inequalities hold.",
    "law of sines": "1. Let A = 30°, B = 60°, and side a = 5 cm opposite A. Find b opposite B.\n2. Rearrange: b = a sin B/sin A.\n3. Substitute: b = 5(√3/2)/(1/2) = 5√3.\nFinal result: b ≈ 8.660 cm and C = 90°. Check: a/sin A = b/sin B = 10 cm. With side-side-angle data, check for two possible triangles.",
    "circle area": "1. Let radius r = 2 cm.\n2. Square the radius: r² = 4 cm².\n3. Substitute: A = π × 4.\nFinal result: A = 4π ≈ 12.566 cm². Check: doubling the radius multiplies area by four.",
    "sphere volume": "1. Let radius r = 3 cm.\n2. Cube the radius: r³ = 27 cm³.\n3. Substitute: V = (4/3)π(27) = 36π.\nFinal result: V ≈ 113.097 cm³. Check: the units are cubic and the volume is positive.",
    "heron's formula": "1. Take triangle sides a = 3, b = 4, c = 5 cm; they satisfy the triangle inequalities.\n2. Semiperimeter s = (3 + 4 + 5)/2 = 6 cm.\n3. Substitute: A = √(6(6−3)(6−4)(6−5)) = √36.\nFinal result: A = 6 cm². Check using the right triangle: (1/2) × 3 × 4 = 6 cm².",
    "chain rule": "1. Differentiate y = (3x + 1)². Inner function g(x) = 3x + 1; outer function f(u) = u².\n2. Compute f′(u) = 2u and g′(x) = 3.\n3. Multiply: y′ = 2(3x + 1) × 3.\nFinal result: y′ = 18x + 6. Check by expanding y = 9x² + 6x + 1 and differentiating each term.",
    "product rule": "1. Differentiate y = x² sin x (radians). Set u = x² and v = sin x.\n2. Compute u′ = 2x and v′ = cos x.\n3. Substitute into u′v + uv′.\nFinal result: y′ = 2x sin x + x² cos x. At x = 0 this equals 0, consistent with the horizontal tangent there.",
    "quotient rule": "1. Differentiate y = x/(x + 1), with x ≠ −1. Set u = x and v = x + 1.\n2. Compute u′ = 1 and v′ = 1.\n3. Substitute: y′ = ((x + 1) − x)/(x + 1)².\nFinal result: y′ = 1/(x + 1)² for x ≠ −1. Check by rewriting y = 1 − 1/(x + 1) and differentiating.",
    "integration by parts": "1. Find ∫x e^x dx. Choose u = x and dv = e^x dx.\n2. Then du = dx and v = e^x.\n3. Substitute: ∫x e^x dx = x e^x − ∫e^x dx.\nFinal result: (x − 1)e^x + C, where C is any constant. Check: its derivative is e^x + (x − 1)e^x = x e^x.",
    "fundamental theorem": "1. Evaluate ∫₀² 2x dx. The integrand is continuous on [0, 2].\n2. An antiderivative is F(x) = x² because F′(x) = 2x.\n3. Evaluate F(2) − F(0) = 4 − 0.\nFinal result: 4. Check: the region under y = 2x is a triangle of base 2 and height 4, with area 4.",
    "taylor series": "1. Approximate e^0.1 about a = 0; every derivative of e^x at 0 is 1.\n2. The series is e^x = 1 + x + x²/2! + x³/3! + ⋯, convergent for every real x.\n3. Through degree 3: 1 + 0.1 + 0.01/2 + 0.001/6 = 1.105166667.\nFinal result: e^0.1 ≈ 1.105166667. The error is at most e^0.1(0.1⁴)/24 < 0.00000461. Check against 1.105170186.",
    "bayes' theorem": "1. Let A mean a defective item and B a positive inspection. P(A) = 0.01, P(B|A) = 0.90, P(B|not A) = 0.05.\n2. Total probability: P(B) = 0.90(0.01) + 0.05(0.99) = 0.0585 > 0.\n3. Substitute: P(A|B) = 0.009/0.0585 = 2/13.\nFinal result: about 15.38%. Check with 10,000 items: 90 true positives and 495 false positives give 90/585 = 2/13.",
    "normal distribution": "1. Take mean μ = 0 and standard deviation σ = 1 (σ must be positive). Find the density at x = 0.\n2. Substitute: f(0) = exp(−(0−0)²/2)/√(2π).\n3. The exponent is 0, so exp(0) = 1.\nFinal result: f(0) = 1/√(2π) ≈ 0.398942. This is density, not point probability; P(X = 0) = 0. The bell curve is symmetric and has total area 1.",
    "gaussian integral": "1. Let I = ∫₋∞∞ e^(−x²) dx, which is positive and convergent.\n2. Square it: I² = ∫∫ e^(−x²−y²) dx dy over the plane; nonnegativity justifies the product integral.\n3. Use polar coordinates with Jacobian r: I² = ∫₀²π ∫₀∞ e^(−r²)r dr dθ.\n4. The inner integral is [−e^(−r²)/2]₀∞ = 1/2, so I² = π.\nFinal result: I = √π ≈ 1.772454; choose the positive root because the integrand is positive.",
    "fourier transform": "1. Use F(ω) = ∫₋∞∞ f(t)e^(−iωt) dt. Choose f(t) = e^(−t) for t ≥ 0 and 0 otherwise; ω is real.\n2. Combine exponents: F(ω) = ∫₀∞ e^(−(1+iω)t) dt.\n3. Antiderivative: −e^(−(1+iω)t)/(1+iω). Its limit at infinity is 0 because the real decay rate is 1.\nFinal result: F(ω) = 1/(1+iω). Check at ω = 0: F(0) = ∫₀∞ e^(−t) dt = 1.",
    "fourier series": "1. Expand f(x) = x on (−π, π), extended periodically with period 2π.\n2. Odd symmetry gives a₀ = aₙ = 0.\n3. bₙ = (2/π)∫₀π x sin(nx) dx = (2/π)[−x cos(nx)/n + sin(nx)/n²]₀π = 2(−1)^(n+1)/n.\nFinal result: x = 2(sin x − sin(2x)/2 + sin(3x)/3 − ⋯) for −π < x < π. At the jumps x = ±π, the series converges to the midpoint 0. Check at x = 0: every term vanishes.",
    "stirling's approximation": "1. Estimate 10! using n! ≈ √(2πn)(n/e)^n for large positive integers n.\n2. Substitute: √(20π)(10/e)^10.\n3. Evaluate: approximately 3,598,695.62.\nFinal result: 10! ≈ 3.599 million. Check against the exact 3,628,800: relative error ≈ 0.83%. This is an approximation, and its relative accuracy improves as n grows.",
}


def resolve_formula(question: str, context: list[dict[str, str]]) -> str | None:
    """Explicit current topics outrank history; reuse history only for references."""
    current = question.lower()
    for label in FORMULA_EXAMPLES:
        if label in current:
            return label
    if not any(word in current for word in ("this", "that", "example", "discriminant", "coefficients", "a, b", "two roots", "two answers", "plus", "derive", "explain more")) and current.strip("?!. ") != "it":
        return None
    for turn in reversed(context):
        if turn.get("role") == "user":
            for label in FORMULA_EXAMPLES:
                if label in turn.get("content", "").lower():
                    return label
    return None
