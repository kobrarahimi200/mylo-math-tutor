import { describe, expect, it } from "vitest";
import { normalizeMath } from "../lib/math";

describe("math delimiters", () => {
  it("normalizes model TeX output for KaTeX", () => {
    expect(normalizeMath(String.raw`Roots: \(x = \pm i\).`)).toBe("Roots: $x = \\pm i$.");
    expect(normalizeMath(String.raw`\[\frac{1}{2}\]`)).toBe("\n$$\n\\frac{1}{2}\n$$\n");
  });
  it("preserves code examples", () => {
    const code = "`\\(x\\)`\n```latex\n\\[x\\]\n```";
    expect(normalizeMath(code)).toBe(code);
  });
});
