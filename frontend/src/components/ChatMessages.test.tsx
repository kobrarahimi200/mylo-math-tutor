import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { normalizeMath } from "../lib/math";
import type { Message } from "../types";
import { MessageBubble } from "./ChatMessages";

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

describe("provider status", () => {
  it("explains how to enable online answers when the API key is missing", () => {
    const message: Message = {
      id: "answer",
      role: "assistant",
      content: "For example, 2 + 2 = 4.",
      metadata: {
        llm: {
          configuredProvider: "openai",
          provider: "offline",
          model: "gpt-6-astra",
          available: false,
          used: false,
          error: "OpenAI is selected but OPENAI_API_KEY is missing.",
        },
        intent: "SOLVE",
        language: "en",
        verification: "verified",
        conversationMemory: true,
      },
    };

    render(<MessageBubble message={message} />);

    expect(screen.getByText("For example, 2 + 2 = 4.")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      "Online answers need an OpenAI API key. Add it in Settings (or the backend .env file), or choose another provider.",
    );
    expect(screen.getByRole("status")).not.toHaveTextContent("OPENAI_API_KEY");
  });
});
