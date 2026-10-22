import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatResponse } from "./types";

const { sendMessage, listConversations, getConversation } = vi.hoisted(() => ({ sendMessage: vi.fn(), listConversations: vi.fn(), getConversation: vi.fn() }));
vi.mock("./lib/api", () => ({ sendMessage, listConversations, getConversation }));
vi.mock("react-plotly.js", () => ({ default: () => null }));
vi.mock("jsxgraph", () => ({ default: { JSXGraph: { initBoard: () => ({ create: () => undefined }), freeBoard: () => undefined } } }));

import App from "./App";

const answer: ChatResponse = {
  answer: "Answer: 2 + 2 = 4",
  conversationId: "test-conversation",
  visualizations: [{ type: "none", title: "", config: {} }],
  steps: [],
  metadata: { intent: "CONCEPT", language: "en", verification: "unverified", conversationMemory: true },
};

describe("chat workspace", () => {
  afterEach(cleanup);

  beforeEach(() => {
    sendMessage.mockReset();
    listConversations.mockReset().mockResolvedValue([]);
    getConversation.mockReset();
    localStorage.clear();
  });

  it("sends a question to the agent and renders its answer", async () => {
    sendMessage.mockResolvedValue(answer);
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "What is 2 + 2?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    expect(await screen.findByText("2 + 2 = 4")).toBeInTheDocument();
    expect(sendMessage).toHaveBeenCalledWith("What is 2 + 2?", "beginner", undefined);
  });

  it("inserts a LaTeX formula into the question before sending", async () => {
    sendMessage.mockResolvedValue(answer);
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "Solve this equation:" } });
    fireEvent.click(screen.getByRole("button", { name: "Formula" }));
    fireEvent.change(screen.getByRole("textbox", { name: "LaTeX formula" }), { target: { value: "\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}" } });
    fireEvent.click(screen.getByRole("button", { name: "Insert formula" }));
    const question = "Solve this equation:\n\n$$\n\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$";
    expect(screen.getByRole("textbox", { name: "Your math question" })).toHaveValue(question);
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await waitFor(() => expect(sendMessage).toHaveBeenCalledWith(question, "beginner", undefined));
  });

  it.each(["$$x^{2}+1$$", "\\[x^{2}+1\\]", "\\(x^{2}+1\\)", "$x^{2}+1$"])("unwraps pasted formula %s before sending", async (formula) => {
    sendMessage.mockResolvedValue(answer);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Formula" }));
    fireEvent.change(screen.getByRole("textbox", { name: "LaTeX formula" }), { target: { value: formula } });
    fireEvent.click(screen.getByRole("button", { name: "Insert formula" }));
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await waitFor(() => expect(sendMessage).toHaveBeenCalledWith("Explain the formula:\n\n$$\nx^{2}+1\n$$", "beginner", undefined));
  });

  it("keeps the preset name when adding a formula to an existing request", () => {
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "Give me an example of this" } });
    fireEvent.click(screen.getByRole("button", { name: "Formula" }));
    fireEvent.click(screen.getByRole("button", { name: "Insert Geometric series" }));
    expect((screen.getByRole("textbox", { name: "Your math question" }) as HTMLTextAreaElement).value).toContain("Geometric series:");
  });

  it.each(["preset", "manual"])("closes the %s picker and returns to the typing box after inserting", (kind) => {
    const scroll = vi.spyOn(Element.prototype, "scrollIntoView");
    render(<App />);
    const input = screen.getByRole("textbox", { name: "Your math question" }) as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "Please explain this formula" } });
    fireEvent.click(screen.getByRole("button", { name: "Formula" }));
    if (kind === "preset") {
      fireEvent.click(screen.getByRole("button", { name: "Insert Chain rule" }));
    } else {
      fireEvent.change(screen.getByRole("textbox", { name: "LaTeX formula" }), { target: { value: "x^2 + 1" } });
      fireEvent.click(screen.getByRole("button", { name: "Insert formula" }));
    }
    expect(input.value).toContain("Please explain this formula");
    expect(input.value).toContain("$$");
    expect(screen.queryByRole("textbox", { name: "LaTeX formula" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Formula" })).toHaveAttribute("aria-expanded", "false");
    expect(input).toHaveFocus();
    expect(input.selectionStart).toBe(input.value.length);
    expect(scroll.mock.contexts).toContain(input);
    scroll.mockRestore();
  });

  it("inserts at the saved cursor position and preserves text after it", () => {
    render(<App />);
    const input = screen.getByRole("textbox", { name: "Your math question" }) as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "Explain this then give an example" } });
    input.setSelectionRange(12, 12);
    fireEvent.select(input);
    fireEvent.click(screen.getByRole("button", { name: "Formula" }));
    fireEvent.change(screen.getByRole("textbox", { name: "LaTeX formula" }), { target: { value: "x^2" } });
    fireEvent.click(screen.getByRole("button", { name: "Insert formula" }));
    expect(input.value).toBe("Explain this\n\n$$\nx^2\n$$\n\n then give an example");
    expect(input).toHaveFocus();
    expect(input.selectionStart).toBe(input.value.indexOf("\n\n then"));
  });

  it("offers more than 15 one-click formula presets", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Formula" }));
    expect(screen.getAllByRole("button", { name: /^Insert (?!formula$)/ })).toHaveLength(25);
    fireEvent.click(screen.getByRole("button", { name: "Insert Quadratic formula" }));
    expect(screen.getByRole("textbox", { name: "Your math question" })).toHaveValue("Explain the quadratic formula:\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$");
  });

  it("sends the selected level and shows which level produced the answer", async () => {
    sendMessage.mockResolvedValue({ ...answer, metadata: { ...answer.metadata, explanationLevel: "advanced" } });
    render(<App />);
    fireEvent.keyDown(screen.getByRole("combobox", { name: "Explanation level" }), { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: /advanced/i }));
    expect(screen.getByText("Formal reasoning, assumptions, and edge cases.")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "Solve 2x + 5 = 15" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    expect(await screen.findByText("advanced explanation")).toBeInTheDocument();
    expect(sendMessage).toHaveBeenCalledWith("Solve 2x + 5 = 15", "advanced", undefined);
    expect(screen.getByText("Changing level updates your latest answer.")).toBeInTheDocument();
  });

  it("updates the latest answer when the level changes without repeating the question", async () => {
    sendMessage.mockResolvedValueOnce(answer).mockResolvedValueOnce({ ...answer, answer: "Advanced explanation of the same result.", metadata: { ...answer.metadata, explanationLevel: "advanced" } });
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "What is 2 + 2?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText("2 + 2 = 4");
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "My next question" } });
    fireEvent.keyDown(screen.getByRole("combobox", { name: "Explanation level" }), { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: /advanced/i }));
    expect(await screen.findByText("Advanced explanation of the same result.")).toBeInTheDocument();
    expect(screen.queryByText("2 + 2 = 4")).not.toBeInTheDocument();
    expect(screen.getAllByText("What is 2 + 2?")).toHaveLength(1);
    expect(sendMessage).toHaveBeenLastCalledWith("What is 2 + 2?", "advanced", "test-conversation", true);
    expect(screen.getByRole("textbox", { name: "Your math question" })).toHaveValue("My next question");
  });

  it("keeps the current answer when updating its level fails", async () => {
    sendMessage.mockResolvedValueOnce(answer).mockRejectedValueOnce(new Error("Ollama unavailable"));
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "2 + 2" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText("2 + 2 = 4");
    fireEvent.keyDown(screen.getByRole("combobox", { name: "Explanation level" }), { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: /advanced/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Your previous answer is still shown");
    expect(screen.getByText("2 + 2 = 4")).toBeInTheDocument();
  });

  it("ignores an older level response when levels change quickly", async () => {
    let finishAdvanced: (value: ChatResponse) => void = () => undefined;
    sendMessage.mockResolvedValueOnce(answer)
      .mockImplementationOnce(() => new Promise<ChatResponse>((resolve) => { finishAdvanced = resolve; }))
      .mockResolvedValueOnce({ ...answer, answer: "Intermediate answer." });
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "2 + 2" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText("2 + 2 = 4");
    const selector = screen.getByRole("combobox", { name: "Explanation level" });
    fireEvent.keyDown(selector, { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: /advanced/i }));
    expect(screen.getByText("Updating the answer for advanced level…")).toBeInTheDocument();
    fireEvent.keyDown(selector, { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: /intermediate/i }));
    await screen.findByText("Intermediate answer.");
    finishAdvanced({ ...answer, answer: "Outdated advanced answer." });
    await waitFor(() => expect(screen.queryByText("Mylo is working")).not.toBeInTheDocument());
    expect(screen.getByText("Intermediate answer.")).toBeInTheDocument();
    expect(screen.queryByText("Outdated advanced answer.")).not.toBeInTheDocument();
  });

  it("restores the last saved conversation after refresh and continues it", async () => {
    const saved = { id: "saved-1", title: "Saved math question", updatedAt: "2026-10-16T12:00:00Z", level: "advanced", messages: [
      { id: "u1", role: "user", content: "Saved math question" },
      { id: "a1", role: "assistant", content: "Saved answer.", metadata: { ...answer.metadata, explanationLevel: "advanced" } },
    ] };
    localStorage.setItem("mylo-active-conversation", saved.id);
    listConversations.mockResolvedValue([saved]);
    getConversation.mockResolvedValue(saved);
    sendMessage.mockResolvedValue({ ...answer, conversationId: saved.id });
    render(<App />);
    expect(await screen.findByText("Saved answer.")).toBeInTheDocument();
    expect(getConversation).toHaveBeenCalledWith(saved.id);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "Why?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await waitFor(() => expect(sendMessage).toHaveBeenCalledWith("Why?", "advanced", saved.id));
  });

  it("opens a saved chat from the sidebar and keeps it listed after starting a new chat", async () => {
    const saved = { id: "saved-2", title: "Triangle lesson", updatedAt: "2026-10-16T12:00:00Z", level: "beginner", messages: [
      { id: "u2", role: "user", content: "Triangle question" }, { id: "a2", role: "assistant", content: "A saved triangle explanation." },
    ] };
    listConversations.mockResolvedValue([saved]);
    getConversation.mockResolvedValue(saved);
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Triangle lesson" }));
    expect(await screen.findByText("A saved triangle explanation.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /New chat/ }));
    expect(screen.queryByText("A saved triangle explanation.")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Triangle lesson" })).toBeInTheDocument();
    expect(localStorage.getItem("mylo-active-conversation")).toBeNull();
  });

  it("shows the provider and actionable fallback errors from the agent", async () => {
    sendMessage.mockResolvedValue({ ...answer, metadata: { ...answer.metadata, llm: { configuredProvider: "ollama", provider: "offline", model: "qwen3:4b", available: false, used: false, error: "Run ollama pull qwen3:4b" } } });
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "2 + 2" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    expect(await screen.findByText("Offline fallback")).toBeInTheDocument();
    expect(screen.getByText("Run ollama pull qwen3:4b")).toBeInTheDocument();
  });

  it("shows a loading indicator while the agent responds", async () => {
    let resolveAnswer: (value: ChatResponse) => void = () => undefined;
    sendMessage.mockImplementation(() => new Promise<ChatResponse>((resolve) => { resolveAnswer = resolve; }));
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "What is a derivative?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    expect(screen.getByText("Mylo is working")).toBeInTheDocument();
    resolveAnswer(answer);
    await waitFor(() => expect(screen.queryByText("Mylo is working")).not.toBeInTheDocument());
  });

  it("shows a recoverable error when the API is unavailable", async () => {
    sendMessage.mockRejectedValue(new Error("API unavailable"));
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Your math question" }), { target: { value: "Help with calculus" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("API unavailable");
  });
});