import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatResponse } from "./types";

const { sendMessage } = vi.hoisted(() => ({ sendMessage: vi.fn() }));
vi.mock("./lib/api", () => ({ sendMessage }));
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