import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ProviderSettings } from "./ProviderSettings";

const overview = (configured: boolean) => ({
  active: { provider: "openai", model: "gpt-x", available: configured, error: "", notice: "OpenAI is unavailable (no credit). Using fallback provider Ollama (local)." },
  provider: "openai",
  fallback: "ollama",
  providers: [
    { id: "openai", label: "OpenAI", needsKey: true, configured, credentialSource: configured ? "keychain" : "none", model: "gpt-x", defaultModel: "gpt-x", paused: "" },
    { id: "ollama", label: "Ollama (local)", needsKey: false, configured: true, credentialSource: "none", model: "qwen3:4b", defaultModel: "qwen3:4b", paused: "" },
  ],
});

function mockApi(configured = false, testResult: object = { ok: false, kind: "billing", message: "The OpenAI API account has no remaining credit or quota." }) {
  const calls: { url: string; method: string; body?: string }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, method: init?.method ?? "GET", body: init?.body as string | undefined });
    const json = url.endsWith("/models") ? { models: ["gpt-x"] } : url.endsWith("/test") ? testResult : overview(configured);
    return new Response(JSON.stringify(json), { status: 200 });
  }));
  return calls;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("ProviderSettings", () => {
  it("shows the active provider, the fallback notice and masks the key field", async () => {
    mockApi(true);
    render(<ProviderSettings />);
    expect(await screen.findByTestId("active-provider")).toHaveTextContent("OpenAI · gpt-x");
    expect(screen.getByRole("status")).toHaveTextContent("Using fallback provider");
    expect(screen.getByLabelText("OpenAI API key")).toHaveAttribute("type", "password");
  });

  it("sends a key once, clears the field and never displays it", async () => {
    const calls = mockApi(false);
    render(<ProviderSettings />);
    const input = await screen.findByLabelText("OpenAI API key");
    await userEvent.type(input, "sk-secret-12345");
    await userEvent.click(screen.getByRole("button", { name: "Save key" }));
    await waitFor(() => expect(calls.some((c) => c.method === "PUT" && c.url.endsWith("/credential"))).toBe(true));
    expect(input).toHaveValue("");
    expect(document.body.textContent).not.toContain("sk-secret-12345");
  });

  it("shows a clear billing error from Test Connection", async () => {
    mockApi(true);
    render(<ProviderSettings />);
    await userEvent.click(await screen.findByRole("button", { name: "Test connection" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("no remaining credit or quota");
  });
});
