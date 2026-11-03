import type { ChatResponse, Level, ConversationSummary, SavedConversation, ProviderOverview, ProviderTestResult } from "../types";

async function apiFetch(path: string, options?: RequestInit): Promise<Response> {
  try {
    const response = await fetch(path, options);
    if (response.status === 502 || response.status === 503 || response.status === 504) {
      throw new Error("The tutor API is unavailable. Start the backend on port 8000, then retry.");
    }
    return response;
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("Cannot connect to the tutor. Check your connection and that the backend is running on port 8000.");
    }
    throw error;
  }
}

export async function sendMessage(message: string, level: Level, conversationId?: string, replaceLatest = false): Promise<ChatResponse> {
  const response = await apiFetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, level, conversationId, ...(replaceLatest ? { replaceLatest: true } : {}) }),
  });
  if (!response.ok) {
    const details = await response.json().catch(() => null);
    const detail = details?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item: { msg?: string }) => item.msg ?? "Invalid request").join("; ")
      : typeof detail === "string" ? detail : response.status >= 500
        ? "The tutor backend failed to process this question. Check that the API is running on port 8000, then retry. Your formula is still available in the input."
        : `The request failed (HTTP ${response.status}). Please check your question and retry.`;
    throw new Error(message);
  }
  return response.json() as Promise<ChatResponse>;
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await apiFetch("/api/conversations");
  if (!response.ok) throw new Error("Could not load saved chats. Check that the API is running.");
  return response.json() as Promise<ConversationSummary[]>;
}

export async function getConversation(id: string): Promise<SavedConversation> {
  const response = await apiFetch(`/api/conversations/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error("Could not open this saved chat. It may no longer be available.");
  return response.json() as Promise<SavedConversation>;
}

async function providerRequest<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const response = await apiFetch(path, { method, ...(body === undefined ? {} : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  if (!response.ok) {
    const details = await response.json().catch(() => null);
    throw new Error(typeof details?.detail === "string" ? details.detail : `The request failed (HTTP ${response.status}).`);
  }
  return response.json() as Promise<T>;
}

export const getProviders = () => providerRequest<ProviderOverview>("/api/providers");
export const saveProviderSettings = (settings: { provider?: string; fallback?: string; models?: Record<string, string> }) => providerRequest<ProviderOverview>("/api/providers/settings", "PUT", settings);
// The key is write-only: the backend stores it in the OS credential store and never returns it.
export const saveProviderKey = (id: string, apiKey: string) => providerRequest<ProviderOverview>(`/api/providers/${encodeURIComponent(id)}/credential`, "PUT", { apiKey });
export const removeProviderKey = (id: string) => providerRequest<ProviderOverview>(`/api/providers/${encodeURIComponent(id)}/credential`, "DELETE");
export const listProviderModels = (id: string) => providerRequest<{ models: string[]; error?: string }>(`/api/providers/${encodeURIComponent(id)}/models`);
export const testProvider = (id: string) => providerRequest<ProviderTestResult>(`/api/providers/${encodeURIComponent(id)}/test`, "POST");
