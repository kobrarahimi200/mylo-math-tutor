import type { ChatResponse, Level, ConversationSummary, SavedConversation } from "../types";

export async function sendMessage(message: string, level: Level, conversationId?: string, replaceLatest = false): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, level, conversationId, ...(replaceLatest ? { replaceLatest: true } : {}) }),
  });
  if (!response.ok) {
    const details = await response.json().catch(() => null);
    throw new Error(details?.detail?.[0]?.msg ?? details?.detail ?? "Mylo could not answer that just now.");
  }
  return response.json() as Promise<ChatResponse>;
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await fetch("/api/conversations");
  if (!response.ok) throw new Error("Could not load saved chats. Check that the API is running.");
  return response.json() as Promise<ConversationSummary[]>;
}

export async function getConversation(id: string): Promise<SavedConversation> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error("Could not open this saved chat. It may no longer be available.");
  return response.json() as Promise<SavedConversation>;
}
