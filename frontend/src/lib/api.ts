import type { ChatResponse, Level } from "../types";

export async function sendMessage(message: string, level: Level, conversationId?: string): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, level, conversationId }),
  });
  if (!response.ok) {
    const details = await response.json().catch(() => null);
    throw new Error(details?.detail?.[0]?.msg ?? details?.detail ?? "Mylo could not answer that just now.");
  }
  return response.json() as Promise<ChatResponse>;
}