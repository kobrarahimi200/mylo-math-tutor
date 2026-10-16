import { useEffect, useRef } from "react";
import { Bot, CheckCheck, UserRound } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import type { Message } from "../types";
import { VisualizationRenderer } from "./Visualizations";

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <article className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`message-avatar ${isUser ? "user-avatar" : "assistant-avatar"}`}>{isUser ? <UserRound size={16} /> : <Bot size={17} />}</div>
      <div className="message-body">
        <div className="message-meta"><strong>{isUser ? "You" : "Mylo"}</strong><span>{isUser ? "QUESTION" : message.metadata?.intent === "SOLVE" ? "WORKED SOLUTION" : "MATH TUTOR"}</span>{message.metadata?.verification === "verified" && <small className="verified-tag"><CheckCheck size={12} /> checked</small>}</div>
        {!isUser && message.metadata?.explanationLevel && <span className="answer-level">{message.metadata.explanationLevel} explanation</span>}
        {!isUser && message.metadata?.llm && <div className="provider-status" role="status">
          <small>{message.metadata.llm.provider === "offline" ? "Offline fallback" : `${message.metadata.llm.provider === "ollama" ? "Ollama" : "OpenAI"} · ${message.metadata.llm.model}`}{message.metadata.verification === "verified" ? " · SymPy checked" : ""}</small>
          {message.metadata.llm.error && <p>{message.metadata.llm.error}</p>}
        </div>}
        <div className={`message-content ${message.language === "fa" ? "rtl-message" : ""}`} dir={message.language === "fa" ? "rtl" : "auto"}>
          <ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]}>{message.content.replace(/^Answer:\s*/i, "")}</ReactMarkdown>
        </div>
        {!isUser && <VisualizationRenderer spec={message.visualization} />}
      </div>
    </article>
  );
}

export function MessageList({ messages, loading, loadingText = "Following the math through, step by step" }: { messages: Message[]; loading: boolean; loadingText?: string }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);
  return <div className="message-list" aria-live="polite">
    {messages.map((message) => <MessageBubble key={message.id} message={message} />)}
    {loading && <div className="typing-state"><span className="typing-icon"><Bot size={17} /></span><span className="typing-copy"><b>Mylo is working</b><small>{loadingText}</small></span><span className="typing-dots"><i /><i /><i /></span></div>}
    <div ref={bottomRef} />
  </div>;
}