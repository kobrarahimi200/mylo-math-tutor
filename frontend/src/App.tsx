import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useState } from "react";
import { ArrowUp, Check, Moon, PanelLeft, Sun, X } from "lucide-react";
import { ChatSidebar, ExplanationLevelSelector, MobileSidebar } from "./components/ChatSidebar";
import { MessageList } from "./components/ChatMessages";
import { sendMessage } from "./lib/api";
import type { Level, Message } from "./types";

const welcome: Message = {
  id: "welcome",
  role: "assistant",
  content: "Hi, I’m Mylo. Bring me a math idea, a tricky equation, or a graph you want to explore.",
};

const suggested = [
  { label: "Explore a derivative", question: "Explain derivatives using y = x²", symbol: "d/dx" },
  { label: "Solve an equation", question: "Solve 2x + 5 = 15", symbol: "x = ?" },
  { label: "See a circle's area", question: "Explain the area of a circle", symbol: "πr²" },
];

function ChatInput({ value, onChange, onSend, disabled }: { value: string; onChange: (value: string) => void; onSend: () => void; disabled: boolean }) {
  const keyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };
  return <form className="composer" onSubmit={(event: FormEvent) => { event.preventDefault(); onSend(); }}>
    <textarea value={value} onChange={(event) => onChange(event.target.value)} onKeyDown={keyDown} placeholder="Ask Mylo anything about math..." rows={1} aria-label="Your math question" disabled={disabled} />
    <div className="composer-footer"><span>Shift + Enter for a new line</span><button className="send-button" type="submit" disabled={!value.trim() || disabled} aria-label="Send question">{disabled ? <span className="button-spinner" /> : <ArrowUp size={18} />}</button></div>
  </form>;
}

function ChatHeader({ dark, onThemeChange, onSettings, mobileProps }: { dark: boolean; onThemeChange: () => void; onSettings: () => void; mobileProps: React.ComponentProps<typeof MobileSidebar> }) {
  return <header className="chat-header"><div className="header-title"><MobileSidebar {...mobileProps} /><span className="header-breadcrumb">Learning space</span><span className="header-divider">/</span><strong>Math conversation</strong></div><div className="header-actions"><span className="live-status"><i /> Tutor ready</span><button className="icon-button" title="Settings" aria-label="Settings" onClick={onSettings}><PanelLeft size={17} /></button><button className="icon-button" title={dark ? "Switch to light mode" : "Switch to dark mode"} aria-label="Toggle color theme" onClick={onThemeChange}>{dark ? <Sun size={17} /> : <Moon size={17} />}</button></div></header>;
}

function SettingsDialog({ open, onClose, level, onLevelChange }: { open: boolean; onClose: () => void; level: Level; onLevelChange: (level: Level) => void }) {
  if (!open) return null;
  return <div className="settings-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title"><div className="settings-title"><div><span className="visual-eyebrow">PREFERENCES</span><h2 id="settings-title">Learning settings</h2></div><button className="icon-button" onClick={onClose} aria-label="Close settings"><X size={17} /></button></div><label className="setting-row"><span><b>Explanation level</b><small>Adjust how Mylo walks through each idea</small></span><ExplanationLevelSelector level={level} onChange={onLevelChange} /></label><div className="setting-row"><span><b>Conversation memory</b><small>Recent turns stay available for follow-up questions</small></span><span className="setting-on"><Check size={13} /> On</span></div></section></div>;
}

export default function App() {
  const [level, setLevel] = useState<Level>(() => (localStorage.getItem("mylo-level") as Level) || "beginner");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([welcome]);
  const [conversationId, setConversationId] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dark, setDark] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const updateLevel = (next: Level) => { setLevel(next); localStorage.setItem("mylo-level", next); };
  const newChat = () => { setMessages([welcome]); setConversationId(undefined); setError(""); setDraft(""); setDrawerOpen(false); };

  useEffect(() => {
    const handleShortcut = (event: globalThis.KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        newChat();
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, []);

  const ask = async (question = draft) => {
    const text = question.trim();
    if (!text || loading) return;
    setError("");
    setDraft("");
    setMessages((previous) => [...previous, { id: crypto.randomUUID(), role: "user", content: text }]);
    setLoading(true);
    try {
      const answer = await sendMessage(text, level, conversationId);
      setConversationId(answer.conversationId);
      setMessages((previous) => [...previous, {
        id: crypto.randomUUID(), role: "assistant", content: answer.answer,
        language: answer.metadata.language, visualization: answer.visualizations[0], metadata: answer.metadata,
      }]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The tutor could not be reached. Check that the API is running.");
      setDraft(text);
    } finally {
      setLoading(false);
    }
  };

  const chooseExample = (question: string) => { setDraft(question); setDrawerOpen(false); };
  const mobileProps = { open: drawerOpen, onOpenChange: setDrawerOpen, level, onLevelChange: updateLevel, onNewChat: newChat, onExample: chooseExample, onSettings: () => { setDrawerOpen(false); setSettingsOpen(true); } };
  return (
    <div className={`app-shell ${dark ? "dark-mode" : ""}`}>
      <ChatSidebar level={level} onLevelChange={updateLevel} onNewChat={newChat} onExample={chooseExample} onSettings={() => setSettingsOpen(true)} />
      <main className="workspace">
        <ChatHeader dark={dark} onThemeChange={() => setDark((value) => !value)} onSettings={() => setSettingsOpen(true)} mobileProps={mobileProps} />
        <section className={`conversation ${messages.length === 1 ? "empty-conversation" : ""}`}>
          {messages.length === 1 ? <div className="welcome-stage"><div className="welcome-orbit"><span className="orbit-ring ring-one" /><span className="orbit-ring ring-two" /><span className="orbit-center">∑</span><span className="orbit-spark spark-one">+</span><span className="orbit-spark spark-two">π</span><span className="orbit-spark spark-three">x²</span></div><span className="welcome-kicker">YOUR PERSONAL MATH STUDIO</span><h1>Good questions<br /><em>change everything.</em></h1><p>Explore the shape of an idea, work through a problem, and see the mathematics come alive.</p><div className="suggested-prompts">{suggested.map((prompt) => <button key={prompt.label} className="suggestion" onClick={() => void ask(prompt.question)}><span className="suggestion-symbol">{prompt.symbol}</span><span>{prompt.label}</span><ArrowUp size={14} className="suggestion-arrow" /></button>)}</div><div className="welcome-note"><span className="online-pulse" /> {welcome.content}</div></div> : <MessageList messages={messages} loading={loading} />}
          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={15} /></button></div>}
        </section>
        <div className="composer-dock"><ChatInput value={draft} onChange={setDraft} onSend={() => void ask()} disabled={loading} /><p className="composer-disclaimer">Mylo can make mistakes. Check important results and steps.</p></div>
        <div className="workspace-footer"><span><i className="footer-dot" /> LangGraph math engine</span><span>Built for curious minds <b>·</b> English + فارسی</span></div>
      </main>
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} level={level} onLevelChange={updateLevel} />
    </div>
  );
}