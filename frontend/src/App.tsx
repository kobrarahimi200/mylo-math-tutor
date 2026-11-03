import { FormEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowUp, Check, Moon, PanelLeft, Sigma, Sun, X } from "lucide-react";
import { ChatSidebar, ExplanationLevelSelector, MobileSidebar } from "./components/ChatSidebar";
import { MessageList } from "./components/ChatMessages";
import { ProviderSettings } from "./components/ProviderSettings";
import { sendMessage, listConversations, getConversation } from "./lib/api";
import { starterExamples } from "./lib/examples";
import type { Level, Message, ConversationSummary } from "./types";

const welcome: Message = {
  id: "welcome",
  role: "assistant",
  content: "Hi, I’m Mylo. Bring me a math idea, a tricky equation, or a graph you want to explore.",
};

const suggested = starterExamples;

const formulaGroups = [
  {
    category: "Algebra & numbers",
    formulas: [
      { label: "Quadratic formula", preview: "x = (−b ± √(b²−4ac)) / 2a", latex: "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}" },
      { label: "Binomial theorem", preview: "(x+y)ⁿ = Σ C(n,k)xⁿ⁻ᵏyᵏ", latex: "(x+y)^n = \\sum_{k=0}^{n} \\binom{n}{k}x^{n-k}y^k" },
      { label: "Difference of squares", preview: "a² − b² = (a−b)(a+b)", latex: "a^2 - b^2 = (a-b)(a+b)" },
      { label: "Arithmetic series", preview: "Sₙ = n(a₁+aₙ) / 2", latex: "S_n = \\frac{n(a_1 + a_n)}{2}" },
      { label: "Geometric series", preview: "Sₙ = a(1−rⁿ) / (1−r)", latex: "S_n = a \\frac{1-r^n}{1-r}" },
      { label: "Change of log base", preview: "logᵦ(x) = ln(x) / ln(b)", latex: "\\log_b(x) = \\frac{\\ln(x)}{\\ln(b)}" },
      { label: "Euler's identity", preview: "eⁱᵖ + 1 = 0", latex: "e^{i\\pi} + 1 = 0" },
    ],
  },
  {
    category: "Geometry & trigonometry",
    formulas: [
      { label: "Pythagorean theorem", preview: "a² + b² = c²", latex: "a^2 + b^2 = c^2" },
      { label: "Law of cosines", preview: "c² = a²+b²−2ab cos(C)", latex: "c^2 = a^2 + b^2 - 2ab\\cos(C)" },
      { label: "Law of sines", preview: "a/sin(A) = b/sin(B) = c/sin(C)", latex: "\\frac{a}{\\sin(A)} = \\frac{b}{\\sin(B)} = \\frac{c}{\\sin(C)}" },
      { label: "Circle area", preview: "A = πr²", latex: "A = \\pi r^2" },
      { label: "Sphere volume", preview: "V = 4πr³ / 3", latex: "V = \\frac{4}{3}\\pi r^3" },
      { label: "Heron's formula", preview: "A = √(s(s−a)(s−b)(s−c))", latex: "A = \\sqrt{s(s-a)(s-b)(s-c)}" },
    ],
  },
  {
    category: "Calculus & analysis",
    formulas: [
      { label: "Chain rule", preview: "(f(g(x)))′ = f′(g(x))g′(x)", latex: "\\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)" },
      { label: "Product rule", preview: "(uv)′ = u′v + uv′", latex: "\\frac{d}{dx}(uv) = u'v + uv'" },
      { label: "Quotient rule", preview: "(u/v)′ = (u′v−uv′) / v²", latex: "\\frac{d}{dx}\\left(\\frac{u}{v}\\right) = \\frac{u'v-uv'}{v^2}" },
      { label: "Integration by parts", preview: "∫u dv = uv − ∫v du", latex: "\\int u \\, dv = uv - \\int v \\, du" },
      { label: "Fundamental theorem", preview: "∫ₐᵇ f′(x)dx = f(b)−f(a)", latex: "\\int_a^b f'(x)\\,dx = f(b)-f(a)" },
      { label: "Taylor series", preview: "f(x) = Σ f⁽ⁿ⁾(a)(x−a)ⁿ/n!", latex: "f(x) = \\sum_{n=0}^{\\infty} \\frac{f^{(n)}(a)}{n!}(x-a)^n" },
    ],
  },
  {
    category: "Probability & transforms",
    formulas: [
      { label: "Bayes' theorem", preview: "P(A|B) = P(B|A)P(A) / P(B)", latex: "P(A \\mid B) = \\frac{P(B \\mid A)P(A)}{P(B)}" },
      { label: "Normal distribution", preview: "f(x) = e⁻⁽ˣ⁻ᵘ⁾²/²σ² / (σ√2π)", latex: "f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}" },
      { label: "Gaussian integral", preview: "∫₋∞∞ e⁻ˣ² dx = √π", latex: "\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx = \\sqrt{\\pi}" },
      { label: "Fourier transform", preview: "F(ω) = ∫ f(t)e⁻ⁱωᵗ dt", latex: "F(\\omega) = \\int_{-\\infty}^{\\infty} f(t)e^{-i\\omega t}\\,dt" },
      { label: "Fourier series", preview: "f(x) = a₀/2 + Σ(aₙ cos nx + bₙ sin nx)", latex: "f(x) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty}(a_n\\cos(nx)+b_n\\sin(nx))" },
      { label: "Stirling's approximation", preview: "n! ≈ √(2πn)(n/e)ⁿ", latex: "n! \\approx \\sqrt{2\\pi n}\\left(\\frac{n}{e}\\right)^n" },
    ],
  },
];

function ChatInput({ value, onChange, onSend, disabled }: { value: string; onChange: (value: string) => void; onSend: () => void; disabled: boolean }) {
  const [formulaOpen, setFormulaOpen] = useState(false);
  const [formula, setFormula] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const selectionRef = useRef<{ start: number; end: number } | null>(null);
  const pendingCaret = useRef<number | null>(null);
  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(Math.max(input.scrollHeight, 44), 220)}px`;
    if (pendingCaret.current !== null && !formulaOpen) {
      const caret = pendingCaret.current;
      pendingCaret.current = null;
      input.focus({ preventScroll: true });
      input.setSelectionRange(caret, caret);
      selectionRef.current = { start: caret, end: caret };
      if (caret === value.length) input.scrollTop = input.scrollHeight;
      input.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [value, formulaOpen]);

  const keyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };
  const appendFormula = (expression: string, label = "formula") => {
    if (disabled) return;
    // Pasted TeX may already include display or inline delimiters.
    expression = expression.trim();
    for (const [open, close] of [["$$", "$$"], ["\\[", "\\]"], ["\\(", "\\)"], ["$", "$"]]) {
      if (expression.startsWith(open) && expression.endsWith(close)) {
        expression = expression.slice(open.length, -close.length).trim();
        break;
      }
    }
    if (!expression) return;
    const selection = selectionRef.current;
    const start = Math.min(selection?.start ?? value.length, value.length);
    const end = Math.min(selection?.end ?? value.length, value.length);
    const before = value.slice(0, start).trimEnd();
    const after = value.slice(end);
    const heading = !value.trim() ? `Explain the ${label.toLowerCase()}:` : label === "formula" ? "" : `${label}:`;
    const inserted = `${before ? `${before}\n\n` : ""}${heading ? `${heading}\n\n` : ""}$$\n${expression}\n$$`;
    const next = inserted + (after ? `\n\n${after}` : "");
    pendingCaret.current = inserted.length;
    onChange(next);
    setFormula("");
    setFormulaOpen(false);
  };
  const insertFormula = () => {
    const expression = formula.trim();
    if (!expression) return;
    appendFormula(expression);
  };
  return <form className="composer" onSubmit={(event: FormEvent) => { event.preventDefault(); onSend(); }}>
    {formulaOpen && <div className="formula-editor"><div className="formula-presets" aria-label="Popular formulas">{formulaGroups.map((group) => <section className="formula-group" key={group.category}><h3>{group.category}</h3><div className="formula-preset-grid">{group.formulas.map((preset) => <button className="formula-preset" type="button" key={preset.label} aria-label={`Insert ${preset.label}`} onClick={() => appendFormula(preset.latex, preset.label)} disabled={disabled}><span>{preset.label}</span><small>{preset.preview}</small></button>)}</div></section>)}</div><div className="formula-manual"><label htmlFor="formula-input">LaTeX formula</label><textarea id="formula-input" value={formula} onChange={(event) => setFormula(event.target.value)} placeholder="\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}" rows={2} disabled={disabled} /><button type="button" onClick={insertFormula} disabled={!formula.trim() || disabled}>Insert formula</button></div></div>}
    <textarea ref={inputRef} value={value} onChange={(event) => {
      selectionRef.current = { start: event.target.selectionStart, end: event.target.selectionEnd };
      onChange(event.target.value);
    }} onSelect={(event) => {
      selectionRef.current = { start: event.currentTarget.selectionStart, end: event.currentTarget.selectionEnd };
    }} onKeyDown={keyDown} placeholder="Ask Mylo anything about math..." rows={1} aria-label="Your math question" disabled={disabled} />
    <div className="composer-footer"><span>Shift + Enter for a new line</span><div className="composer-actions"><button className="formula-toggle" type="button" onClick={() => {
      if (!formulaOpen && inputRef.current) {
        selectionRef.current = { start: inputRef.current.selectionStart, end: inputRef.current.selectionEnd };
      }
      setFormulaOpen((open) => !open);
    }} aria-expanded={formulaOpen}><Sigma size={15} /> Formula</button><button className="send-button" type="submit" disabled={!value.trim() || disabled} aria-label="Send question">{disabled ? <span className="button-spinner" /> : <ArrowUp size={18} />}</button></div></div>
  </form>;
}

function ChatHeader({ dark, onThemeChange, onSettings, mobileProps }: { dark: boolean; onThemeChange: () => void; onSettings: () => void; mobileProps: React.ComponentProps<typeof MobileSidebar> }) {
  return <header className="chat-header"><div className="header-title"><MobileSidebar {...mobileProps} /><span className="header-breadcrumb">Learning space</span><span className="header-divider">/</span><strong>Math conversation</strong></div><div className="header-actions"><span className="live-status"><i /> Tutor ready</span><button className="icon-button" title="Settings" aria-label="Settings" onClick={onSettings}><PanelLeft size={17} /></button><button className="icon-button" title={dark ? "Switch to light mode" : "Switch to dark mode"} aria-label="Toggle color theme" onClick={onThemeChange}>{dark ? <Sun size={17} /> : <Moon size={17} />}</button></div></header>;
}

function SettingsDialog({ open, onClose, level, onLevelChange }: { open: boolean; onClose: () => void; level: Level; onLevelChange: (level: Level) => void }) {
  if (!open) return null;
  return <div className="settings-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title"><div className="settings-title"><div><span className="visual-eyebrow">PREFERENCES</span><h2 id="settings-title">Learning settings</h2></div><button className="icon-button" onClick={onClose} aria-label="Close settings"><X size={17} /></button></div><label className="setting-row"><span><b>Explanation level</b><small>Adjust how Mylo walks through each idea</small></span><ExplanationLevelSelector level={level} onChange={onLevelChange} /></label><ProviderSettings /><div className="setting-row"><span><b>Conversation memory</b><small>Recent turns stay available for follow-up questions</small></span><span className="setting-on"><Check size={13} /> On</span></div></section></div>;
}

export default function App() {
  const [level, setLevel] = useState<Level>(() => (localStorage.getItem("mylo-level") as Level) || "beginner");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([welcome]);
  const [conversationId, setConversationId] = useState<string>();
  const [savedChats, setSavedChats] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dark, setDark] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const requestVersion = useRef(0);
  const [loadingText, setLoadingText] = useState("Following the math through, step by step");

  const requestAnswer = async (text: string, requestedLevel: Level, replaceId?: string, restoreDraft = false) => {
    const version = ++requestVersion.current;
    setError("");
    setLoading(true);
    setLoadingText(restoreDraft ? "Following the math through, step by step" : `Updating the answer for ${requestedLevel} level…`);
    try {
      const answer = replaceId
        ? await sendMessage(text, requestedLevel, conversationId, true)
        : await sendMessage(text, requestedLevel, conversationId);
      if (version !== requestVersion.current) return;
      setConversationId(answer.conversationId);
      localStorage.setItem("mylo-active-conversation", answer.conversationId);
      const message: Message = {
        id: replaceId ?? crypto.randomUUID(), role: "assistant", content: answer.answer,
        language: answer.metadata.language, visualization: answer.visualizations[0], metadata: answer.metadata,
      };
      setMessages((previous) => replaceId
        ? previous.map((item) => item.id === replaceId ? message : item)
        : [...previous, message]);
      try {
        const saved = await listConversations();
        if (version === requestVersion.current) setSavedChats(saved);
      } catch {
        if (version === requestVersion.current) setError("Your answer is ready, but the saved chat list could not be refreshed.");
      }
    } catch (caught) {
      if (version !== requestVersion.current) return;
      const detail = caught instanceof Error ? caught.message : "The tutor could not be reached. Check that the API is running.";
      setError(replaceId ? `Could not update the explanation. Your previous answer is still shown. ${detail}` : detail);
      if (restoreDraft) setDraft(text);
    } finally {
      if (version === requestVersion.current) setLoading(false);
    }
  };

  const updateLevel = (next: Level) => {
    if (next === level) return;
    setLevel(next);
    localStorage.setItem("mylo-level", next);
    const questionIndex = messages.map((message) => message.role).lastIndexOf("user");
    if (questionIndex === -1) {
      ++requestVersion.current;
      return;
    }
    const previousAnswer = messages.slice(questionIndex + 1).find((message) => message.role === "assistant");
    void requestAnswer(messages[questionIndex].content, next, previousAnswer?.id);
  };
  const newChat = () => {
    ++requestVersion.current;
    setLoading(false);
    localStorage.removeItem("mylo-active-conversation");
    setMessages([welcome]); setConversationId(undefined); setError(""); setDraft(""); setDrawerOpen(false);
  };

  const openConversation = async (id: string) => {
    const version = ++requestVersion.current;
    setLoading(true);
    setLoadingText("Opening saved conversation…");
    setError("");
    setDrawerOpen(false);
    try {
      const saved = await getConversation(id);
      if (version !== requestVersion.current) return;
      setMessages([welcome, ...saved.messages]);
      setConversationId(saved.id);
      setLevel(saved.level);
      setDraft("");
      localStorage.setItem("mylo-active-conversation", saved.id);
      localStorage.setItem("mylo-level", saved.level);
    } catch (caught) {
      if (version === requestVersion.current) setError(caught instanceof Error ? caught.message : "Could not open saved chat.");
    } finally {
      if (version === requestVersion.current) setLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    const version = requestVersion.current;
    const activeId = localStorage.getItem("mylo-active-conversation");
    void (async () => {
      try {
        const saved = await listConversations();
        if (cancelled || version !== requestVersion.current) return;
        setSavedChats(saved);
        if (!activeId || version !== requestVersion.current || !saved.some((chat) => chat.id === activeId)) return;
        const active = await getConversation(activeId);
        if (cancelled || version !== requestVersion.current) return;
        setMessages([welcome, ...active.messages]);
        setConversationId(active.id);
        setLevel(active.level);
      } catch (caught) {
        if (!cancelled && version === requestVersion.current) setError(caught instanceof Error ? caught.message : "Could not load saved chats.");
      }
    })();
    return () => { cancelled = true; };
  }, []);

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
    setDraft("");
    setMessages((previous) => [...previous, { id: crypto.randomUUID(), role: "user", content: text }]);
    await requestAnswer(text, level, undefined, true);
  };

  const chooseExample = (question: string) => { setDraft(question); setDrawerOpen(false); };
  const historyProps = { conversations: savedChats, activeConversationId: conversationId, onOpenConversation: (id: string) => void openConversation(id) };
  const mobileProps = { ...historyProps, open: drawerOpen, onOpenChange: setDrawerOpen, level, onLevelChange: updateLevel, onNewChat: newChat, onExample: chooseExample, onSettings: () => { setDrawerOpen(false); setSettingsOpen(true); } };
  return (
    <div className={`app-shell ${dark ? "dark-mode" : ""}`}>
      <ChatSidebar {...historyProps} level={level} onLevelChange={updateLevel} onNewChat={newChat} onExample={chooseExample} onSettings={() => setSettingsOpen(true)} />
      <main className="workspace">
        <ChatHeader dark={dark} onThemeChange={() => setDark((value) => !value)} onSettings={() => setSettingsOpen(true)} mobileProps={mobileProps} />
        <section className={`conversation ${messages.length === 1 ? "empty-conversation" : ""}`}>
          {messages.length === 1 ? <div className="welcome-stage"><div className="welcome-orbit"><span className="orbit-ring ring-one" /><span className="orbit-ring ring-two" /><span className="orbit-center">∑</span><span className="orbit-spark spark-one">+</span><span className="orbit-spark spark-two">π</span><span className="orbit-spark spark-three">x²</span></div><span className="welcome-kicker">YOUR PERSONAL MATH STUDIO</span><h1>Good questions<br /><em>change everything.</em></h1><p>Explore the shape of an idea, work through a problem, and see the mathematics come alive.</p><div className="suggested-prompts">{suggested.map((prompt) => <button key={prompt.label} className="suggestion" onClick={() => void ask(prompt.question)}><span className="suggestion-symbol">{prompt.symbol}</span><span>{prompt.label}</span><ArrowUp size={14} className="suggestion-arrow" /></button>)}</div><div className="welcome-note"><span className="online-pulse" /> {welcome.content}</div></div> : <MessageList messages={messages} loading={loading} loadingText={loadingText} />}
          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={15} /></button></div>}
        </section>
        <div className="composer-dock"><ChatInput value={draft} onChange={setDraft} onSend={() => void ask()} disabled={loading} /><p className="composer-disclaimer">Mylo can make mistakes. Check important results and steps.</p></div>
        <div className="workspace-footer"><span><i className="footer-dot" /> LangGraph math engine</span><span>Built for curious minds <b>·</b> English + فارسی</span></div>
      </main>
      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} level={level} onLevelChange={updateLevel} />
    </div>
  );
}