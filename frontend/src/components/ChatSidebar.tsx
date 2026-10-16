import * as Dialog from "@radix-ui/react-dialog";
import * as Select from "@radix-ui/react-select";
import { BookOpen, Check, ChevronDown, Command, Menu, MessageSquarePlus, Settings2, Sparkles, X } from "lucide-react";
import { starterExamples } from "../lib/examples";
import type { Level, ConversationSummary } from "../types";



const levelDescriptions: Record<Level, string> = {
  beginner: "Every operation explained in plain language.",
  intermediate: "Key steps, method, and a result check.",
  advanced: "Formal reasoning, assumptions, and edge cases.",
};

export function ExplanationLevelSelector({ level, onChange }: { level: Level; onChange: (level: Level) => void }) {
  return (
    <div className="level-control"><Select.Root value={level} onValueChange={(value) => onChange(value as Level)}>
      <Select.Trigger className="level-select" aria-label="Explanation level">
        <span>{level}</span><ChevronDown size={15} />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="select-menu" position="popper" sideOffset={6}>
          <Select.Viewport>
            {(["beginner", "intermediate", "advanced"] as Level[]).map((option) => (
              <Select.Item className="select-option" value={option} key={option}>
                <span><Select.ItemText>{option}</Select.ItemText><small className="level-option-description">{levelDescriptions[option]}</small></span><Select.ItemIndicator><Check size={14} /></Select.ItemIndicator>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root><p className="level-description">{levelDescriptions[level]}<span>Changing level updates your latest answer.</span></p></div>
  );
}

type SidebarProps = {
  level: Level; onLevelChange: (level: Level) => void; onNewChat: () => void;
  onExample: (question: string) => void; onSettings: () => void;
  conversations?: ConversationSummary[]; activeConversationId?: string; onOpenConversation?: (id: string) => void;
};

function SidebarContents({ level, onLevelChange, onNewChat, onExample, onSettings, conversations = [], activeConversationId, onOpenConversation }: SidebarProps) {
  return (
    <>
      <a className="brand" href="#home" aria-label="Mylo home">
        <span className="brand-mark"><Sparkles size={18} strokeWidth={2.2} /></span>
        <span className="brand-name">mylo<span>.</span></span>
        <span className="brand-caption">MATH STUDIO</span>
      </a>
      <button className="new-chat-button" onClick={onNewChat}><MessageSquarePlus size={17} />New chat<span className="shortcut">⌘ K</span></button>
      <div className="sidebar-section">
        <div className="section-label">SAVED CHATS</div>
        <div className="saved-chat-list">
          {conversations.length === 0 && <p className="history-empty">Your conversations will be saved here.</p>}
          {conversations.map((chat) => <button key={chat.id} className={`conversation-link ${chat.id === activeConversationId ? "active" : ""}`} aria-current={chat.id === activeConversationId ? "page" : undefined} title={chat.title} onClick={() => onOpenConversation?.(chat.id)}><span className="conversation-dot" /><span className="saved-chat-title">{chat.title}</span></button>)}
        </div>
      </div>
      <div className="sidebar-section level-section">
        <div className="section-label">EXPLANATION LEVEL</div>
        <ExplanationLevelSelector level={level} onChange={onLevelChange} />
      </div>
      <div className="sidebar-section examples-section">
        <div className="section-label"><BookOpen size={13} /> TRY AN EXAMPLE</div>
        {starterExamples.map((example) => <button key={example.label} className="example-link" onClick={() => onExample(example.question)}><span className="example-symbol" aria-hidden="true">{example.symbol}</span>{example.sidebar}</button>)}
      </div>
      <div className="sidebar-bottom">
        <button className="settings-button" onClick={onSettings}><Settings2 size={17} />Settings</button>
        <div className="sidebar-profile"><span className="avatar">M</span><span><b>Mylo Tutor</b><small>Mathematics, made clear</small></span><Command size={15} className="profile-mark" /></div>
      </div>
    </>
  );
}

export function ChatSidebar(props: SidebarProps) {
  return <aside className="sidebar"><SidebarContents {...props} /></aside>;
}

export function MobileSidebar(props: SidebarProps & { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Trigger asChild><button className="icon-button mobile-menu" aria-label="Open menu"><Menu size={19} /></button></Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content className="mobile-drawer">
          <Dialog.Title className="sr-only">Navigation</Dialog.Title>
          <Dialog.Close asChild><button className="icon-button drawer-close" aria-label="Close menu"><X size={18} /></button></Dialog.Close>
          <SidebarContents {...props} />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}