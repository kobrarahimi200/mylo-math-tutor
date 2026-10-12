import * as Dialog from "@radix-ui/react-dialog";
import * as Select from "@radix-ui/react-select";
import { BookOpen, Check, ChevronDown, Command, Menu, MessageSquarePlus, Settings2, Sparkles, X } from "lucide-react";
import type { Level } from "../types";

const examples = ["Explain derivatives using y = x²", "Solve 2x + 5 = 15", "Explore the area of a circle", "Show how a changes y = ax²"];

export function ExplanationLevelSelector({ level, onChange }: { level: Level; onChange: (level: Level) => void }) {
  return (
    <Select.Root value={level} onValueChange={(value) => onChange(value as Level)}>
      <Select.Trigger className="level-select" aria-label="Explanation level">
        <span>{level}</span><ChevronDown size={15} />
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="select-menu" position="popper" sideOffset={6}>
          <Select.Viewport>
            {(["beginner", "intermediate", "advanced"] as Level[]).map((option) => (
              <Select.Item className="select-option" value={option} key={option}>
                <Select.ItemText>{option}</Select.ItemText><Select.ItemIndicator><Check size={14} /></Select.ItemIndicator>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}

function SidebarContents({ level, onLevelChange, onNewChat, onExample, onSettings }: {
  level: Level; onLevelChange: (level: Level) => void; onNewChat: () => void;
  onExample: (question: string) => void; onSettings: () => void;
}) {
  return (
    <>
      <a className="brand" href="#home" aria-label="Mylo home">
        <span className="brand-mark"><Sparkles size={18} strokeWidth={2.2} /></span>
        <span className="brand-name">mylo<span>.</span></span>
        <span className="brand-caption">MATH STUDIO</span>
      </a>
      <button className="new-chat-button" onClick={onNewChat}><MessageSquarePlus size={17} />New chat<span className="shortcut">⌘ K</span></button>
      <div className="sidebar-section">
        <div className="section-label">WORKSPACE</div>
        <button className="conversation-link active"><span className="conversation-dot" />Current conversation</button>
      </div>
      <div className="sidebar-section level-section">
        <div className="section-label">EXPLANATION LEVEL</div>
        <ExplanationLevelSelector level={level} onChange={onLevelChange} />
      </div>
      <div className="sidebar-section examples-section">
        <div className="section-label"><BookOpen size={13} /> TRY AN EXAMPLE</div>
        {examples.map((example) => <button key={example} className="example-link" onClick={() => onExample(example)}>{example}</button>)}
      </div>
      <div className="sidebar-bottom">
        <button className="settings-button" onClick={onSettings}><Settings2 size={17} />Settings</button>
        <div className="sidebar-profile"><span className="avatar">M</span><span><b>Mylo Tutor</b><small>Mathematics, made clear</small></span><Command size={15} className="profile-mark" /></div>
      </div>
    </>
  );
}

export function ChatSidebar(props: {
  level: Level; onLevelChange: (level: Level) => void; onNewChat: () => void;
  onExample: (question: string) => void; onSettings: () => void;
}) {
  return <aside className="sidebar"><SidebarContents {...props} /></aside>;
}

export function MobileSidebar(props: {
  open: boolean; onOpenChange: (open: boolean) => void; level: Level; onLevelChange: (level: Level) => void;
  onNewChat: () => void; onExample: (question: string) => void; onSettings: () => void;
}) {
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