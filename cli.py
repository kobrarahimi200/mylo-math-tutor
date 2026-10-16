from __future__ import annotations

import sys
from uuid import uuid4
import chat_history
from typing import Any

from agent import build_graph, create_initial_state, is_persian


BANNER = """====================================
       MYLO MATH TUTOR
====================================

Your AI Mathematics Tutor

Commands:
  /help
  /level beginner
  /level intermediate
  /level advanced
  /debug on
  /debug off
  /reset
  /history
  /load <conversation-id>
  /exit

Ask your math question:
> """


def print_help() -> None:
    print(
        """
Commands:
  /help          Show available commands
  /level <level> Set explanation level: beginner | intermediate | advanced
  /debug on      Show graph execution trace
  /debug off     Hide graph execution trace
  /reset         Clear the current state and start a new session
  /history       List saved conversations
  /load <id>     Reopen a saved conversation
  /exit          Exit the tutor
        """.strip()
    )


def handle_command(command: str, session: dict[str, Any]) -> bool:
    cmd = command.strip()
    if not cmd:
        print("Please ask a math question or use a command.")
        return False

    if cmd == "/help":
        print_help()
        return False

    if cmd == "/history":
        saved = chat_history.list_conversations(20)
        for conversation in saved:
            print(f"{conversation['id']}  {conversation['title']}")
        if not saved:
            print("No saved conversations yet.")
        return False

    if cmd.startswith("/load "):
        conversation_id = cmd.split(maxsplit=1)[1].strip()
        saved = chat_history.get_conversation(conversation_id)
        if saved is None:
            print("Conversation not found. Use /history to see saved chats.")
            return False
        session["conversation_id"] = conversation_id
        session["conversation_context"] = chat_history.recent_context(conversation_id)
        session["user_level"] = saved["level"]
        for message in saved["messages"]:
            print(f"\n{message['role'].capitalize()}:\n{message['content']}")
        return False

    if cmd == "/exit":
        print("Goodbye!")
        return True

    if cmd.startswith("/level "):
        level = cmd.split(maxsplit=1)[1].lower()
        if level not in {"beginner", "intermediate", "advanced"}:
            print("Unsupported level. Choose beginner, intermediate, or advanced.")
            return False
        session["user_level"] = level
        print(f"Explanation level set to: {level}")
        return False

    if cmd == "/debug on":
        session["debug"] = True
        print("Debug mode enabled.")
        return False

    if cmd == "/debug off":
        session["debug"] = False
        print("Debug mode disabled.")
        return False

    if cmd == "/reset":
        session.clear()
        session["user_level"] = "beginner"
        session["debug"] = False
        print("Session reset.")
        return False

    print("Unknown command. Type /help to see available commands.")
    return False


def debug_summary(result: dict[str, Any]) -> None:
    state_updates = ", ".join(result.get("debug_trace", [])[-5:]) if result.get("debug_trace") else "none"
    print("\n[DEBUG]")
    print(f"Current node: {result.get('current_node', 'unknown')}")
    print(f"Selected branch: {result.get('selected_branch', 'unknown')}")
    print(f"State updates: {state_updates}")
    print(f"Verification status: {result.get('verification_status', 'unverified')}")
    print(f"Retry count: {result.get('retry_count', 0)}")


def run_session() -> None:
    session: dict[str, Any] = {"user_level": "beginner", "debug": False}
    print(BANNER)

    while True:
        try:
            user_input = input("> ")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return
        except EOFError:
            print("\nGoodbye!")
            return

        if not user_input.strip():
            print("Please ask a math question or use a command.")
            continue

        if user_input.startswith("/"):
            if handle_command(user_input, session):
                return
            continue

        state = create_initial_state(user_input, session.get("user_level", "beginner"), session.get("conversation_context", []))
        graph = build_graph()
        result = graph.invoke(state)
        answer = result.get("final_answer") or "I could not produce an answer."
        session["conversation_context"] = (session.get("conversation_context", []) + [
            {"role": "user", "content": user_input}, {"role": "assistant", "content": answer},
        ])[-12:]
        conversation_id = session.setdefault("conversation_id", str(uuid4()))
        spec = result.get("visualization_spec", {"type": "none"})
        chat_history.save_exchange(conversation_id, user_input, {
            "answer": answer,
            "visualizations": [] if spec.get("type") == "none" else [spec],
            "steps": result.get("solution_steps", []),
            "metadata": {"conversationMemory": True, "language": result.get("language", "en"), "intent": result.get("intent", "CLARIFY"), "verification": result.get("verification_status", "unverified"), "explanationLevel": session.get("user_level", "beginner"), "llm": result.get("llm_status", {})},
        }, session.get("user_level", "beginner"))
        status = result.get("llm_status", {})
        print(f"Provider: {status.get('provider', 'offline')} | Model: {status.get('model') or 'local math'}")
        if status.get("error"):
            print(status["error"])

        if session.get("debug"):
            debug_summary(result)

        prefix = "Mylo:" if not is_persian(answer) else "Mylo:"
        print(f"\n{prefix}\n{answer}\n")


def main() -> None:
    try:
        run_session()
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        print("Please check the input or configuration and try again.")


if __name__ == "__main__":
    main()
