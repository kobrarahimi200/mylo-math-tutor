from __future__ import annotations

import sys
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

        state = create_initial_state(user_input, session.get("user_level", "beginner"))
        graph = build_graph()
        result = graph.invoke(state)
        answer = result.get("final_answer") or "I could not produce an answer."

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
