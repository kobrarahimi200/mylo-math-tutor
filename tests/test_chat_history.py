import importlib

from fastapi.testclient import TestClient

import api
import chat_history
import cli


def test_saved_chat_survives_module_reload_and_preserves_graphs():
    client = TestClient(api.app)
    first = client.post("/api/chat", json={"message": "Solve 5x² = 676", "level": "advanced"}).json()
    conversation_id = first["conversationId"]
    importlib.reload(chat_history)
    saved = TestClient(api.app).get(f"/api/conversations/{conversation_id}")
    assert saved.status_code == 200
    body = saved.json()
    assert body["level"] == "advanced"
    assert body["messages"][0]["content"] == "Solve 5x² = 676"
    assert body["messages"][1]["content"] == first["answer"]
    assert body["messages"][1]["visualization"] == first["visualizations"][0]
    listing = client.get("/api/conversations").json()
    assert listing[0]["id"] == conversation_id


def test_reopened_conversation_supplies_model_context(monkeypatch):
    client = TestClient(api.app)
    initial = client.post("/api/chat", json={"message": "Explain the quadratic formula"}).json()
    importlib.reload(chat_history)
    second = client.post("/api/chat", json={"message": "What does the discriminant tell me?", "conversationId": initial["conversationId"]}).json()
    assert "two distinct real roots" in second["answer"]
    assert len(chat_history.get_conversation(initial["conversationId"])["messages"]) == 4


def test_level_change_replaces_saved_answer_without_duplicate_question():
    client = TestClient(api.app)
    first = client.post("/api/chat", json={"message": "2 + 2", "level": "beginner"}).json()
    second = client.post("/api/chat", json={"message": "2 + 2", "level": "advanced", "conversationId": first["conversationId"], "replaceLatest": True}).json()
    saved = client.get(f"/api/conversations/{first['conversationId']}").json()
    assert len(saved["messages"]) == 2
    assert saved["messages"][1]["content"] == second["answer"]
    assert saved["level"] == "advanced"


def test_history_keeps_all_turns_but_model_context_is_bounded():
    for index in range(10):
        chat_history.save_exchange("bounded", f"question {index}", {"answer": f"answer {index}"}, "beginner")
    assert len(chat_history.get_conversation("bounded")["messages"]) == 20
    context = chat_history.recent_context("bounded")
    assert len(context) == 12
    assert context[0]["content"] == "question 4"


def test_missing_history_returns_404_and_empty_list():
    client = TestClient(api.app)
    assert client.get("/api/conversations").json() == []
    assert client.get("/api/conversations/missing").status_code == 404


def test_cli_can_list_and_load_saved_chat(capsys):
    chat_history.save_exchange("cli-test", "2 + 2", {"answer": "4"}, "advanced")
    session = {}
    cli.handle_command("/history", session)
    assert "cli-test" in capsys.readouterr().out
    cli.handle_command("/load cli-test", session)
    assert session["conversation_id"] == "cli-test"
    assert session["user_level"] == "advanced"
    assert session["conversation_context"][-1]["content"] == "4"
    cli.handle_command("/reset", session)
    assert chat_history.get_conversation("cli-test") is not None


def test_saved_history_is_readable_from_a_separate_process():
    import subprocess
    import sys
    chat_history.save_exchange("restart-check", "2 + 2", {"answer": "4"}, "beginner")
    result = subprocess.run([sys.executable, "-c", "import chat_history; print(chat_history.get_conversation('restart-check')['messages'][-1]['content'])"], capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == "4"


def test_cli_saves_and_reopens_conversation_across_sessions():
    import subprocess
    import sys
    first = subprocess.run([sys.executable, "cli.py"], input="2 + 2\n/history\n/exit\n", capture_output=True, text=True, check=True, timeout=15)
    assert "Final result" in first.stdout
    saved = chat_history.list_conversations()
    assert len(saved) == 1
    second = subprocess.run([sys.executable, "cli.py"], input=f"/load {saved[0]['id']}\n/exit\n", capture_output=True, text=True, check=True, timeout=15)
    assert "User:\n2 + 2" in second.stdout
    assert "Assistant:" in second.stdout
