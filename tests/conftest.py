import pytest


@pytest.fixture(autouse=True)
def offline_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")


@pytest.fixture(autouse=True)
def isolated_chat_history(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAT_HISTORY_DB", str(tmp_path / "history.sqlite3"))
