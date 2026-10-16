"""Provider configuration and request-local status shared by API, CLI, and graph."""
from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class LLMRuntime:
    model: BaseChatModel | None = None
    status: dict[str, Any] = field(default_factory=dict)


_runtime: ContextVar[LLMRuntime | None] = ContextVar("mylo_llm_runtime", default=None)


def create_llm_runtime() -> LLMRuntime:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    runtime = LLMRuntime(status={"configuredProvider": provider, "provider": "offline", "model": "", "available": False, "used": False, "error": ""})
    try:
        if provider == "offline":
            return runtime
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            model = os.getenv("OLLAMA_MODEL", "qwen3:4b").strip()
            runtime.status["model"] = model
            try:
                response = httpx.get(f"{base_url}/api/tags", timeout=3, trust_env=False)
                response.raise_for_status()
                names = {entry.get("name", entry.get("model", "")) for entry in response.json()["models"]}
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                runtime.status["error"] = f"Cannot reach Ollama at {base_url}. Start Ollama with 'ollama serve' and check OLLAMA_BASE_URL. Offline math is still available."
                return runtime
            if model not in names and (":" in model or f"{model}:latest" not in names):
                runtime.status["error"] = f"Ollama model '{model}' is not installed. Run 'ollama pull {model}' on the Ollama server, then retry."
                return runtime
            runtime.model = ChatOllama(model=model, base_url=base_url, temperature=0.2, reasoning=False, client_kwargs={"timeout": 120, "trust_env": False})
        elif provider == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            runtime.status["model"] = model
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                runtime.status["error"] = "OpenAI is selected but OPENAI_API_KEY is missing. Set the key or select LLM_PROVIDER=ollama."
                return runtime
            runtime.model = ChatOpenAI(model=model, api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None, temperature=0.2, timeout=120, max_retries=1)
        else:
            runtime.status["error"] = f"Unknown LLM_PROVIDER '{provider}'. Choose ollama, openai, or offline."
            return runtime
        runtime.status.update(provider=provider, available=True)
    except Exception:
        runtime.status["error"] = f"Could not initialize {provider}. Check provider configuration and installed dependencies."
    return runtime


def begin_llm_request() -> LLMRuntime:
    runtime = create_llm_runtime()
    _runtime.set(runtime)
    return runtime


def activate_runtime(runtime: LLMRuntime | None) -> None:
    if runtime is not None:
        _runtime.set(runtime)


def current_runtime() -> LLMRuntime:
    runtime = _runtime.get()
    if runtime is None:
        runtime = create_llm_runtime()
        _runtime.set(runtime)
    return runtime


def get_llm() -> BaseChatModel | None:
    return current_runtime().model


def llm_status() -> dict[str, Any]:
    return dict(current_runtime().status)


def record_llm_success() -> None:
    current_runtime().status["used"] = True


def record_llm_failure() -> None:
    runtime = current_runtime()
    provider = runtime.status["configuredProvider"]
    runtime.status.update(provider="offline", available=False, used=False, error=f"{provider.capitalize()} inference failed or returned an empty response. Check the server, model, and credentials, then retry. Using offline math fallback.")
    runtime.model = None
