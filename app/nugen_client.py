"""Thin client for the documented Nugen chat completions endpoint.

Endpoint and schema follow https://docs.nugen.in/api-reference/inference/generate-chat-completions:
POST /api/v3/inference/chat/completions with `model`, `messages`, `max_tokens`,
`temperature`, `prompt_truncate_len`, `stream`. Non-streaming responses return
`choices[0].message.content`, `usage` and (for aligned models) `confidence_score`.
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.nugen.in"
CHAT_PATH = "/api/v3/inference/chat/completions"
RETRYABLE_STATUS = {502, 503, 504}


class NugenError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ChatResult:
    content: str
    model: str
    latency_seconds: float
    usage: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    finish_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


def mock_enabled() -> bool:
    return os.getenv("NUGEN_MOCK", "").lower() in {"1", "true", "yes"}


def chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 500,
    temperature: float = 0.2,
    timeout: Optional[float] = None,
) -> ChatResult:
    if mock_enabled():
        from app.mock_nugen import mock_chat

        return mock_chat(model, messages)

    api_key = (os.getenv("NUGEN_API_KEY") or "").strip()
    if not api_key:
        raise NugenError("NUGEN_API_KEY is not configured")
    if not model:
        raise NugenError("model id is not configured")

    url = os.getenv("NUGEN_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + CHAT_PATH
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Server default is 1500 tokens, which would silently cut grounded prompts.
        "prompt_truncate_len": int(os.getenv("NUGEN_PROMPT_TRUNCATE_LEN", "4096")),
        "stream": False,
    }
    timeout = timeout or float(os.getenv("NUGEN_TIMEOUT_SECONDS", "90"))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    start = time.perf_counter()
    last_error: Optional[NugenError] = None
    for attempt in range(2):  # one retry, only for transient failures
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = NugenError(f"Nugen request failed: {type(exc).__name__}: {exc}")
            continue
        if response.status_code in RETRYABLE_STATUS and attempt == 0:
            last_error = NugenError(
                f"Nugen API returned {response.status_code}: {response.text[:300]}",
                response.status_code,
            )
            continue
        if response.status_code >= 400:
            raise NugenError(
                f"Nugen API returned {response.status_code}: {response.text[:300]}",
                response.status_code,
            )
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise NugenError(f"Nugen API returned non-JSON body: {exc}") from exc

        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = (message.get("content") or "").strip()
        if not content:
            raise NugenError("Nugen API returned an empty completion")
        return ChatResult(
            content=content,
            model=model,
            latency_seconds=round(time.perf_counter() - start, 3),
            usage=data.get("usage"),
            confidence=data.get("confidence_score"),
            finish_reason=choices[0].get("finish_reason") if choices else None,
            raw=data,
        )
    raise last_error or NugenError("Nugen request failed")
