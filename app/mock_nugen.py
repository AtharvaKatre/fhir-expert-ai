"""Deterministic offline stand-in for Nugen, enabled only with NUGEN_MOCK=1.

Used by tests and to smoke-test the benchmark harness without spending
credits. Mock answers are not model output and every response is flagged
with "mock": true.
"""
import json
import re
from typing import Dict, List

from app.nugen_client import ChatResult


def mock_chat(model: str, messages: List[Dict[str, str]]) -> ChatResult:
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if system.startswith("You route"):
        from app.router import keyword_route

        question = re.search(r"Question: (.*)$", user, re.DOTALL).group(1)
        resource_type = re.search(r"FHIR resourceType: (\S+)", user).group(1)
        experts = keyword_route(question, "" if resource_type == "(none)" else resource_type)
        content = json.dumps({"experts": experts, "reason": "mock keyword routing"})
    elif system.startswith("You merge"):
        content = "[MOCK SYNTHESIS] " + user.split("SPECIALIST ANSWERS:", 1)[-1].strip()
    else:
        name = re.search(r"You are the (.+?) in a panel", system).group(1)
        content = f"[MOCK {name}] Placeholder answer; no model was called."
    return ChatResult(content=content, model=model, latency_seconds=0.0, confidence=None)
