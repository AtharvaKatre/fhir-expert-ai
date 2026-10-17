"""Router: a Nugen model selects which expert models to consult.

The router model returns JSON. If the call fails or the output cannot be
parsed into known expert keys, a deterministic keyword router is used so a
single router failure never blocks the request.
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import nugen_client
from app.experts import BASE_DIR, EXPERTS

ROUTER_EXAMPLES_PATH = Path(os.getenv("ROUTER_EXAMPLES_PATH", BASE_DIR / "data" / "router_examples.json"))
DEFAULT_ROUTER_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_EXPERTS = 3


@dataclass
class RouteDecision:
    experts: List[str]
    method: str  # "model" or "keyword_fallback"
    reason: str = ""
    model_id: Optional[str] = None
    latency_seconds: Optional[float] = None
    error: Optional[str] = None
    raw_output: Optional[str] = field(default=None, repr=False)


def router_model_id() -> str:
    return (os.getenv("ROUTER_MODEL_ID") or DEFAULT_ROUTER_MODEL).strip()


def load_router_examples(limit: int = 8) -> List[Dict[str, Any]]:
    try:
        examples = json.loads(ROUTER_EXAMPLES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    valid = [
        e for e in examples
        if isinstance(e, dict) and e.get("question") and e.get("experts")
        and all(k in EXPERTS for k in e["experts"])
    ]
    return valid[-limit:]  # newest corrections are most relevant


def keyword_route(question: str, resource_type: str = "") -> List[str]:
    text = f"{question} {resource_type}".lower()
    scores = {}
    for key, expert in EXPERTS.items():
        scores[key] = sum(1 for kw in expert.keywords if kw in text)
    if resource_type in {"Claim", "ClaimResponse", "Coverage", "CoverageEligibilityRequest",
                         "CoverageEligibilityResponse"}:
        scores["prior_authorization"] += 3
    if resource_type in {"Observation", "Condition", "AllergyIntolerance", "MedicationRequest",
                         "MedicationStatement", "MedicationDispense", "MedicationAdministration"}:
        scores["clinical_terminology"] += 1
    selected = [k for k, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0][:MAX_EXPERTS]
    return selected or ["fhir_structure"]


def _build_messages(question: str, resource_type: str) -> List[Dict[str, str]]:
    catalog = "\n".join(f"- {e.key}: {e.description}" for e in EXPERTS.values())
    examples = load_router_examples()
    example_text = "\n".join(
        f'Q: {e["question"]}\nA: {json.dumps({"experts": e["experts"]})}' for e in examples
    )
    system = (
        "You route healthcare interoperability questions to specialist models. "
        "Select every specialist whose expertise is needed, usually 1 or 2, at most 3. "
        'Respond with JSON only, in the form {"experts": ["key", ...], "reason": "short reason"}. '
        "Use only these keys:\n" + catalog
    )
    user = ""
    if example_text:
        user += f"Examples of correct routing:\n{example_text}\n\n"
    user += f"FHIR resourceType: {resource_type or '(none)'}\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_router_output(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    experts = data.get("experts") if isinstance(data, dict) else None
    if not isinstance(experts, list):
        return None
    keys = list(dict.fromkeys(k for k in experts if isinstance(k, str) and k in EXPERTS))[:MAX_EXPERTS]
    if not keys:
        return None
    return {"experts": keys, "reason": str(data.get("reason", ""))[:300]}


def route(question: str, resource_type: str = "") -> RouteDecision:
    model_id = router_model_id()
    try:
        result = nugen_client.chat(
            model_id, _build_messages(question, resource_type), max_tokens=120, temperature=0.0,
            timeout=float(os.getenv("ROUTER_TIMEOUT_SECONDS", "30")),
        )
    except nugen_client.NugenError as exc:
        return RouteDecision(
            experts=keyword_route(question, resource_type), method="keyword_fallback",
            reason="router model call failed", model_id=model_id, error=str(exc),
        )

    parsed = parse_router_output(result.content)
    if not parsed:
        return RouteDecision(
            experts=keyword_route(question, resource_type), method="keyword_fallback",
            reason="router output was not valid JSON with known expert keys", model_id=model_id,
            latency_seconds=result.latency_seconds, raw_output=result.content[:500],
        )
    return RouteDecision(
        experts=parsed["experts"], method="model", reason=parsed["reason"], model_id=model_id,
        latency_seconds=result.latency_seconds, raw_output=result.content[:500],
    )
