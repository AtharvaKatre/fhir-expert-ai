"""Application-controlled orchestration: route -> parallel experts -> synthesis.

Nugen exposes models through inference endpoints; it has no agent-to-agent
delegation API, so this module performs the delegation itself.
"""
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from app import nugen_client
from app.experts import EXPERTS, build_expert_messages
from app.router import RouteDecision, route
from app.synthesis import synthesize

_pool = ThreadPoolExecutor(max_workers=int(os.getenv("EXPERT_MAX_WORKERS", "6")))


class AllExpertsFailed(RuntimeError):
    def __init__(self, trace: Dict[str, Any]):
        super().__init__("All selected experts failed")
        self.trace = trace


@dataclass
class ExpertOutput:
    expert: str
    name: str
    model_id: Optional[str]
    status: str  # "ok", "error", "not_configured", "timeout"
    answer: str = ""
    latency_seconds: Optional[float] = None
    confidence: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _call_expert(key: str, question: str, resource: Dict[str, Any]) -> ExpertOutput:
    expert = EXPERTS[key]
    if not expert.model_id:
        return ExpertOutput(key, expert.name, None, "not_configured",
                            error=f"{expert.model_env} is not set")
    try:
        result = nugen_client.chat(
            expert.model_id, build_expert_messages(expert, question, resource),
            max_tokens=int(os.getenv("EXPERT_MAX_TOKENS", "450")), temperature=0.2,
            timeout=float(os.getenv("EXPERT_TIMEOUT_SECONDS", "90")),
        )
    except nugen_client.NugenError as exc:
        return ExpertOutput(key, expert.name, expert.model_id, "error", error=str(exc))
    return ExpertOutput(
        key, expert.name, expert.model_id, "ok", answer=result.content,
        latency_seconds=result.latency_seconds, confidence=result.confidence, usage=result.usage,
    )


def run_experts(keys: List[str], question: str, resource: Dict[str, Any]) -> List[ExpertOutput]:
    deadline = float(os.getenv("EXPERT_TIMEOUT_SECONDS", "90")) + 15
    futures = {key: _pool.submit(_call_expert, key, question, resource) for key in keys}
    wait(futures.values(), timeout=deadline)
    outputs = []
    for key, future in futures.items():
        if future.done():
            try:
                outputs.append(future.result())
            except Exception as exc:  # defensive: never let one expert sink the request
                outputs.append(ExpertOutput(key, EXPERTS[key].name, EXPERTS[key].model_id, "error",
                                            error=f"{type(exc).__name__}: {exc}"))
        else:
            future.cancel()
            outputs.append(ExpertOutput(key, EXPERTS[key].name, EXPERTS[key].model_id, "timeout",
                                        error=f"no response within {deadline:.0f}s"))
    return outputs


def orchestrate(question: str, resource: Optional[Dict[str, Any]] = None,
                experts_override: Optional[List[str]] = None) -> Dict[str, Any]:
    resource = resource or {}
    resource_type = resource.get("resourceType", "") if isinstance(resource, dict) else ""
    start = time.perf_counter()

    if experts_override:
        decision = RouteDecision(experts=experts_override, method="manual_override")
    else:
        decision = route(question, resource_type)

    outputs = run_experts(decision.experts, question, resource)
    successes = [o for o in outputs if o.status == "ok"]
    failed = [o.expert for o in outputs if o.status != "ok"]

    trace = {
        "trace_id": uuid.uuid4().hex,
        "question": question,
        "resource_type": resource_type or None,
        "resource": resource or None,
        "route": asdict(decision),
        "experts": [asdict(o) for o in outputs],
        "mock": nugen_client.mock_enabled(),
    }
    if not successes:
        trace["total_latency_seconds"] = round(time.perf_counter() - start, 3)
        raise AllExpertsFailed(trace)

    synthesis = synthesize(question, successes, failed)
    trace.update(
        answer=synthesis.answer,
        synthesis=asdict(synthesis),
        degraded=bool(failed) or decision.method == "keyword_fallback" or synthesis.method == "concatenation_fallback",
        total_latency_seconds=round(time.perf_counter() - start, 3),
    )
    return trace
