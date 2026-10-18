"""Synthesis: combine successful expert answers into one response."""
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from app import nugen_client
from app.experts import EXPERTS

DEFAULT_SYNTH_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# Small models often prefix "Here's the combined answer:"; strip it so the UI shows only the answer.
PREAMBLE = re.compile(r"^\s*here(?: is|'s) (?:the|a) (?:combined |merged |final )?(?:answer|response)[^:\n]*:\s*",
                      re.IGNORECASE)
LOW_CONFIDENCE_THRESHOLD =float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "50"))


@dataclass
class SynthesisResult:
    answer: str
    method: str  # "model", "single_expert", or "concatenation_fallback"
    model_id: Optional[str] = None
    latency_seconds: Optional[float] = None
    error: Optional[str] = None


def synth_model_id() -> str:
    return (os.getenv("SYNTH_MODEL_ID") or DEFAULT_SYNTH_MODEL).strip()


def _label(output) -> str:
    name = EXPERTS[output.expert].name
    if output.confidence is not None and output.confidence < LOW_CONFIDENCE_THRESHOLD:
        return f"{name} (low confidence: {output.confidence:.0f}/100)"
    return name


def _concatenate(outputs) -> str:
    return "\n\n".join(f"**{_label(o)}**\n{o.answer}" for o in outputs)


def synthesize(question: str, outputs, failed_experts: List[str]) -> SynthesisResult:
    """`outputs` are successful ExpertOutput objects (see orchestrator)."""
    if len(outputs) == 1:
        # Nothing to reconcile; avoid an extra model call that could only lose detail.
        return SynthesisResult(answer=outputs[0].answer, method="single_expert")

    expert_block = "\n\n".join(f"[{_label(o)}]\n{o.answer}" for o in outputs)
    missing = ""
    if failed_experts:
        names = ", ".join(EXPERTS[k].name for k in failed_experts)
        missing = f"\nThese specialists were unavailable, so note any gap they would have covered: {names}."
    messages = [
        {
            "role": "system",
            "content": (
                "You merge answers from healthcare interoperability specialists into one answer. "
                "Use only facts stated by the specialists; do not add new facts. "
                "If specialists disagree, state the disagreement. Prefer specialists without a low-confidence flag. "
                "Answer the question directly in at most 10 sentences or bullets." + missing
            ),
        },
        {"role": "user", "content": f"QUESTION:\n{question}\n\nSPECIALIST ANSWERS:\n{expert_block}"},
    ]
    model_id = synth_model_id()
    try:
        result = nugen_client.chat(model_id, messages, max_tokens=600, temperature=0.1)
    except nugen_client.NugenError as exc:
        return SynthesisResult(
            answer=_concatenate(outputs), method="concatenation_fallback", model_id=model_id, error=str(exc)
        )
    answer = PREAMBLE.sub("", result.content)
    return SynthesisResult(
        answer=answer or result.content, method="model", model_id=model_id,
        latency_seconds=result.latency_seconds,
    )
