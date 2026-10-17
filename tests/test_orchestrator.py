import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import nugen_client, router
from app.experts import EXPERTS
from app.main import app
from app.nugen_client import ChatResult, NugenError

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


@pytest.fixture(autouse=True)
def configured(monkeypatch, tmp_path):
    monkeypatch.delenv("NUGEN_MOCK", raising=False)
    monkeypatch.setenv("EXPERT_FHIR_MODEL_ID", "model_fhir")
    monkeypatch.setenv("EXPERT_CLINICAL_MODEL_ID", "model_clinical")
    monkeypatch.setenv("EXPERT_PRIOR_AUTH_MODEL_ID", "model_pa")
    monkeypatch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(router, "ROUTER_EXAMPLES_PATH", tmp_path / "router_examples.json")


def fake_chat_factory(route_to=("fhir_structure", "prior_authorization"), fail_models=(),
                      router_text=None, synth_fails=False, calls=None):
    def fake_chat(model, messages, max_tokens=500, temperature=0.2, timeout=None):
        system = messages[0]["content"]
        if calls is not None:
            calls.append((model, time.perf_counter(), threading.get_ident()))
        if system.startswith("You route"):
            text = router_text if router_text is not None else json.dumps({"experts": list(route_to), "reason": "test"})
            return ChatResult(text, model, 0.01)
        if system.startswith("You merge"):
            if synth_fails:
                raise NugenError("synth down", 503)
            return ChatResult("SYNTHESIZED ANSWER", model, 0.01)
        if model in fail_models:
            raise NugenError("Nugen API returned 500: boom", 500)
        time.sleep(0.2)
        return ChatResult(f"answer from {model}", model, 0.2, confidence=88.0)
    return fake_chat


def test_orchestrate_routes_calls_experts_and_synthesizes(monkeypatch):
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory())
    response = client.post("/orchestrate", json={"question": "Which resource represents a prior authorization request?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"]["method"] == "model"
    assert body["route"]["experts"] == ["fhir_structure", "prior_authorization"]
    assert [e["status"] for e in body["experts"]] == ["ok", "ok"]
    assert body["synthesis"]["method"] == "model"
    assert body["answer"] == "SYNTHESIZED ANSWER"
    assert body["degraded"] is False


def test_experts_run_in_parallel(monkeypatch):
    calls = []
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory(
        route_to=("fhir_structure", "clinical_terminology", "prior_authorization"), calls=calls))
    start = time.perf_counter()
    response = client.post("/orchestrate", json={"question": "Explain this coverage and lab data"})
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert len(response.json()["experts"]) == 3
    assert elapsed < 0.55  # three 0.2s expert calls would take >= 0.6s sequentially


def test_single_expert_failure_is_isolated(monkeypatch):
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory(fail_models=("model_pa",)))
    body = client.post("/orchestrate", json={"question": "prior auth for an MRI order?"}).json()
    statuses = {e["expert"]: e["status"] for e in body["experts"]}
    assert statuses == {"fhir_structure": "ok", "prior_authorization": "error"}
    assert body["synthesis"]["method"] == "single_expert"
    assert body["answer"] == "answer from model_fhir"
    assert body["degraded"] is True


def test_all_experts_fail_returns_502_with_trace(monkeypatch):
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory(fail_models=("model_fhir", "model_pa")))
    response = client.post("/orchestrate", json={"question": "prior auth for an MRI order?"})
    assert response.status_code == 502
    assert len(response.json()["detail"]["trace"]["experts"]) == 2


def test_unconfigured_expert_is_reported(monkeypatch):
    monkeypatch.setenv("EXPERT_PRIOR_AUTH_MODEL_ID", "")
    assert EXPERTS["prior_authorization"].model_id == ""
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory())
    body = client.post("/orchestrate", json={"question": "prior auth?"}).json()
    statuses = {e["expert"]: e["status"] for e in body["experts"]}
    assert statuses["prior_authorization"] == "not_configured"


def test_router_falls_back_to_keywords_on_bad_output(monkeypatch):
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory(router_text="I think the clinical one"))
    body = client.post("/orchestrate", json={
        "question": "What does the LOINC code mean?",
        "resource": {"resourceType": "Observation", "status": "final"},
    }).json()
    assert body["route"]["method"] == "keyword_fallback"
    assert "clinical_terminology" in body["route"]["experts"]


def test_router_failure_falls_back(monkeypatch):
    def failing(model, messages, **kwargs):
        if messages[0]["content"].startswith("You route"):
            raise NugenError("Quota limit reached", 429)
        return fake_chat_factory()(model, messages, **kwargs)
    monkeypatch.setattr(nugen_client, "chat", failing)
    body = client.post("/orchestrate", json={"question": "Does this Claim need prior authorization from the payer?"}).json()
    assert body["route"]["method"] == "keyword_fallback"
    assert "429" in body["route"]["error"] or "Quota" in body["route"]["error"]
    assert "prior_authorization" in body["route"]["experts"]


def test_synthesis_failure_falls_back_to_concatenation(monkeypatch):
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory(synth_fails=True))
    body = client.post("/orchestrate", json={"question": "Which resource for prior authorization?"}).json()
    assert body["synthesis"]["method"] == "concatenation_fallback"
    assert "answer from model_fhir" in body["answer"] and "answer from model_pa" in body["answer"]


def test_router_ignores_unknown_expert_keys():
    parsed = router.parse_router_output('Sure: {"experts": ["made_up", "prior_authorization"], "reason": "x"}')
    assert parsed["experts"] == ["prior_authorization"]
    assert router.parse_router_output('{"experts": ["made_up"]}') is None


def test_manual_expert_override_and_validation(monkeypatch):
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory())
    body = client.post("/orchestrate", json={"question": "anything here", "experts": ["clinical_terminology"]}).json()
    assert body["route"]["method"] == "manual_override"
    assert client.post("/orchestrate", json={"question": "anything", "experts": ["nope"]}).status_code == 422
    assert client.post("/orchestrate", json={"question": "anything", "resource": {"status": "final"}}).status_code == 422


def test_feedback_roundtrip_and_improvement_script(monkeypatch, tmp_path):
    monkeypatch.setattr(nugen_client, "chat", fake_chat_factory())
    trace = client.post("/orchestrate", json={"question": "How is a prior authorization decision returned?"}).json()
    response = client.post("/feedback", json={
        "trace_id": trace["trace_id"], "rating": "down", "comment": "missing preAuthRef",
        "expected_experts": ["prior_authorization"], "expected_key_points": ["ClaimResponse", "preAuthRef"],
    })
    assert response.json() == {"status": "recorded", "trace_found": True}
    assert client.get("/feedback/summary").json()["with_key_points"] == 1
    assert client.post("/feedback", json={"trace_id": trace["trace_id"], "rating": "meh"}).status_code == 422

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "improve_from_feedback.py"),
         "--feedback", str(tmp_path / "feedback.jsonl"), "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "new router examples: 1" in result.stdout
    assert "new benchmark cases: 1" in result.stdout


def test_router_examples_are_included_in_router_prompt(monkeypatch):
    router.ROUTER_EXAMPLES_PATH.write_text(json.dumps([
        {"question": "Is an MRI covered?", "experts": ["prior_authorization"]}]), encoding="utf-8")
    messages = router._build_messages("new question", "")
    assert "Is an MRI covered?" in messages[1]["content"]


def test_feedback_export_requires_admin_token(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.get("/feedback/export").status_code == 403
    monkeypatch.setenv("ADMIN_TOKEN", "secret-token")
    assert client.get("/feedback/export", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.get("/feedback/export", headers={"X-Admin-Token": "secret-token"}).status_code == 200


def test_mock_mode_end_to_end(monkeypatch):
    monkeypatch.setenv("NUGEN_MOCK", "1")
    body = client.post("/orchestrate", json={"question": "Which Da Vinci guides support prior authorization?"}).json()
    assert body["mock"] is True
    assert "prior_authorization" in body["route"]["experts"]
    assert body["answer"]
