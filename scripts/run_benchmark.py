"""Repeatable benchmark for the router -> experts -> synthesis pipeline.

Scores, per case:
  - routing precision / recall against `expected_experts`
  - key-point coverage of the final answer (case-insensitive substring match;
    each key point is a list of acceptable alternatives)
  - pass = coverage >= PASS_THRESHOLD, every `must_include` point present,
    and at least one expert answered

Usage (from the project root):
  python scripts/run_benchmark.py --label baseline
  python scripts/run_benchmark.py --label after-feedback --compare results/orchestration_baseline.json
  python scripts/run_benchmark.py --mode stage1_single_model --label stage1     # stage 1 app (/analyze) baseline
  python scripts/run_benchmark.py --mode routing_only --label routing   # router calls only (cheap)
  python scripts/run_benchmark.py --mock --label harness-check           # no API calls
  python scripts/run_benchmark.py --rescore results/orchestration_panel-v1.json   # re-apply scoring, no API calls
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

PASS_THRESHOLD = 0.6
DEFAULT_CASE_FILES = [ROOT / "benchmark" / "orchestration_benchmark.json",
                      ROOT / "benchmark" / "feedback_cases.json"]


def load_cases(paths):
    cases = []
    for path in paths:
        path = Path(path)
        if path.exists():
            cases.extend(json.loads(path.read_text(encoding="utf-8")))
    return cases


def key_point_coverage(answer: str, key_points) -> float:
    if not key_points:
        return 1.0
    text = answer.lower()
    hits = sum(1 for alternatives in key_points if any(a.lower() in text for a in alternatives))
    return round(hits / len(key_points), 3)


def routing_scores(selected, expected):
    selected, expected = set(selected), set(expected)
    if not expected:
        return None, None, None
    tp = len(selected & expected)
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(expected)
    return round(precision, 3), round(recall, 3), selected == expected


def run_case(case, mode):
    from app.orchestrator import AllExpertsFailed, orchestrate
    from app.router import route

    resource = case.get("resource") or {}
    start = time.perf_counter()
    if mode == "stage1_single_model":
        # The stage 1 application exactly as submitted: one aligned model, resource-aware grounding.
        from app.main import MODEL_ID, call_nugen, choose_docs

        docs = choose_docs(resource.get("resourceType", ""), case["question"])
        try:
            answer, _, _, _ = call_nugen(case["question"], resource, docs)
            trace = {"route": {"experts": ["fhir_structure"], "method": "single_model"}, "answer": answer,
                     "experts": [{"expert": "stage1", "status": "ok", "error": None}]}
        except Exception as exc:
            trace = {"route": {"experts": ["fhir_structure"], "method": "single_model"}, "answer": "",
                     "experts": [{"expert": "stage1", "status": "error", "error": str(exc), "model_id": MODEL_ID}]}
    elif mode == "routing_only":
        decision = route(case["question"], resource.get("resourceType", ""))
        trace = {"route": {"experts": decision.experts, "method": decision.method, "error": decision.error},
                 "answer": "", "experts": []}
    else:
        override = ["fhir_structure"] if mode == "single_fhir_expert" else None
        try:
            trace = orchestrate(case["question"], resource, override)
        except AllExpertsFailed as exc:
            trace = {**exc.trace, "answer": ""}
    latency = round(time.perf_counter() - start, 3)

    precision, recall, exact = routing_scores(trace["route"]["experts"], case.get("expected_experts", []))
    coverage = key_point_coverage(trace.get("answer", ""), case.get("key_points", []))
    must_ok = key_point_coverage(trace.get("answer", ""), case.get("must_include", [])) == 1.0
    expert_statuses = {e["expert"]: e["status"] for e in trace.get("experts", [])}
    answered = any(s == "ok" for s in expert_statuses.values())
    return {
        "id": case.get("id"),
        "question": case["question"],
        "expected_experts": case.get("expected_experts", []),
        "selected_experts": trace["route"]["experts"],
        "route_method": trace["route"]["method"],
        "routing_precision": precision,
        "routing_recall": recall,
        "routing_exact": exact,
        "expert_statuses": expert_statuses,
        "key_point_coverage": None if mode == "routing_only" else coverage,
        "must_include_ok": must_ok,
        "passed": None if mode == "routing_only" else (answered and coverage >= PASS_THRESHOLD and must_ok),
        "synthesis_method": (trace.get("synthesis") or {}).get("method"),
        "latency_seconds": latency,
        "answer": trace.get("answer", ""),
        "errors": [e["error"] for e in trace.get("experts", []) if e.get("error")]
        + ([trace["route"]["error"]] if trace["route"].get("error") else []),
    }


def summarize(rows, mode):
    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 3) if values else None

    statuses = [s for r in rows for s in r["expert_statuses"].values()]
    summary = {
        "cases": len(rows),
        "routing_precision": mean(r["routing_precision"] for r in rows),
        "routing_recall": mean(r["routing_recall"] for r in rows),
        "routing_exact_match": mean(float(r["routing_exact"]) for r in rows if r["routing_exact"] is not None),
        "router_fallback_rate": mean(float(r["route_method"] == "keyword_fallback") for r in rows),
        "avg_latency_seconds": mean(r["latency_seconds"] for r in rows),
    }
    if mode != "routing_only":
        summary.update(
            key_point_coverage=mean(r["key_point_coverage"] for r in rows),
            pass_rate=mean(float(r["passed"]) for r in rows),
            expert_calls=len(statuses),
            expert_failure_rate=round(sum(s != "ok" for s in statuses) / len(statuses), 3) if statuses else None,
        )
    return summary


def rescore(path: Path, cases):
    """Apply the current scoring rules (e.g. new must_include points) to saved answers."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    by_id = {c.get("id"): c for c in cases}
    for row in saved["cases"]:
        case = by_id.get(row["id"])
        if not case or saved["mode"] == "routing_only":
            continue
        answered = any(s == "ok" for s in row["expert_statuses"].values()) and bool(row["answer"])
        row["key_point_coverage"] = key_point_coverage(row["answer"], case.get("key_points", []))
        row["must_include_ok"] = key_point_coverage(row["answer"], case.get("must_include", [])) == 1.0
        row["passed"] = answered and row["key_point_coverage"] >= PASS_THRESHOLD and row["must_include_ok"]
    before = saved["summary"]
    saved["summary"] = summarize(saved["cases"], saved["mode"])
    saved["rescored_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{path.name}: pass_rate {before.get('pass_rate')} -> {saved['summary']['pass_rate']}, "
          f"coverage {before.get('key_point_coverage')} -> {saved['summary']['key_point_coverage']}")
    print("  now failing:", [r["id"] for r in saved["cases"] if r["passed"] is False])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--mode", choices=["orchestrated", "stage1_single_model", "single_fhir_expert", "routing_only"],
                        default="orchestrated")
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASE_FILES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compare", help="previous results JSON to diff against")
    parser.add_argument("--mock", action="store_true", help="use the offline mock instead of the Nugen API")
    parser.add_argument("--rescore", help="re-score the saved answers in a results file with the current cases")
    args = parser.parse_args()

    if args.rescore:
        rescore(Path(args.rescore), load_cases(args.cases))
        return

    if args.mock:
        os.environ["NUGEN_MOCK"] = "1"
    cases = load_cases(args.cases)[: args.limit]
    rows = []
    for i, case in enumerate(cases, 1):
        row = run_case(case, args.mode)
        rows.append(row)
        print(f"[{i}/{len(cases)}] {row['id']}: route={row['selected_experts']} ({row['route_method']}) "
              f"coverage={row['key_point_coverage']} passed={row['passed']} {row['latency_seconds']}s"
              + (f" errors={len(row['errors'])}" if row["errors"] else ""))

    from app.experts import EXPERTS
    from app.router import router_model_id
    from app.synthesis import synth_model_id

    result = {
        "label": args.label,
        "mode": args.mode,
        "mock": os.getenv("NUGEN_MOCK") == "1",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "router_model_id": router_model_id(),
            "synthesis_model_id": synth_model_id(),
            "experts": {k: e.model_id or None for k, e in EXPERTS.items()},
            "pass_threshold": PASS_THRESHOLD,
        },
        "summary": summarize(rows, args.mode),
        "cases": rows,
    }
    out = ROOT / "results" / f"orchestration_{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSUMMARY", json.dumps(result["summary"], indent=2))
    print(f"Saved {out.relative_to(ROOT)}")

    if args.compare:
        previous = json.loads(Path(args.compare).read_text(encoding="utf-8"))["summary"]
        print(f"\nDELTA vs {args.compare}")
        for key, value in result["summary"].items():
            before = previous.get(key)
            if isinstance(value, (int, float)) and isinstance(before, (int, float)):
                print(f"  {key}: {before} -> {value} ({value - before:+.3f})")


if __name__ == "__main__":
    main()
