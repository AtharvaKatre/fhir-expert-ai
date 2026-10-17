"""Turn user feedback into concrete improvements, then re-run the benchmark.

1. Routing corrections (`expected_experts`) become few-shot examples in
   data/router_examples.json, which the router model sees on every request.
2. Expected key points (usually from thumbs-down feedback) become new
   regression cases in benchmark/feedback_cases.json.

Usage (from the project root):
  python scripts/improve_from_feedback.py                       # uses FEEDBACK_PATH or data/feedback.jsonl
  python scripts/improve_from_feedback.py --feedback exported.jsonl --dry-run
Then:
  python scripts/run_benchmark.py --label after-feedback --compare results/orchestration_<previous>.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.experts import EXPERTS  # noqa: E402

ROUTER_EXAMPLES = ROOT / "data" / "router_examples.json"
FEEDBACK_CASES = ROOT / "benchmark" / "feedback_cases.json"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feedback", default=os.getenv("FEEDBACK_PATH", ROOT / "data" / "feedback.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    feedback = read_jsonl(Path(args.feedback))
    examples = read_json(ROUTER_EXAMPLES, [])
    cases = read_json(FEEDBACK_CASES, [])
    known_questions = {e["question"].strip().lower() for e in examples}
    known_case_ids = {c["id"] for c in cases}
    new_examples, new_cases, skipped = [], [], 0

    for row in feedback:
        trace = row.get("trace") or {}
        question = trace.get("question")
        if not question:
            skipped += 1  # trace expired from the server's memory; nothing to replay
            continue
        expected = [k for k in (row.get("expected_experts") or []) if k in EXPERTS]
        if not expected and row.get("rating") == "up":
            expected = trace.get("route", {}).get("experts", [])

        if row.get("expected_experts") and question.strip().lower() not in known_questions:
            new_examples.append({"question": question, "experts": expected, "source": row["trace_id"]})
            known_questions.add(question.strip().lower())

        key_points = [p.strip() for p in (row.get("expected_key_points") or []) if p and p.strip()]
        case_id = f"fb-{row['trace_id'][:10]}"
        if key_points and case_id not in known_case_ids:
            case = {"id": case_id, "question": question, "expected_experts": expected,
                    "key_points": [[p] for p in key_points], "source": "user_feedback"}
            if trace.get("resource"):
                case["resource"] = trace["resource"]
            new_cases.append(case)
            known_case_ids.add(case_id)

    print(f"feedback rows: {len(feedback)} (skipped without trace: {skipped})")
    print(f"new router examples: {len(new_examples)}")
    print(f"new benchmark cases: {len(new_cases)}")
    if args.dry_run:
        print(json.dumps({"router_examples": new_examples, "benchmark_cases": new_cases}, indent=2))
        return
    if new_examples:
        ROUTER_EXAMPLES.parent.mkdir(exist_ok=True)
        ROUTER_EXAMPLES.write_text(json.dumps(examples + new_examples, indent=2), encoding="utf-8")
    if new_cases:
        FEEDBACK_CASES.write_text(json.dumps(cases + new_cases, indent=2), encoding="utf-8")
    print("Next: python scripts/run_benchmark.py --label after-feedback --compare <previous results json>")


if __name__ == "__main__":
    main()
