"""Evaluate each expert on Nugen's evaluation service, against the base model.

For every expert this uploads the benchmark cases it is expected to answer
(question + reference_answer) and runs an eval-compare between the aligned
expert and the base model it was aligned from. That shows whether each
alignment earns its place in the fleet, using Nugen's LLM-judged metrics.

Endpoints used (https://docs.nugen.in):
  POST /api/v3/benchmarks/upload          (multipart: file, name, document_id, description)
  GET  /api/v3/benchmarks/{id}/status
  POST /api/v3/evaluations/create         (model_id, benchmark_id, baseline_model_id, evaluation_name)
  GET  /api/v3/evaluations/{id}/status    (QUEUED, PROCESSING, READY, FAILED)
  GET  /api/v3/evaluations/{id}/results

Usage (from the project root; spends Nugen credits unless --dry-run):
  python scripts/nugen_evaluate.py --dry-run
  python scripts/nugen_evaluate.py --expert clinical_terminology prior_authorization
  EXPERT_PRIOR_AUTH_MODEL_ID=<new id> python scripts/nugen_evaluate.py --expert prior_authorization --label v2
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from provision_experts import BASE_MODEL, call, load_state, poll, save_state  # noqa: E402

from app.experts import EXPERTS  # noqa: E402
from run_benchmark import DEFAULT_CASE_FILES, load_cases  # noqa: E402

# Document each benchmark is associated with (required by /benchmarks/upload).
ANCHOR_DOCS = {
    "fhir_structure": "01_fhir_core_resources.txt",
    "clinical_terminology": "clinical_terminology_and_coding.txt",
    "prior_authorization": "prior_authorization_workflows.txt",
}


def samples_for(expert_key, cases):
    samples = []
    for case in cases:
        if expert_key not in case.get("expected_experts", []) or not case.get("reference_answer"):
            continue
        instruction = case["question"]
        if case.get("resource"):
            instruction += "\n\nFHIR resource:\n" + json.dumps(case["resource"], indent=2)
        samples.append({"sample_num": len(samples) + 1, "instruction": instruction,
                        "response": case["reference_answer"]})
    return samples


def document_id(name):
    for doc in call("GET", "/documents/list").get("documents", []):
        if doc["document_name"] == name:
            return doc["document_id"]
    sys.exit(f"document {name} not found; run provision_experts.py align first")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expert", nargs="+", choices=list(EXPERTS), default=list(EXPERTS))
    parser.add_argument("--baseline-model", default=BASE_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--label", default="", help="suffix for output files, so earlier results are kept")
    args = parser.parse_args()

    cases = load_cases(DEFAULT_CASE_FILES)
    state = load_state()
    for key in args.expert:
        expert = EXPERTS[key]
        samples = samples_for(key, cases)
        print(f"[{key}] model={expert.model_id or '(not configured)'} samples={len(samples)} "
              f"baseline={args.baseline_model}")
        if args.dry_run:
            print(json.dumps(samples[:2], indent=2))
            continue
        if not expert.model_id:
            print(f"[{key}] skipped: {expert.model_env} not set")
            continue

        suffix = f"_{args.label}" if args.label else ""
        path = ROOT / "results" / f"nugen_benchmark_{key}{suffix}.json"
        path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
        uploaded = call("POST", "/benchmarks/upload",
                        files={"file": (path.name, path.read_bytes(), "application/json")},
                        data={"name": f"expert-panel-{key}{suffix}", "document_id": document_id(ANCHOR_DOCS[key]),
                              "description": f"Expert panel benchmark slice for {key}"})
        benchmark_id = uploaded["benchmark_id"]
        if uploaded.get("status") != "READY":
            poll(f"/benchmarks/{benchmark_id}/status", done=("READY",), every=10)

        created = call("POST", "/evaluations/create", json={
            "model_id": expert.model_id, "benchmark_id": benchmark_id,
            "baseline_model_id": args.baseline_model, "evaluation_name": f"expert-panel-{key}{suffix}",
        })
        evaluation_id = created["evaluation_id"]
        state.setdefault("evaluations", {})[key + suffix] = {"benchmark_id": benchmark_id, "evaluation_id": evaluation_id}
        save_state(state)
        poll(f"/evaluations/{evaluation_id}/status", done=("READY",), every=30)
        results = call("GET", f"/evaluations/{evaluation_id}/results")
        out = ROOT / "results" / f"nugen_eval_{key}{suffix}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[{key}] saved {out.relative_to(ROOT)}: {json.dumps(results.get('comparison') or results.get('metrics'))}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
