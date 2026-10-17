"""Create, align and deploy the expert models with the documented Nugen API.

Endpoints used (https://docs.nugen.in):
  POST   /api/v3/documents/create                     (multipart: files, categories, names)
  GET    /api/v3/documents/list
  GET    /api/v3/documents/{document_id}/status
  POST   /api/v3/alignment-projects/create             (alignment_name, base_model_id, document_ids)
  GET    /api/v3/alignment-projects/{alignment_id}/status
  GET    /api/v3/models/aligned
  POST   /api/v3/models/{model_id}/deployment
  GET    /api/v3/models/{model_id}/deployment/status
  DELETE /api/v3/models/{model_id}/deployment

Usage (from the project root; these calls spend Nugen credits):
  python scripts/provision_experts.py status
  python scripts/provision_experts.py align --expert clinical_terminology prior_authorization [--new-docs-only]
  python scripts/provision_experts.py deploy --model-id <model_id> [--wait]
  python scripts/provision_experts.py undeploy --model-id <model_id>

Progress is stored in data/provision_state.json so an interrupted run can resume.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API = os.getenv("NUGEN_BASE_URL", "https://api.nugen.in").rstrip("/") + "/api/v3"
STATE_PATH = ROOT / "data" / "provision_state.json"
BASE_MODEL = os.getenv("EXPERT_BASE_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

EXPERT_SPECS = {
    "clinical_terminology": {
        "alignment_name": "fhir-clinical-terminology-expert",
        "env": "EXPERT_CLINICAL_MODEL_ID",
        "existing_docs": ["02_observation_and_results.txt", "03_conditions_allergies_medications.txt"],
        "new_docs": ["expert_docs/clinical_terminology/clinical_terminology_and_coding.txt"],
    },
    "prior_authorization": {
        "alignment_name": "fhir-prior-authorization-expert",
        "env": "EXPERT_PRIOR_AUTH_MODEL_ID",
        "existing_docs": ["05_healthcare_prior_authorization_and_fhir.txt"],
        "new_docs": ["expert_docs/prior_authorization/prior_authorization_workflows.txt"],
    },
}


def headers():
    key = (os.getenv("NUGEN_API_KEY") or "").strip()
    if not key:
        sys.exit("NUGEN_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def call(method, path, **kwargs):
    response = requests.request(method, API + path, headers=headers(), timeout=120, **kwargs)
    if response.status_code >= 400:
        sys.exit(f"{method} {path} -> {response.status_code}: {response.text[:500]}")
    return response.json() if response.content else {}


def load_state():
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}


def save_state(state):
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def poll(path, done, failed=("FAILED", "STOPPED"), every=20, limit=3600, field="status"):
    start = time.time()
    while True:
        data = call("GET", path)
        status = data.get(field)
        print(f"  {path}: {status}")
        if status in done:
            return data
        if status in failed:
            sys.exit(f"{path} ended in {status}: {json.dumps(data)[:500]}")
        if time.time() - start > limit:
            sys.exit(f"timed out waiting for {path}")
        time.sleep(every)


def ensure_documents(spec, state_entry, new_docs_only=False):
    if state_entry.get("document_ids"):
        return state_entry["document_ids"]
    listing = call("GET", "/documents/list").get("documents", [])
    by_name = {d["document_name"]: d for d in listing}
    ids = []
    for name in [] if new_docs_only else spec["existing_docs"]:
        doc = by_name.get(name)
        if not doc or doc.get("status") != "READY":
            sys.exit(f"expected an existing READY document named {name}")
        ids.append(doc["document_id"])

    to_upload = [ROOT / rel for rel in spec["new_docs"]]
    for path in to_upload:
        if path.name in by_name:
            ids.append(by_name[path.name]["document_id"])
            print(f"  reusing uploaded document {path.name}")
    to_upload = [p for p in to_upload if p.name not in by_name]
    if to_upload:
        files = [("files", (p.name, p.read_bytes(), "text/plain")) for p in to_upload]
        data = {"categories": ["Healthcare"], "names": [p.name for p in to_upload]}
        created = call("POST", "/documents/create", files=files, data=data)
        new_ids = created.get("document_ids") or created.get("documents") or []
        new_ids = [d["document_id"] if isinstance(d, dict) else d for d in new_ids]
        if len(new_ids) != len(to_upload):
            sys.exit(f"unexpected upload response: {created}")
        for doc_id in new_ids:
            poll(f"/documents/{doc_id}/status", done=("READY",), every=10)
        ids.extend(new_ids)
    state_entry["document_ids"] = ids
    return ids


def find_model_id(alignment_id):
    for model in call("GET", "/models/aligned").get("domain_aligned_models", []):
        if model.get("alignment_id") == alignment_id:
            return model["model_id"]
    return None


def deploy(model_id, wait=True):
    status = call("GET", f"/models/{model_id}/deployment/status")
    if status.get("status") == "DEPLOYED":
        print(f"  {model_id} already DEPLOYED")
        return
    if status.get("status") == "UNDEPLOYED":
        call("POST", f"/models/{model_id}/deployment")
        print(f"  deployment requested for {model_id} (docs: 5-15 minutes)")
    if wait:
        result = poll(f"/models/{model_id}/deployment/status", done=("DEPLOYED",), failed=(), every=30)
        if result.get("error"):
            print(f"  warning: {result['error']}")


def cmd_align(experts, wait_deploy, new_docs_only=False):
    state = load_state()
    for key in experts:
        spec = EXPERT_SPECS[key]
        entry = state.setdefault(key, {})
        print(f"[{key}] documents")
        doc_ids = ensure_documents(spec, entry, new_docs_only)
        save_state(state)
        if not entry.get("alignment_id"):
            print(f"[{key}] creating alignment on {BASE_MODEL} with {len(doc_ids)} documents")
            created = call("POST", "/alignment-projects/create", json={
                "alignment_name": spec["alignment_name"],
                "base_model_id": BASE_MODEL,
                "document_ids": doc_ids,
                "description": f"Expert model for the multi-expert FHIR analyzer: {key}",
            })
            entry["alignment_id"] = created["alignment_id"]
            save_state(state)
        print(f"[{key}] waiting for alignment {entry['alignment_id']}")
        poll(f"/alignment-projects/{entry['alignment_id']}/status", done=("READY",), every=30)
        entry["model_id"] = entry.get("model_id") or find_model_id(entry["alignment_id"])
        save_state(state)
        if not entry["model_id"]:
            sys.exit(f"[{key}] alignment READY but no aligned model found yet; re-run later")
        print(f"[{key}] deploying {entry['model_id']}")
        deploy(entry["model_id"], wait=wait_deploy)
    print("\nAdd to .env / hosting environment:")
    for key in experts:
        if state.get(key, {}).get("model_id"):
            print(f"{EXPERT_SPECS[key]['env']}={state[key]['model_id']}")


def cmd_status():
    models = call("GET", "/models/aligned").get("domain_aligned_models", [])
    for model in models:
        print(f"{model['model_id']}: {model.get('deployment_status')} (base {model.get('base_model_id')})")
    for project in call("GET", "/alignment-projects/list").get("alignment_projects", []):
        print(f"{project['alignment_id']} {project['alignment_name']}: {project['status']}")
    print("local state:", json.dumps(load_state(), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    align = sub.add_parser("align")
    align.add_argument("--expert", nargs="+", choices=list(EXPERT_SPECS), default=list(EXPERT_SPECS))
    align.add_argument("--no-wait-deploy", action="store_true")
    align.add_argument("--new-docs-only", action="store_true",
                       help="align only on the new expert_docs files (each >= 500 words)")
    dep = sub.add_parser("deploy")
    dep.add_argument("--model-id", required=True)
    dep.add_argument("--wait", action="store_true")
    undep = sub.add_parser("undeploy")
    undep.add_argument("--model-id", required=True)
    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "align":
        cmd_align(args.expert, wait_deploy=not args.no_wait_deploy, new_docs_only=args.new_docs_only)
    elif args.command == "deploy":
        deploy(args.model_id, wait=args.wait)
    elif args.command == "undeploy":
        print(call("DELETE", f"/models/{args.model_id}/deployment"))


if __name__ == "__main__":
    main()
