"""Closed-book knowledge probe: did alignment teach the model its training facts?

Asks single-fact questions drawn from the training documents, with no grounding,
at temperature 0, and scores whether the expected fact appears in the answer.
Compares any number of models (e.g. base model, aligned v1, aligned v2).

Usage (from the project root; makes len(questions) x len(models) API calls):
  python scripts/probe_alignment.py --models meta-llama/Llama-3.2-3B-Instruct model_v1 model_v2 --label pa
  python scripts/probe_alignment.py --models x --rescore results/alignment_probe_pa.json
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app import nugen_client  # noqa: E402


def contains_fact(text: str, accept) -> bool:
    """Whole-token match, so "2780" does not count as "278"."""
    return any(re.search(r"(?<![\w$])" + re.escape(a) + r"(?![\w])", text, re.IGNORECASE) for a in accept)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--questions", default=str(ROOT / "benchmark" / "pa_closed_book_probe.json"))
    parser.add_argument("--label", default="probe")
    parser.add_argument("--rescore", help="re-score a saved probe results file (no API calls)")
    args = parser.parse_args()

    if args.rescore:
        path = Path(args.rescore)
        saved = json.loads(path.read_text(encoding="utf-8"))
        for row in saved["questions"]:
            for answer in row["answers"].values():
                answer["correct"] = contains_fact(answer["text"], row["accept"])
        saved["summary"] = {m: sum(r["answers"][m]["correct"] for r in saved["questions"]) for m in saved["models"]}
        path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Correct out of", len(saved["questions"]), json.dumps(saved["summary"], indent=2))
        return

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "accept": q["accept"], "answers": {}}
        for model in args.models:
            messages = [{"role": "user", "content": q["question"] + " Answer in one or two sentences."}]
            try:
                text = nugen_client.chat(model, messages, max_tokens=120, temperature=0.0).content
            except nugen_client.NugenError as exc:
                text = f"ERROR: {exc}"
            hit = contains_fact(text, q["accept"])
            row["answers"][model] = {"correct": hit, "text": text}
        rows.append(row)
        print(q["id"].ljust(22), "  ".join(("Y" if row["answers"][m]["correct"] else ".") for m in args.models))

    summary = {m: sum(r["answers"][m]["correct"] for r in rows) for m in args.models}
    print("\nCorrect out of", len(rows), json.dumps(summary, indent=2))
    out = ROOT / "results" / f"alignment_probe_{args.label}.json"
    out.write_text(json.dumps({"run_at": datetime.now(timezone.utc).isoformat(), "models": args.models,
                               "summary": summary, "questions": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
