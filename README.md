# FHIR Expert Panel (Nugen multi-expert system)

**Live app:** https://fhir-expert-ai.onrender.com (free tier: the first request after idle can take about a minute to wake up)

**Problem.** Prior authorization is one of the most costly bottlenecks in US healthcare, and CMS-0057-F turns it into a FHIR API workflow starting in 2026–2027. The teams implementing those APIs must work out which FHIR data describes an order, what clinical codes mean, and what the payer requires. A single general model blurs these three areas. They need different expertise.

**Users.** The primary users are health-IT and interoperability analysts and developers at providers, payers and EHR vendors who are building prior-authorization workflows. Provider prior-auth coordinators and payer utilization-management analysts can use it to understand the standards and workflow. It does not decide coverage for a specific order, because that needs payer-specific rules (a natural next expert) and real data integration with HIPAA safeguards. It is not intended for patients.

**Solution.** A **mother/router model** delegates each question to a fleet of **Nugen domain-aligned expert models**, combines their expertise, and improves over time:

1. the **mother (router) model** picks the relevant experts,
2. the selected **expert models run in parallel**, each grounded on its own domain documents,
3. a **synthesis model** merges their answers,
4. **user feedback** and a **repeatable benchmark** drive an improvement loop.

It extends the original single-model FHIR Resource Analyzer (stage 1). That analyzer still works at `POST /analyze` and `/single`; see [`STAGE1_REPORT.md`](STAGE1_REPORT.md) for that initial experiment, where strict accuracy went from 26.7% to 60.0% with grounding.

## Architecture

```text
Browser (/)  ──POST /orchestrate──►  FastAPI
                                       │
                         ┌─────────────┴─────────────┐
                         │ Mother/router model (base Llama-3.2-3B)
                         │  JSON {"experts":[...]}; keyword fallback on failure
                         └─────────────┬─────────────┘
              ┌────────────────────────┼────────────────────────┐   parallel (thread pool),
              ▼                        ▼                        ▼   per-expert timeout/isolation
   FHIR Structure Expert   Clinical Data & Terminology   Prior Authorization & Payer
   (aligned model #1)      Expert (aligned model #2)     Expert (aligned model #3)
   + docs 01, 04,          + docs 02, 03, terminology    + doc 05, PA workflows
     practitioners
              └────────────────────────┼────────────────────────┘
                                       ▼
                         Synthesis model (base Llama-3.2-3B)
                         single expert → passthrough; failure → labelled concatenation
                                       ▼
                    answer + full trace (route, expert outputs, latencies, confidence)
                                       │
                 POST /feedback (👍/👎, correct experts, expected key points)
                                       ▼
   scripts/improve_from_feedback.py → data/router_examples.json (router few-shot)
                                    → benchmark/feedback_cases.json (regression cases)
   scripts/run_benchmark.py         → results/orchestration_<label>.json (+ delta vs previous run)
```

### Why application-controlled orchestration

The Nugen public API ([docs.nugen.in](https://docs.nugen.in), [OpenAPI](https://api.nugen.in/openapi-public.json)) exposes models through inference endpoints. It has **no agent-creation or agent-to-agent delegation endpoint**. The API guidance itself suggests "give each domain or use case its own aligned model and place a router in front". So routing, fan-out and synthesis happen in this app, and every model call goes to the documented `POST /api/v3/inference/chat/completions` endpoint.

### Failure handling

| Failure | Behaviour |
|---|---|
| Router call fails or returns invalid JSON / unknown keys | Deterministic keyword router; `route.method = "keyword_fallback"` |
| One expert errors, times out or is not configured | Other experts continue; status recorded per expert; `degraded = true` |
| All selected experts fail | HTTP 502 with the full trace |
| Synthesis call fails | Expert answers returned, labelled and concatenated |
| Transient 502/503/504 or connection error | One retry; 4xx (including 429 quota) is not retried |

Aligned models return a `confidence_score`. Experts below `LOW_CONFIDENCE_THRESHOLD` (default 50) are flagged to the synthesizer.

## Project structure

```text
app/
  main.py            FastAPI app: /, /single, /analyze (stage 1), /orchestrate, /experts, /feedback*
  nugen_client.py    Nugen chat/completions client (retry, timeouts, confidence)
  router.py          Router model + keyword fallback + few-shot examples from feedback
  experts.py         Expert registry (model ids from env, grounding docs, prompts)
  orchestrator.py    Parallel expert fan-out with failure isolation
  synthesis.py       Synthesis model + fallbacks
  feedback.py        Feedback capture (JSONL) with trace lookup
  mock_nugen.py      Offline mock, used only when NUGEN_MOCK=1
  index.html         Multi-expert UI     single.html  Stage 1 UI
domain_docs/         Stage 1 corpus (already uploaded to Nugen)
expert_docs/         Expert alignment + grounding docs (alignment uploads need ≥500 words each)
benchmark/           orchestration_benchmark.json (15 cases) + feedback_cases.json (from user feedback)
scripts/             provision_experts.py, run_benchmark.py, improve_from_feedback.py, nugen_evaluate.py
data/                router_examples.json (router few-shot), feedback_live.jsonl (exported live feedback)
results/             Benchmark outputs (stage 1 baseline, panel v1–v3) and stage 1 results
tests/               test_app.py (stage 1), test_orchestrator.py
```

## Configuration

Copy `.env.example` to `.env`. Every credential and model id is an environment variable:

| Variable | Purpose |
|---|---|
| `NUGEN_API_KEY` | **Required.** Nugen API key (never commit it) |
| `EXPERT_FHIR_MODEL_ID` | FHIR structure expert (default: stage 1 aligned model) |
| `EXPERT_CLINICAL_MODEL_ID` | Clinical/terminology expert (from `provision_experts.py`) |
| `EXPERT_PRIOR_AUTH_MODEL_ID` | Prior-authorization expert (from `provision_experts.py`) |
| `ROUTER_MODEL_ID`, `SYNTH_MODEL_ID` | Default `meta-llama/Llama-3.2-3B-Instruct` (base model, inference-ready) |
| `ADMIN_TOKEN` | Enables `GET /feedback/export` (header `X-Admin-Token`) |
| `FEEDBACK_PATH` | Feedback JSONL location (default `data/feedback.jsonl`) |
| `NUGEN_MOCK=1` | Offline mock responses for tests/demo; responses are flagged `"mock": true` |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # add NUGEN_API_KEY
uvicorn app.main:app --reload      # http://localhost:8000
pytest -q
```

To run without credits (no Nugen calls), set mock mode:

```bash
NUGEN_MOCK=1 EXPERT_CLINICAL_MODEL_ID=mock EXPERT_PRIOR_AUTH_MODEL_ID=mock uvicorn app.main:app
```

## Provision the expert models (spends Nugen credits)

```bash
python scripts/provision_experts.py status
python scripts/provision_experts.py align --expert clinical_terminology prior_authorization
#   uploads expert_docs/*, creates alignments on meta-llama/Llama-3.2-3B-Instruct,
#   waits for READY, deploys, prints EXPERT_*_MODEL_ID lines for .env
#   if alignment rejects the older short documents: add --new-docs-only
python scripts/provision_experts.py deploy --model-id model_fhir-healthcare-reasoning_alignment_01m1tybce6bsr93 --wait
```

Deployment takes 5–15 minutes per model. Each organisation has 10 deployment slots.

## Benchmark and improvement loop

```bash
python scripts/run_benchmark.py --mode routing_only --label routing-v1      # router only (cheap)
python scripts/run_benchmark.py --mode stage1_single_model --label stage1 # stage 1 app baseline
python scripts/run_benchmark.py --label panel-v1                            # full system
# collect feedback in the UI, then (for a deployed app, export first):
curl -H "X-Admin-Token: $ADMIN_TOKEN" https://<app-url>/feedback/export > feedback.jsonl
python scripts/improve_from_feedback.py --feedback feedback.jsonl
python scripts/run_benchmark.py --label panel-v2 --compare results/orchestration_panel-v1.json
```

Metrics: routing precision, recall and exact match against `expected_experts`; key-point coverage of the final answer; pass rate; expert failure rate; router fallback rate; latency.

### Results (live Nugen models)

A case passes when key-point coverage is at least 0.6 **and** every `must_include` point (the core fact, e.g. *PractitionerRole*) is present. Earlier runs were re-scored with this stricter rule (`run_benchmark.py --rescore`).

| Run | Cases | Pass rate | Key-point coverage | Expert failures | Avg latency |
|---|---:|---:|---:|---:|---:|
| Stage 1 app (one aligned model + grounding) | 15 | 46.7% | 0.594 | 0 / 15 | 7.3 s |
| Expert panel v1 | 15 | 93.3% | 0.948 | 1 / 26 (gateway 504, isolated) | 30.8 s |
| Expert panel v2 (+ 1 feedback case, PractitionerRole fix) | 16 | 81.2% | 0.811 | 0 / 28 | 18.0 s |
| **Expert panel v3** (+ CRD fix) | 16 | **100%** | **0.934** | 0 / 28 | 17.2 s |
| Expert panel v4 (+ synthesis uses each part from its specialist) | 16 | 100% | 0.929 | 0 / 28 | 15.9 s |

Prior-authorization cases: stage 1 passed 0 / 4, the panel passes 4 / 4. Router recall against the expected experts is 0.933 and precision 0.733, because the router tends to add the FHIR structure expert as a second opinion. The v1 to v3 latency drop is most likely warm models and the absence of gateway timeouts, not a code change. Raw outputs are in [`results/`](results/).

### One improvement cycle, end to end (live)

1. **Testing the deployed app** showed two wrong answers. "Which resource links a Practitioner to its Organization?" got *Organization* (correct: **PractitionerRole**). The MRI question claimed *CoverageEligibilityRequest* tells the EHR whether prior authorization is required (correct: **Da Vinci CRD via CDS Hooks**).
2. **User feedback:** a 👎 with key points `CRD, CDS Hooks` was exported from the live app (`/feedback/export`), and `improve_from_feedback.py` turned it into regression case `fb-58f98fc327`.
3. **The benchmark was tightened.** v1 had "passed" the PractitionerRole case at 2/3 coverage without the core fact, so `must_include` points were added and old runs re-scored.
4. **v2 confirmed the diagnosis.** The PractitionerRole case now passed after adding [`expert_docs/fhir_structure/practitioners_and_organizations.txt`](expert_docs/fhir_structure/practitioners_and_organizations.txt) to the FHIR expert's grounding. The feedback case and `multi-01` still failed on CRD. `multi-01` had passed in v1 by chance.
5. **The fix:** a "common mistakes" section in the prior-auth grounding contrasting CoverageEligibilityRequest with CRD. **v3: 16 / 16.**

Neither fix retrained a model. Both are inference-time grounding changes, so the improvement shipped with a normal deploy. With inference capture enabled, the same failures are also captured on Nugen as training data for re-alignment.

Caveats: 16 cases is small, and single runs at temperature 0.2 vary by about one case (for example `fhir-01` dipped in v2 on wording alone). Lexical scoring is a proxy, so answers were also checked by hand. I wrote the grounding docs and benchmark, so they share vocabulary.

### Per-expert evaluation on Nugen (alignment alone)

`scripts/nugen_evaluate.py` uploaded each expert's benchmark slice (question plus reference answer) and ran Nugen's LLM-judged `eval-compare` of the aligned expert against its base model, **without the reference documents** the panel adds at inference time. This isolates what alignment contributes.

| Expert | Judged correct: aligned vs base | Relevance: aligned vs base |
|---|---:|---:|
| Clinical terminology | **5/7 vs 3/7** (0.71 vs 0.43) | 0.80 vs 0.74 |
| FHIR structure | 3/7 vs 2/7 (0.43 vs 0.29) | 0.61 vs 0.66 |
| Prior authorization | 1/6 vs 1/6 (0.17 vs 0.17) | 0.48 vs 0.48 |
| Prior authorization **v2** (about 3× training corpus) | 0/6 vs 1/6 (0.00 vs 0.17) | 0.42 vs 0.38 |

**Finding:** alignment clearly helped the clinical expert and helped the FHIR expert a little, but **added nothing for the prior-authorization expert**. A direct spot check confirmed this: without grounding, the prior-auth expert gave a word-for-word identical answer to base Llama on the CMS-0057-F question, and both hallucinated on the Da Vinci question. Its training corpus was the smallest (two documents, about 1,000 words). So the panel's 4/4 on prior-authorization cases comes from **routing plus inference-time grounding**, not from alignment. **Follow-up experiment (v2):** I re-aligned the prior-auth expert on a corpus about 3× larger (about 3,200 words: CMS-0057-F, Da Vinci CRD, DTR and PAS, plus a resource FAQ; see `expert_docs/prior_authorization/`) and re-ran the same evaluation. v2 learned something: without grounding it now says "72 hours" for expedited decisions, where v1 said it could not verify. But it gives "30 days" instead of 7 calendar days for standard decisions, and still invents answers about Da Vinci and CRD. Judged correctness did not improve (a one-question difference is noise). **Why more data did not score better: a closed-book probe.** `scripts/probe_alignment.py` asked 20 single-fact questions taken from the training documents (`benchmark/pa_closed_book_probe.json`), with no grounding, at temperature 0:

| Model | Correct | Declined ("unable to verify") | Confidently wrong | Identical to base |
|---|---:|---:|---:|---:|
| Base Llama-3.2-3B | 0 | 16 | 4 | n/a |
| Prior-auth v1 (about 1,150 words) | 0 | 15 | 5 | 17 / 20 |
| Prior-auth v2 (about 3,200 words) | 3 | 3 | 14 | 0 / 20 |

v1's alignment was effectively a no-op. v2's alignment strongly changed *behaviour* (it stopped declining and answered confidently) but taught few *facts*, so confidently wrong answers rose from 4 to 14. Nugen's judge scores a refusal and a wrong answer the same, which hides this trade-off. This matches published findings that fine-tuning injects new facts slowly and can increase hallucination, while retrieval injects them reliably. **Conclusion:** inference-time grounding is what makes the panel correct. v1 stays in the live panel, because v2 hallucinates far more. Raw answers: `results/alignment_probe_pa.json`. The next lever would be explicit question/answer training data through auto-align (Enterprise; its config accepts an instructions/responses JSONL dataset), gated on this evaluation. Caveats: 6–7 questions per expert, single runs. Raw results: `results/nugen_eval_*.json`.

### How the system improves over time

| Loop | Signal | Mechanism | Status |
|---|---|---|---|
| Routing | 👎 + "which experts should have answered" | `improve_from_feedback.py` adds router few-shot examples | Implemented |
| Regression | 👎 + expected key points | New cases in `benchmark/feedback_cases.json`, re-run `run_benchmark.py --compare` | Implemented |
| Expert quality | Benchmark slice per expert, with reference answers | `scripts/nugen_evaluate.py`: Nugen `benchmarks/upload` + `evaluations/create` (expert vs base model) | Run: see *Per-expert evaluation on Nugen* above |
| Expert retraining | Captured production inference | Nugen inference capture → `auto-align` → `deployment/refresh` (same model id, no app change) | **Capture enabled** on all three experts (verified via `GET /api/v3/inference/capture`); auto-align is Enterprise-only, requested from Nugen |

```bash
python scripts/nugen_evaluate.py --dry-run                 # show per-expert benchmark slices
python scripts/nugen_evaluate.py                           # upload + eval-compare each expert vs base model
```

## Deploy (Render, Docker)

1. Render → **New → Web Service** → select this repo. Runtime **Docker**, instance **Free**, health check path `/health`. Alternatively, **New → Blueprint** uses [`render.yaml`](render.yaml).
2. Add environment variables: `NUGEN_API_KEY`, `EXPERT_CLINICAL_MODEL_ID`, `EXPERT_PRIOR_AUTH_MODEL_ID`, `ADMIN_TOKEN`.
3. Every push to `main` redeploys. Check `https://<service>.onrender.com/health`: all experts should show `"configured": true`.

Any Docker host also works: `docker build -t fhir-expert-ai . && docker run -p 8000:8000 --env-file .env fhir-expert-ai`. The container honours `$PORT`.

## API

- `POST /orchestrate` `{"question": "...", "resource": {...optional FHIR...}, "experts": [optional manual override]}` returns the answer plus the trace.
- `POST /feedback` `{"trace_id", "rating": "up"|"down", "comment", "expected_experts", "expected_key_points"}`
- `GET /experts`, `GET /health`, `GET /feedback/summary`, `GET /feedback/export` (admin)
- `POST /analyze`: stage 1 single-model analyzer (unchanged)

## Limitations

- Feedback is stored in a JSONL file. Render's free tier wipes the disk on every redeploy, so export feedback (`/feedback/export`) before pushing, or point `FEEDBACK_PATH` at a persistent volume.
- Key-point scoring is a lexical proxy for correctness. Nugen's `POST /evaluations/create` (LLM-judged) can complement it for the per-expert models.
- The router and synthesizer are 3B base models. Their errors are bounded by the fallbacks and measured by the benchmark.
- This is not a FHIR validator and not clinical advice. Do not enter real patient data.
