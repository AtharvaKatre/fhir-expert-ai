# Nugen Exercise Report — FHIR Resource Analyzer

## 1. Objective

The goal of this exercise was to create a domain-aligned model on the Nugen platform, evaluate its behavior through inference requests, and build a small application around the resulting model.

I chose HL7 FHIR as the domain because it provides a structured healthcare data model with distinct resources and relationships. That made it possible to create a focused corpus and benchmark both resource-level understanding and relational reasoning.

## 2. Domain corpus

The alignment corpus contains five compact reference documents:

1. FHIR core resources
2. Observation and diagnostic results
3. Conditions, allergies and medication resources
4. Patient/Encounter workflow
5. Healthcare prior authorization and FHIR

The documents were written as focused domain references and contain no real patient data.

## 3. Aligned model

Base model: `Llama-V3p2-3b-Reasoning`

Aligned model:

`model_fhir-healthcare-reasoning_alignment_01m1tybce6bsr93`

The aligned model was evaluated with a 15-question benchmark covering:

- FHIR resource purpose
- Similar-resource distinctions
- Resource relationships
- Medication workflow
- Prior authorization
- Insufficient-information scenarios

## 4. Experiment A — aligned model only

The benchmark was sent directly to the aligned model without application-level domain grounding.

- Requests completed: 15/15
- Correct: 4/15
- Partial: 5/15
- Incorrect: 6/15
- Strict accuracy: 26.7%
- Approximate partial-credit score: 43.3%
- Average latency: 2.734 seconds

A notable failure mode was plausible but invalid FHIR terminology/resource hallucination. This showed that domain alignment alone did not guarantee precise resource-level reasoning on the benchmark.

## 5. Experiment B — aligned model + targeted grounding

The same 15 questions were sent to the same aligned model. The application selected the relevant domain documents for each question and included those documents in the inference prompt.

- Requests completed: 15/15
- Correct: 9/15
- Partial: 4/15
- Incorrect: 2/15
- Strict accuracy: 60.0%
- Approximate partial-credit score: 73.3%
- Average latency: 1.573 seconds

Strict accuracy improved by **33.3 percentage points**, from 26.7% to 60.0%. The number of incorrect answers fell from 6 to 2.

The grounded run was also faster in this experiment (1.573s vs 2.734s average), but these are separate API runs and latency is variable, so the latency difference is not treated as a model-quality improvement.

## 6. What grounding improved

The grounded model correctly handled several distinctions that were unreliable in the model-only run, including:

- Observation measurements vs DiagnosticReport-level interpretation
- MedicationRequest vs MedicationStatement
- Patient, Encounter and Observation relationships in an outpatient example
- MedicationRequest vs MedicationDispense vs MedicationAdministration
- Basic prior-authorization workflow reasoning

The grounded run also avoided the invalid FHIR resource hallucinations observed in the model-only experiment.

## 7. Application design

The resulting application is a small FastAPI service with a browser UI.

### Request flow

```text
User enters FHIR JSON + question
              |
              v
       FastAPI /analyze
              |
       Validate resourceType
              |
       Select relevant docs
              |
              v
       Nugen aligned model
              |
              v
    Answer + request metadata
```

### Resource-aware context selection

The application maps common FHIR resource types to relevant documents. It then adds documents based on question terms when appropriate. For example:

- `Observation` → core FHIR + Observation/Diagnostic Results
- `AllergyIntolerance` → Conditions/Allergies/Medications
- `MedicationRequest` → Conditions/Allergies/Medications
- Prior-authorization questions → prior-authorization reference in addition to the resource-specific context

This keeps the prompt focused instead of sending the entire corpus for every request.

### Deterministic validation

The API validates that the incoming value is a JSON object and contains a non-empty `resourceType`. It deliberately does not claim to perform full FHIR schema validation.

This separation is intentional: deterministic application logic handles input boundaries, while the aligned model handles natural-language reasoning over the supplied resource and domain context.

## 8. Manual application validation

After the benchmark, the interactive application was tested with five representative cases.

| Test | Result | Notes |
|---|---|---|
| Observation + Encounter | Mostly correct | Identified Observation and Encounter; did not explicitly state the `encounter` reference relationship |
| AllergyIntolerance | Correct | Identified resource and `Patient/example` reference |
| Medication workflow | Correct | Correctly distinguished Request, Administration and Dispense |
| Prior authorization | Correct | Identified missing clinical/administrative information and payer requirements |
| Insufficient prior-auth information | Partial | Correctly avoided making an approval decision but identified only one missing category |

These tests confirmed that the application-level grounding behaved as intended across different resource types and workflow questions.

## 9. Remaining limitations

The model still produced incomplete answers on some benchmark questions. Grounding improves factual precision but does not guarantee complete reasoning.

For that reason, the application should not be treated as:

- a full FHIR validator,
- a clinical decision system,
- a payer authorization engine, or
- an authoritative source of regulatory compliance decisions.

A production implementation would add stronger schema validation, structured output contracts, application-level checks for critical relationships, authentication/authorization, logging/observability, and more extensive evaluation against representative domain data.

## 10. Engineering decisions

### Why inference-time grounding?

The benchmark showed a substantial improvement when relevant domain references were supplied during inference. The application therefore uses the aligned model as the reasoning component and targeted reference documents as contextual grounding.

### Why resource-aware selection?

A small routing layer makes the grounding explicit and keeps prompts focused. It also makes the behavior inspectable: the UI reports which reference documents were used for each answer.

### Why keep validation outside the model?

A language model can produce plausible but invalid FHIR terminology. Input validation therefore remains deterministic and outside the model. In a production system, critical output claims would also be validated before being consumed downstream.

## 11. Reproduction

1. Create and deploy the FHIR-aligned model in Nugen.
2. Configure `NUGEN_API_KEY` and `NUGEN_MODEL_ID` in `.env`.
3. Install dependencies from `requirements.txt`.
4. Run `uvicorn app.main:app --reload`.
5. Open the browser UI.
6. Submit the included blood-pressure example.
7. Try the AllergyIntolerance, MedicationRequest and prior-authorization examples.
8. Run `pytest -q` for the included API tests.
9. Review the benchmark artifacts under `results/`.

## 12. Conclusion

The experiment suggests that Nugen domain alignment is useful as part of a larger application architecture, while targeted inference-time grounding provided a substantial additional improvement on this benchmark.

The main engineering takeaway is not that the aligned model replaces deterministic validation. Instead, it can serve as a reasoning layer when the application supplies focused domain context and validates the boundaries around model-generated output.
