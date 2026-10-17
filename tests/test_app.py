from fastapi.testclient import TestClient

from app.main import app, choose_docs

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_resource_validation():
    response = client.post(
        "/analyze",
        json={"question": "What is this?", "resource": {"status": "final"}},
    )
    assert response.status_code == 422


def test_resource_type_validation():
    response = client.post(
        "/analyze",
        json={
            "question": "What is this?",
            "resource": {"resourceType": ""},
        },
    )
    assert response.status_code == 422


def test_context_selection_for_observation():
    docs = choose_docs(
        "Observation",
        "How is this measurement related to the patient and encounter?",
    )
    assert docs == [
        "01_fhir_core_resources.txt",
        "02_observation_and_results.txt",
        "04_patient_encounter_workflow.txt",
    ]


def test_context_selection_for_prior_authorization():
    docs = choose_docs(
        "MedicationRequest",
        "What information is needed for prior authorization?",
    )
    assert docs == [
        "03_conditions_allergies_medications.txt",
        "05_healthcare_prior_authorization_and_fhir.txt",
    ]
