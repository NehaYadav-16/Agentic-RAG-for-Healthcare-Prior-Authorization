from pathlib import Path

from paig.data_gen import generate_synthetic_claims
from paig.denial_model import train_denial_model
from paig.pipeline import PriorAuthAgent, PriorAuthRequest
from paig.policy_store import load_policies
from paig.retriever import TfidfRetriever

ROOT = Path(__file__).resolve().parents[1]


def _build_agent() -> PriorAuthAgent:
    chunks = load_policies(ROOT / "data" / "policies")
    retriever = TfidfRetriever(chunks)
    df = generate_synthetic_claims(n=1200, seed=3)
    model, _ = train_denial_model(df)
    return PriorAuthAgent(retriever=retriever, denial_model=model)


def test_strong_claim_is_likely_approved():
    agent = _build_agent()
    request = PriorAuthRequest(
        claim_id="CLM-TEST-001",
        payer="Generic Commercial Payer",
        procedure_category="lumbar_mri",
        patient_age=50,
        documentation_completeness=0.95,
        conservative_treatment_tried=1,
        criteria_met_ratio=0.9,
        clinical_query="low back pain radiculopathy 8 weeks failed conservative therapy",
    )
    result = agent.run(request)
    assert result.decision == "LIKELY_APPROVED"
    assert result.appeal_letter is None
    assert len(result.retrieved) > 0


def test_weak_claim_triggers_appeal_letter_with_citations():
    agent = _build_agent()
    request = PriorAuthRequest(
        claim_id="CLM-TEST-002",
        payer="Regional HMO Plan",
        procedure_category="bariatric_surgery",
        patient_age=35,
        documentation_completeness=0.1,
        conservative_treatment_tried=0,
        criteria_met_ratio=0.1,
        clinical_query="requesting sleeve gastrectomy",
    )
    result = agent.run(request)
    assert result.decision == "HIGH_DENIAL_RISK"
    assert result.appeal_letter is not None
    assert "POL-SURG-031" in result.appeal_letter
