"""Interactive-ish CLI demo: runs a handful of representative claims
through the full agent and prints the decision, citations, criteria
checklist, and (if applicable) the drafted appeal letter.

    python scripts/demo_cli.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paig.denial_model import load_model  # noqa: E402
from paig.pipeline import PriorAuthAgent, PriorAuthRequest  # noqa: E402
from paig.policy_store import load_policies  # noqa: E402
from paig.retriever import TfidfRetriever  # noqa: E402

SAMPLE_REQUESTS = [
    PriorAuthRequest(
        claim_id="CLM-DEMO-001",
        payer="Generic Commercial Payer",
        procedure_category="lumbar_mri",
        patient_age=47,
        documentation_completeness=0.9,
        conservative_treatment_tried=1,
        criteria_met_ratio=0.85,
        clinical_query="low back pain 8 weeks failed physical therapy",
    ),
    PriorAuthRequest(
        claim_id="CLM-DEMO-002",
        payer="Regional HMO Plan",
        procedure_category="bariatric_surgery",
        patient_age=38,
        documentation_completeness=0.3,
        conservative_treatment_tried=0,
        criteria_met_ratio=0.25,
        clinical_query="BMI 36 requesting sleeve gastrectomy",
    ),
]


def main():
    chunks = load_policies(ROOT / "data" / "policies")
    retriever = TfidfRetriever(chunks)

    model_path = ROOT / "models" / "denial_model.joblib"
    if not model_path.exists():
        raise SystemExit("Run scripts/train_model.py first.")
    model = load_model(model_path)

    agent = PriorAuthAgent(retriever=retriever, denial_model=model)

    for req in SAMPLE_REQUESTS:
        result = agent.run(req)
        print("=" * 80)
        print(f"Claim: {result.claim_id}  ->  Decision: {result.decision}  (risk={result.denial_risk:.1%})")
        print("Audit trail:")
        for step in result.audit_trail:
            print(f"  - {step}")
        print("Criteria:")
        for c in result.criteria:
            mark = "PASS" if c.met else "FAIL"
            print(f"  [{mark}] {c.description} ({c.citation})")
        if result.appeal_letter:
            print("\nDrafted appeal letter:\n")
            print(result.appeal_letter)
        print()


if __name__ == "__main__":
    main()
