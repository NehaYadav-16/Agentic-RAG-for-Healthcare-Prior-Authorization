"""End-to-end evaluation:

1. Retrieval quality: Precision@k against a small hand-labeled gold set
   (one query per procedure category -> expected policy doc_id).
2. Pipeline simulation: runs the full agent over a sample of synthetic
   claims and reports the decision distribution, denial-model metrics,
   and the resulting dollar cost-savings estimate.

    python scripts/evaluate_rag.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paig.cost_savings import estimate_savings  # noqa: E402
from paig.denial_model import load_model  # noqa: E402
from paig.pipeline import PriorAuthAgent, PriorAuthRequest  # noqa: E402
from paig.policy_store import load_policies  # noqa: E402
from paig.retriever import TfidfRetriever  # noqa: E402

GOLD_QUERIES = [
    ("low back pain radiculopathy 6 weeks conservative therapy failed", "POL-RAD-014"),
    ("knee locking catching meniscal tear arthroscopy", "POL-ORTHO-022"),
    ("BMI 42 diabetes weight management program bariatric", "POL-SURG-031"),
    ("spondylolisthesis instability failed conservative fusion", "POL-ORTHO-045"),
    ("positive stress test chest pain cardiac catheterization", "POL-CARD-018"),
    ("snoring daytime sleepiness STOP-BANG home sleep test", "POL-PULM-009"),
    ("physical therapy progress plateau additional visits", "POL-REHAB-005"),
    ("heavy smoker screening low dose CT chest nodule", "POL-RAD-027"),
    ("family history breast cancer BRCA genetic testing counseling", "POL-GEN-011"),
    ("home mobility evaluation power wheelchair face to face", "POL-DME-003"),
]


def precision_at_k(retriever: TfidfRetriever, k: int = 3) -> float:
    hits = 0
    for query, expected_doc_id in GOLD_QUERIES:
        results = retriever.retrieve(query, top_k=k)
        retrieved_doc_ids = {r.chunk.doc_id for r in results}
        if expected_doc_id in retrieved_doc_ids:
            hits += 1
    return hits / len(GOLD_QUERIES)


def main():
    policies_dir = ROOT / "data" / "policies"
    chunks = load_policies(policies_dir)
    retriever = TfidfRetriever(chunks)

    p_at_1 = precision_at_k(retriever, k=1)
    p_at_3 = precision_at_k(retriever, k=3)
    print(f"Retrieval Precision@1: {p_at_1:.1%}")
    print(f"Retrieval Precision@3: {p_at_3:.1%}")

    model_path = ROOT / "models" / "denial_model.joblib"
    if not model_path.exists():
        raise SystemExit("Run scripts/train_model.py first.")
    model = load_model(model_path)

    claims_path = ROOT / "data" / "claims" / "claims_synthetic.csv"
    df = pd.read_csv(claims_path).sample(n=500, random_state=7)

    agent = PriorAuthAgent(retriever=retriever, denial_model=model)

    decisions = []
    for _, row in df.iterrows():
        req = PriorAuthRequest(
            claim_id=row["claim_id"],
            payer=row["payer"],
            procedure_category=row["procedure_category"],
            patient_age=int(row["patient_age"]),
            documentation_completeness=row["documentation_completeness"],
            conservative_treatment_tried=int(row["conservative_treatment_tried"]),
            criteria_met_ratio=row["criteria_met_ratio"],
            prior_denial_history=int(row["prior_denial_history"]),
            peer_to_peer_requested=int(row["peer_to_peer_requested"]),
            days_to_submit=int(row["days_to_submit"]),
        )
        result = agent.run(req)
        decisions.append(result.decision)

    decision_counts = pd.Series(decisions).value_counts().to_dict()
    print(f"Decision distribution over {len(df)} sampled claims: {decision_counts}")

    savings = estimate_savings(num_claims=10_000)
    print("Projected savings for 10,000 claims/month processed with PAIG:")
    print(json.dumps(savings.__dict__, indent=2))

    report = {
        "retrieval_precision_at_1": p_at_1,
        "retrieval_precision_at_3": p_at_3,
        "decision_distribution_sample_500": decision_counts,
        "projected_savings_10k_claims": savings.__dict__,
    }
    report_path = ROOT / "models" / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
