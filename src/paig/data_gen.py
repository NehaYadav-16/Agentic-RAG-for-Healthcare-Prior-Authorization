"""Generates a synthetic prior-authorization claims dataset.

No real patient or claims data is used anywhere in this project. Feature
correlations are designed to mimic realistic denial drivers (missing
documentation, criteria not met, no prior conservative treatment) so the
downstream denial-risk model has real signal to learn from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROCEDURE_CATEGORIES = [
    "lumbar_mri",
    "knee_arthroscopy",
    "bariatric_surgery",
    "spinal_fusion",
    "cardiac_cath",
    "home_sleep_study",
    "physical_therapy_extended",
    "ct_chest_lowdose",
    "genetic_testing_brca",
    "dme_wheelchair",
]

PAYERS = ["Generic Commercial Payer", "Regional HMO Plan", "National PPO Plan"]


def generate_synthetic_claims(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generates `n` synthetic prior-auth claim rows with a denial label."""
    rng = np.random.default_rng(seed)

    procedure_category = rng.choice(PROCEDURE_CATEGORIES, size=n)
    payer = rng.choice(PAYERS, size=n)
    patient_age = rng.integers(18, 85, size=n)

    # Core signal features: whether documented criteria are actually met.
    documentation_completeness = rng.uniform(0, 1, size=n)  # 0 = missing, 1 = complete
    conservative_treatment_tried = rng.integers(0, 2, size=n)
    criteria_met_ratio = rng.uniform(0, 1, size=n)  # fraction of policy criteria satisfied
    prior_denial_history = rng.integers(0, 2, size=n)
    peer_to_peer_requested = rng.integers(0, 2, size=n)
    days_to_submit = rng.integers(1, 30, size=n)

    # Denial probability: driven mainly by unmet criteria + poor documentation,
    # with a bit of noise and payer-level variation to simulate real inconsistency.
    payer_strictness = np.array(
        [{"Generic Commercial Payer": 0.05, "Regional HMO Plan": 0.12, "National PPO Plan": 0.0}[p] for p in payer]
    )
    logits = (
        -6.0
        + 4.5 * (1 - criteria_met_ratio)
        + 2.7 * (1 - documentation_completeness)
        + 0.9 * (1 - conservative_treatment_tried)
        + 0.75 * prior_denial_history
        + 1.5 * payer_strictness
        + rng.normal(0, 0.3, size=n)
    )
    denial_prob = 1 / (1 + np.exp(-logits))
    denied = rng.binomial(1, denial_prob)

    df = pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:06d}" for i in range(n)],
            "procedure_category": procedure_category,
            "payer": payer,
            "patient_age": patient_age,
            "documentation_completeness": documentation_completeness.round(3),
            "conservative_treatment_tried": conservative_treatment_tried,
            "criteria_met_ratio": criteria_met_ratio.round(3),
            "prior_denial_history": prior_denial_history,
            "peer_to_peer_requested": peer_to_peer_requested,
            "days_to_submit": days_to_submit,
            "denied": denied,
        }
    )
    return df


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "claims"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_claims()
    frame.to_csv(out_dir / "claims_synthetic.csv", index=False)
    print(f"Wrote {len(frame)} rows to {out_dir / 'claims_synthetic.csv'}")
    print(f"Denial rate: {frame['denied'].mean():.1%}")
