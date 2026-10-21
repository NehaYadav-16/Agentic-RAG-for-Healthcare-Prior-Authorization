from paig.data_gen import generate_synthetic_claims
from paig.denial_model import predict_denial_risk, train_denial_model


def test_model_trains_and_scores_sensibly():
    df = generate_synthetic_claims(n=1500, seed=1)
    pipeline, metrics = train_denial_model(df)

    assert metrics.roc_auc > 0.75  # model should clearly beat random (0.5)

    high_risk_claim = {
        "procedure_category": "bariatric_surgery",
        "payer": "Regional HMO Plan",
        "patient_age": 40,
        "documentation_completeness": 0.1,
        "conservative_treatment_tried": 0,
        "criteria_met_ratio": 0.1,
        "prior_denial_history": 1,
        "peer_to_peer_requested": 0,
        "days_to_submit": 10,
    }
    low_risk_claim = {
        **high_risk_claim,
        "documentation_completeness": 0.95,
        "conservative_treatment_tried": 1,
        "criteria_met_ratio": 0.95,
        "prior_denial_history": 0,
    }

    high_risk_score = predict_denial_risk(pipeline, high_risk_claim)
    low_risk_score = predict_denial_risk(pipeline, low_risk_claim)

    assert high_risk_score > low_risk_score
