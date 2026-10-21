"""Streamlit demo UI for PAIG.

    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from paig.data_gen import PAYERS, PROCEDURE_CATEGORIES  # noqa: E402
from paig.denial_model import load_model  # noqa: E402
from paig.pipeline import PriorAuthAgent, PriorAuthRequest  # noqa: E402
from paig.policy_store import load_policies  # noqa: E402
from paig.retriever import TfidfRetriever  # noqa: E402

st.set_page_config(page_title="PAIG - Prior Authorization Intelligence & Grounding", layout="wide")


@st.cache_resource
def load_agent() -> PriorAuthAgent:
    chunks = load_policies(ROOT / "data" / "policies")
    retriever = TfidfRetriever(chunks)
    model_path = ROOT / "models" / "denial_model.joblib"
    model = load_model(model_path)
    return PriorAuthAgent(retriever=retriever, denial_model=model)


st.title("PAIG: Prior Authorization Intelligence & Grounding")
st.caption(
    "Agentic RAG for prior-authorization decisioning and denial-appeal drafting, "
    "grounded in payer medical policy documents. Synthetic data only -- research/demo project."
)

try:
    agent = load_agent()
except SystemExit as e:
    st.error(str(e))
    st.stop()
except FileNotFoundError:
    st.error("Model not found. Run `python scripts/generate_data.py` then `python scripts/train_model.py` first.")
    st.stop()

with st.form("claim_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        claim_id = st.text_input("Claim ID", value="CLM-UI-001")
        payer = st.selectbox("Payer", PAYERS)
        procedure_category = st.selectbox("Procedure category", PROCEDURE_CATEGORIES)
    with col2:
        patient_age = st.slider("Patient age", 18, 90, 45)
        documentation_completeness = st.slider("Documentation completeness", 0.0, 1.0, 0.7)
        criteria_met_ratio = st.slider("Policy criteria met (clinical review estimate)", 0.0, 1.0, 0.6)
    with col3:
        conservative_treatment_tried = st.checkbox("Conservative treatment tried", value=True)
        prior_denial_history = st.checkbox("Prior denial history for this patient")
        peer_to_peer_requested = st.checkbox("Peer-to-peer already requested")

    clinical_query = st.text_area(
        "Clinical summary (used to retrieve the relevant policy sections)",
        value="Patient with persistent low back pain after 8 weeks of physical therapy.",
    )
    submitted = st.form_submit_button("Run PAIG agent")

if submitted:
    request = PriorAuthRequest(
        claim_id=claim_id,
        payer=payer,
        procedure_category=procedure_category,
        patient_age=patient_age,
        documentation_completeness=documentation_completeness,
        conservative_treatment_tried=int(conservative_treatment_tried),
        criteria_met_ratio=criteria_met_ratio,
        prior_denial_history=int(prior_denial_history),
        peer_to_peer_requested=int(peer_to_peer_requested),
        clinical_query=clinical_query,
    )
    result = agent.run(request)

    st.subheader("Decision")
    badge = "🟢 LIKELY APPROVED" if result.decision == "LIKELY_APPROVED" else "🔴 HIGH DENIAL RISK"
    st.markdown(f"### {badge}")
    st.metric("Model denial-risk score", f"{result.denial_risk:.1%}")

    st.subheader("Audit trail (explainability log)")
    for step in result.audit_trail:
        st.write(f"- {step}")

    st.subheader("Retrieved policy citations")
    for r in result.retrieved:
        with st.expander(f"{r.chunk.citation}  (score={r.score:.3f})"):
            st.write(r.chunk.text)

    st.subheader("Criteria checklist")
    for c in result.criteria:
        icon = "✅" if c.met else "❌"
        st.write(f"{icon} {c.description} — *{c.citation}*")

    if result.appeal_letter:
        st.subheader("Drafted appeal letter")
        st.text_area("Appeal letter draft", result.appeal_letter, height=350)

st.divider()
st.caption(
    "⚠️ Research/demo project using synthetic data. Not a medical device and not legal advice. "
    "Any real deployment requires a licensed clinical/UM reviewer in the loop and validation "
    "against your organization's actual payer contracts and policies."
)
