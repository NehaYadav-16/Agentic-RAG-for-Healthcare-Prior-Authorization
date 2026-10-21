"""The agentic pipeline: a sequence of small, inspectable steps (retrieve
-> check criteria -> score risk -> decide -> draft appeal if needed) that
together form the "agent." Each step appends to an audit trail so the
full reasoning chain is reviewable -- the explainability property that
matters most for regulated healthcare decisioning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.pipeline import Pipeline

from .appeal_generator import AppealContext, LetterBackend, get_default_backend
from .criteria_checker import CriterionResult, check_criteria, unmet_criteria
from .denial_model import predict_denial_risk
from .retriever import RetrievedChunk, TfidfRetriever

DENIAL_RISK_THRESHOLD = 0.5


@dataclass
class PriorAuthRequest:
    claim_id: str
    payer: str
    procedure_category: str
    patient_age: int
    documentation_completeness: float
    conservative_treatment_tried: int
    criteria_met_ratio: float
    prior_denial_history: int = 0
    peer_to_peer_requested: int = 0
    days_to_submit: int = 7
    clinical_query: str = ""

    def as_feature_dict(self) -> Dict:
        return {
            "procedure_category": self.procedure_category,
            "payer": self.payer,
            "patient_age": self.patient_age,
            "documentation_completeness": self.documentation_completeness,
            "conservative_treatment_tried": self.conservative_treatment_tried,
            "criteria_met_ratio": self.criteria_met_ratio,
            "prior_denial_history": self.prior_denial_history,
            "peer_to_peer_requested": self.peer_to_peer_requested,
            "days_to_submit": self.days_to_submit,
        }


@dataclass
class PriorAuthResult:
    claim_id: str
    decision: str  # "LIKELY_APPROVED" | "HIGH_DENIAL_RISK"
    denial_risk: float
    retrieved: List[RetrievedChunk]
    criteria: List[CriterionResult]
    appeal_letter: Optional[str]
    audit_trail: List[str] = field(default_factory=list)


class PriorAuthAgent:
    """Orchestrates the retrieve -> check -> score -> decide -> draft flow."""

    def __init__(
        self,
        retriever: TfidfRetriever,
        denial_model: Pipeline,
        letter_backend: Optional[LetterBackend] = None,
        risk_threshold: float = DENIAL_RISK_THRESHOLD,
    ):
        self.retriever = retriever
        self.denial_model = denial_model
        self.letter_backend = letter_backend or get_default_backend()
        self.risk_threshold = risk_threshold

    def run(self, request: PriorAuthRequest) -> PriorAuthResult:
        trail: List[str] = []

        query = request.clinical_query or request.procedure_category.replace("_", " ")
        retrieved = self.retriever.retrieve(
            query, top_k=4, procedure_category=request.procedure_category
        )
        trail.append(
            f"Retrieved {len(retrieved)} policy section(s) for '{request.procedure_category}': "
            + ", ".join(r.chunk.citation for r in retrieved)
        )

        criteria = check_criteria(retrieved, request.as_feature_dict() | {
            "criteria_met_ratio": request.criteria_met_ratio,
            "documentation_completeness": request.documentation_completeness,
            "conservative_treatment_tried": request.conservative_treatment_tried,
        })
        missing = unmet_criteria(criteria)
        trail.append(f"Criteria check: {len(criteria) - len(missing)}/{len(criteria)} satisfied")

        risk = predict_denial_risk(self.denial_model, request.as_feature_dict())
        trail.append(f"Denial-risk model score: {risk:.2%}")

        # The trained risk model is the primary decision signal; the
        # criteria checklist is explainability/audit context for the
        # reviewer and the appeal letter, not a second hard gate -- using
        # both as independent gates double-penalizes borderline claims.
        decision = "HIGH_DENIAL_RISK" if risk >= self.risk_threshold else "LIKELY_APPROVED"
        trail.append(f"Decision: {decision}")

        appeal_letter = None
        if decision == "HIGH_DENIAL_RISK":
            procedure_title = retrieved[0].chunk.title if retrieved else request.procedure_category
            ctx = AppealContext(
                claim_id=request.claim_id,
                payer=request.payer,
                procedure_title=procedure_title,
                patient_age=request.patient_age,
                unmet=missing,
                met=[c for c in criteria if c.met],
                denial_risk=risk,
            )
            appeal_letter = self.letter_backend.draft(ctx)
            trail.append(f"Appeal letter drafted via '{self.letter_backend.name}' backend")

        return PriorAuthResult(
            claim_id=request.claim_id,
            decision=decision,
            denial_risk=risk,
            retrieved=retrieved,
            criteria=criteria,
            appeal_letter=appeal_letter,
            audit_trail=trail,
        )
