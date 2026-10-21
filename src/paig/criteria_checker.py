"""Maps structured claim fields onto policy criteria and produces a
citation-grounded checklist -- the explainability layer that lets a
reviewer see *why* a claim is likely to be approved or denied."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .retriever import RetrievedChunk


@dataclass
class CriterionResult:
    description: str
    met: bool
    citation: str


def check_criteria(
    retrieved: List[RetrievedChunk],
    claim: Dict,
) -> List[CriterionResult]:
    """Heuristic, citation-grounded criteria check.

    Each policy's "Medical Necessity Criteria" section is retrieved and
    paired with the claim's structured fields. This is intentionally
    simple/rule-based rather than free-generated, so every result is
    traceable to a specific policy citation -- the property insurers and
    auditors actually need from an automated decision system.
    """
    results: List[CriterionResult] = []
    criteria_met_ratio = float(claim.get("criteria_met_ratio", 0.0))
    documentation_completeness = float(claim.get("documentation_completeness", 0.0))
    conservative_treatment_tried = bool(claim.get("conservative_treatment_tried", 0))

    necessity_chunks = [r for r in retrieved if "Medical Necessity" in r.chunk.heading]
    documentation_chunks = [r for r in retrieved if "Documentation" in r.chunk.heading]

    for r in necessity_chunks:
        met = criteria_met_ratio >= 0.6
        results.append(
            CriterionResult(
                description=f"Medical necessity criteria for {r.chunk.title}",
                met=met,
                citation=r.chunk.citation,
            )
        )

    for r in documentation_chunks:
        met = documentation_completeness >= 0.6
        results.append(
            CriterionResult(
                description=f"Documentation requirements for {r.chunk.title}",
                met=met,
                citation=r.chunk.citation,
            )
        )

    if necessity_chunks:
        results.append(
            CriterionResult(
                description="Conservative treatment attempted prior to request",
                met=conservative_treatment_tried,
                citation=necessity_chunks[0].chunk.citation,
            )
        )

    return results


def unmet_criteria(results: List[CriterionResult]) -> List[CriterionResult]:
    return [r for r in results if not r.met]
