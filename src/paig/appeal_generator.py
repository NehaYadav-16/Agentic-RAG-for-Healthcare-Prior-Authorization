"""Grounded appeal-letter drafting.

Two backends:

- `TemplateBackend` (default): fully deterministic, offline, zero API
  dependency. Fills a professional letter template using only the
  retrieved policy citations and criteria checklist -- it cannot
  hallucinate facts because it has no generative freedom.
- `AnthropicBackend` (optional): if `ANTHROPIC_API_KEY` is set and the
  `anthropic` package is installed, asks Claude to polish the same
  grounded facts into more natural prose, under a strict system prompt
  that forbids introducing new claims or citations. Falls back to the
  template automatically if unavailable.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from .criteria_checker import CriterionResult

SYSTEM_PROMPT = (
    "You are drafting a prior-authorization appeal letter for a healthcare "
    "provider's billing office. You MUST only use the facts, criteria, and "
    "policy citations given to you. Do not invent clinical facts, do not "
    "invent policy citations, and do not promise an outcome. Keep it to "
    "one page, professional tone, cite the given policy sections inline."
)


@dataclass
class AppealContext:
    claim_id: str
    payer: str
    procedure_title: str
    patient_age: int
    unmet: List[CriterionResult]
    met: List[CriterionResult]
    denial_risk: float


class LetterBackend(ABC):
    @abstractmethod
    def draft(self, ctx: AppealContext) -> str: ...


class TemplateBackend(LetterBackend):
    name = "template"

    def draft(self, ctx: AppealContext) -> str:
        met_lines = "\n".join(f"  - {c.description} (see {c.citation})" for c in ctx.met) or "  - None on file"
        unmet_lines = (
            "\n".join(f"  - {c.description} (see {c.citation})" for c in ctx.unmet) or "  - None outstanding"
        )
        return f"""Re: Appeal of Adverse Determination -- Claim {ctx.claim_id}
Payer: {ctx.payer}
Procedure: {ctx.procedure_title}
Patient Age: {ctx.patient_age}

To the Utilization Review Department:

This letter requests reconsideration of the adverse determination issued
for the above-referenced claim. Our internal grounded-RAG review of the
applicable medical policy identified the following:

Criteria satisfied by the current medical record:
{met_lines}

Criteria requiring additional documentation or clinical justification:
{unmet_lines}

Model-estimated denial risk at time of submission: {ctx.denial_risk:.0%}

We respectfully request that the above be reviewed in light of the
attached clinical documentation, and that a peer-to-peer review be
scheduled if the outstanding items cannot be resolved by record review
alone.

Sincerely,
Utilization Management / Prior Authorization Team
(Auto-drafted by PAIG -- Prior Authorization Intelligence & Grounding.
 This draft must be reviewed and signed off by a licensed reviewer
 before submission.)
"""


class AnthropicBackend(LetterBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5-5"):
        import anthropic  # local import: optional dependency

        self._client = anthropic.Anthropic()
        self._model = model

    def draft(self, ctx: AppealContext) -> str:
        template_draft = TemplateBackend().draft(ctx)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=700,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following grounded draft into more "
                        "natural professional prose, keeping every fact, "
                        "citation, and number exactly as given:\n\n"
                        f"{template_draft}"
                    ),
                }
            ],
        )
        return "".join(block.text for block in message.content if hasattr(block, "text"))


def get_default_backend() -> LetterBackend:
    """Picks the best available backend: Anthropic if configured, else the
    deterministic template (so the project always works out of the box)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicBackend()
        except Exception:
            return TemplateBackend()
    return TemplateBackend()
