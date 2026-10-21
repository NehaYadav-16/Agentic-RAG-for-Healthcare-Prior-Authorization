"""Cost-savings estimator.

Translates automation of prior-auth review and appeal drafting into
dollar terms, using configurable, clearly-labeled assumptions (the
defaults are illustrative and sourced loosely from publicly reported
industry figures -- CAQH Index, AHA surveys -- and should be replaced
with an organization's own time-and-motion study before being used for
real budgeting).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostAssumptions:
    avg_manual_review_minutes: float = 25.0
    avg_manual_appeal_minutes: float = 45.0
    staff_hourly_wage_usd: float = 32.0
    automation_time_reduction_pct: float = 0.70
    appeal_rate: float = 0.20  # fraction of claims that end up needing an appeal


@dataclass
class SavingsEstimate:
    num_claims: int
    manual_hours: float
    automated_hours: float
    hours_saved: float
    manual_cost_usd: float
    automated_cost_usd: float
    savings_usd: float
    savings_pct: float


def estimate_savings(num_claims: int, assumptions: CostAssumptions | None = None) -> SavingsEstimate:
    a = assumptions or CostAssumptions()

    manual_minutes = num_claims * a.avg_manual_review_minutes
    manual_minutes += num_claims * a.appeal_rate * a.avg_manual_appeal_minutes
    manual_hours = manual_minutes / 60.0

    automated_hours = manual_hours * (1 - a.automation_time_reduction_pct)
    hours_saved = manual_hours - automated_hours

    manual_cost = manual_hours * a.staff_hourly_wage_usd
    automated_cost = automated_hours * a.staff_hourly_wage_usd
    savings = manual_cost - automated_cost
    savings_pct = (savings / manual_cost) if manual_cost else 0.0

    return SavingsEstimate(
        num_claims=num_claims,
        manual_hours=round(manual_hours, 2),
        automated_hours=round(automated_hours, 2),
        hours_saved=round(hours_saved, 2),
        manual_cost_usd=round(manual_cost, 2),
        automated_cost_usd=round(automated_cost, 2),
        savings_usd=round(savings, 2),
        savings_pct=round(savings_pct, 4),
    )


def project_annual_savings(monthly_claims: int, assumptions: CostAssumptions | None = None) -> SavingsEstimate:
    return estimate_savings(monthly_claims * 12, assumptions)
