from paig.cost_savings import CostAssumptions, estimate_savings


def test_savings_are_positive_under_default_assumptions():
    result = estimate_savings(num_claims=10_000)
    assert result.savings_usd > 0
    assert 0 < result.savings_pct < 1


def test_zero_automation_reduction_means_zero_savings():
    assumptions = CostAssumptions(automation_time_reduction_pct=0.0)
    result = estimate_savings(num_claims=1000, assumptions=assumptions)
    assert result.savings_usd == 0
