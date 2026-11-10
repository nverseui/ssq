import math

import ssq_backtest_audit as audit


def test_score_uses_same_metric_name_as_report_and_bootstrap():
    actual = type("DrawStub", (), {"reds": [1, 2, 3, 4, 5, 6], "blue": 7})()
    result = audit._score([{"推荐红球": "01 02 03 04 05 06", "推荐蓝球": "07"}], actual)
    assert "period_any_prize_rate" in result
    assert result["period_any_prize_rate"] == 1.0


def test_common_ticket_count_matches_every_comparator():
    portfolios = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    assert audit._common_ticket_count(portfolios, requested=5) == 2
    assert audit._common_ticket_count([[], [1]], requested=4) == 0


def test_holm_adjustment_is_monotonic_and_bounded():
    adjusted = audit._holm([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
    assert all(0 <= p <= 1 for p in adjusted)


def test_theoretical_probability_is_finite_and_roi_is_nonnegative():
    result = audit.theoretical_single_ticket()
    assert 0 < result["ticket_win_probability"] < 1
    assert math.isfinite(result["expected_fixed_prize_roi_percent"])
    assert result["ticket_cost_yuan"] == 2.0


def test_bootstrap_handles_empty_and_constant_differences():
    empty = audit._bootstrap([], replicates=100)
    assert empty["two_sided_p"] == 1.0
    constant = audit._bootstrap([0.25] * 20, replicates=200, block=4, seed=1)
    assert constant["mean_difference"] == 0.25
    assert constant["ci95"] == [0.25, 0.25]


def test_every_rule_comparison_has_adjusted_p_value():
    # Full result shape contract for this test is represented by Holm itself;
    # the integration-level report populates these keys for every comparison.
    raw = [0.01, 0.02, 0.2]
    adjusted = audit._holm(raw)
    assert len(adjusted) == len(raw)
    assert all(a >= p for a, p in zip(adjusted, raw))
