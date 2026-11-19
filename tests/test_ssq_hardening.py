import ssq_hardening as hardening


def test_largest_remainder_tail_quota_is_exact():
    quota = hardening.largest_remainder_tail_quota(
        {3: 0.05, 4: 0.33, 5: 0.50, 6: 0.12},
        n_sets=15,
    )
    assert sum(quota.values()) == 15
    assert set(quota) == {3, 4, 5, 6}


def test_largest_remainder_tail_quota_handles_extreme_ticket_counts():
    for n in (1, 2, 5, 20):
        quota = hardening.largest_remainder_tail_quota(
            {3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0},
            n_sets=n,
        )
        assert sum(quota.values()) == n
        assert quota[5] == n


def test_stable_softmax_is_normalized_and_degenerate_safe():
    weights = hardening.stable_softmax([1e300, 1e300, -1e300])
    assert abs(float(weights.sum()) - 1.0) < 1e-12

    equal = hardening.stable_softmax([float("nan"), float("inf")])
    assert abs(float(equal.sum()) - 1.0) < 1e-12


def test_centered_residual_keeps_relative_direction():
    centered = hardening.centered_residual_tilt([1.0, 2.0, 3.0])
    assert abs(float(centered.mean())) < 1e-12
    assert centered[0] < 0 < centered[-1]


def test_exact_overlap_portfolio_never_relaxes():
    candidates = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16],
    ]
    portfolio = hardening.exact_overlap_portfolio(candidates, n_sets=3, max_overlap=3)
    assert len(portfolio) == 2


def test_validate_portfolio_reports_overlap_and_duplicate():
    result = hardening.validate_portfolio(
        red_combos=[[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]],
        blues=[1, 2],
        n_sets=2,
        max_overlap=3,
    )
    assert not result["valid"]
