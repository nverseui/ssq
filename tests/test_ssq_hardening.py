import math

import numpy as np
import ssq_hardening as hardening


def test_largest_remainder_tail_quota_is_exact():
    quota = hardening.largest_remainder_tail_quota(
        {3: 0.05, 4: 0.33, 5: 0.50, 6: 0.12}, n_sets=15
    )
    assert sum(quota.values()) == 15
    assert set(quota) == {3, 4, 5, 6}


def test_largest_remainder_handles_zero_and_invalid_probabilities():
    for n in (1, 2, 5, 20):
        quota = hardening.largest_remainder_tail_quota(
            {3: 0.0, 4: float("nan"), 5: 1.0, 6: float("inf")}, n_sets=n
        )
        assert sum(quota.values()) == n
        assert quota[5] == n


def test_stable_softmax_is_finite_and_normalized():
    weights = hardening.stable_softmax([1e300, 1e300, -1e300])
    assert np.isfinite(weights).all()
    assert math.isclose(float(weights.sum()), 1.0)


def test_centered_residual_has_zero_mean_and_preserves_order():
    centered = hardening.centered_residual_tilt(np.array([1.0, 2.0, 3.0]))
    assert abs(float(centered.mean())) < 1e-12
    assert centered[0] < 0 < centered[-1]


def test_exact_overlap_portfolio_does_not_relax_constraint():
    candidates = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16],
    ]
    portfolio = hardening.exact_overlap_portfolio(candidates, n_sets=3, max_overlap=2)
    assert len(portfolio) == 2
    assert all(len(set(a) & set(b)) <= 2
               for i, a in enumerate(portfolio) for b in portfolio[i + 1:])


def test_validator_flags_overlap_and_red_duplicate():
    result = hardening.validate_portfolio(
        red_combos=[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7, 8]],
        blues=[1, 2], n_sets=2, max_overlap=3
    )
    assert not result["valid"]
    assert any("overlap" in issue for issue in result["issues"])


def test_hardened_calibrator_outputs_finite_six_expected_probability():
    import ssq_master_v3 as engine

    draws = [engine.Draw(2026000 + i, [1, 5, 11, 17, 24, 31], (i % 16) + 1)
             for i in range(30)]
    # Use varied valid draws so feature maps are exercised without external data.
    draws = [engine.Draw(2026000 + i,
                         sorted({((i * 3 + j * 5) % 33) + 1 for j in range(6)}),
                         (i % 16) + 1) for i in range(30)]
    calibrated, _, _, weights = hardening.HardenedCalibrator().calibrate(draws)
    assert np.isfinite(calibrated).all()
    assert math.isclose(float(calibrated.sum()), 6.0, rel_tol=1e-9)
    assert all(math.isfinite(v) for v in weights.values())
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9)
