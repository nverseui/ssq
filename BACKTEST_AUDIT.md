# SSQ research audit runner

The isolated `ssq_backtest_audit.py` research runner performs strict walk-forward evaluation, compares the current V3 generator to an explicit single-banker baseline and uniform random tickets, computes exact single-ticket theoretical odds, repeats random controls, reports paired moving-block bootstrap intervals, and Holm-adjusts the comparison p-values.

Run the long-window default (500 out-of-sample periods):

```bash
python ssq_backtest_audit.py --eval 500 --sets 5 --random-replicates 32
```

Periods are scored by a pool of worker processes (`--workers`, default `min(4, CPU count)`); pass `--workers 1` to run serially. Each period depends only on its own history prefix and an issue-seeded RNG, and results are collected in period order, so the report is identical for any worker count.

The report is written to `ssq_backtest_audit.json`. Historical data must contain at least 200 validated draws, and the requested window must leave at least 50 training draws. All schemes are evaluated on the same minimum actual ticket count for each target period; generated shortfalls are recorded separately and cost/ROI use only tickets actually evaluated.

The single-banker comparator uses the current code's `generate_v2_baseline_tickets` implementation. It is an explicit new baseline, **not a faithful historical V2 reconstruction**, because the original V2 implementation is not present in this repository. The random comparator uniformly samples legal SSQ tickets and does not apply the V3 shape filters; therefore it compares with ordinary random play, not a matched-shape control. Ablations compare overlap-cap, same-tail and blue selection variants against the strict overlap-cap baseline.

Prize returns use fixed estimates in `ssq_master_v3.eval_prize`. First and second prizes are floating in real play, so ROI is only an estimate. Read the confidence intervals and Holm-adjusted p-values; do not infer a future predictive advantage from raw historical results. The audit calibrator is research-only and does not silently alter production prediction code.

## Calibration residual centering

`StableAuditCalibrator.calibrate` centres the residual before it tilts the base estimate:

```python
residual = residual - residual.mean()
calibrated = engine.norm_expected_6(np.clip(base + self.lr * residual * engine.PICK, .02, None))
```

Feature scores are max-normalised, so their mean is around 0.5 while the empirical red-ball hit rate averages 6/33 ≈ 0.18. The raw residual therefore carried a large negative bias (about −0.28). Adding it uncentred pushed every calibrated probability below the 0.02 clip floor, collapsing the ranking to a constant. The downstream effect was severe: `high` became `1..12`, every V3 candidate failed the `sum >= 75` shape rule, and 475 of 500 audited periods generated zero tickets, leaving only 80 tickets in the comparison.

Centring keeps the residual's relative tilt and restores a non-degenerate ranking. After the fix the same 500-period run evaluates 2467 tickets with full 5-ticket periods in 475 of 500 periods. Reports produced before this fix (including a 0-ticket-dominated `ssq_backtest_audit.json`) are not comparable with current output, and their summary statistics should be discarded.
