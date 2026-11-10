# SSQ research audit runner

This research harness is isolated from the ticket-generation CLI. It borrows the useful evaluation practices from open-source SSQ/lottery repositories: strict walk-forward evaluation, a uniform-random null, exact single-ticket theoretical odds, repeated random seeds, paired moving-block bootstrap intervals, and Holm correction for multiple ablation comparisons.

## Run

```bash
python ssq_backtest_audit.py --eval 30 --sets 5 --random-replicates 32
```

The script uses the existing official-data loader and requires at least 200 validated historical draws. It writes `ssq_backtest_audit.json` with period-level details, actual ticket counts and shortfalls, wins, fixed-prize returns, costs, ROI, random-null distribution, theoretical probabilities, bootstrap intervals, and Holm-adjusted p-values.

## Ablations

The report compares the strict final overlap-cap baseline with: no pairwise overlap cap, no same-tail restriction, and a fixed hot-blue selection. It reports missed tickets and maximum pairwise red overlap. The `no_overlap_cap` variant is not to be interpreted as a Jaccard improvement; the production condition is a **red-number intersection cap** (not Jaccard similarity). The baseline post-filters generated tickets and reports a shortfall rather than claiming the cap held when it did not.

## Interpretation and caveats

- The included V3 generator and its shape filters are heuristic. No gain is presumed.
- The audit uses `StableAuditCalibrator`, a research-only calibration path that normalizes inverse-MSE weights directly to avoid overflow/NaN. This does **not** silently change the production `OptimalCalibrator`; promote a calibration change only after a separately reviewed code change and regression tests.
- The random null is uniform over legal SSQ combinations, without the V3 shape filters. That is appropriate for the question “does the filtered strategy beat an ordinary random ticket?” but is not a matched-shape control. A future study can add a matched-constraint random null.
- Fixed prize values follow `eval_prize`; first/second prizes are variable in practice, so reported ROI is an estimate, not actual historic payout or a guarantee.
- A small test window has weak power, especially for rare large prizes. Read confidence intervals and corrected p-values; do not conclude predictive advantage from raw win rate or a lucky payout.
- `period_details` make every period and each ablation auditable. Exact theoretical probabilities are per-ticket; multi-ticket probabilities depend on ticket overlap and are evaluated empirically against the random null.

The conceptual sources reviewed for this approach are [`alf1ei/lottery-backtest`](https://github.com/alf1ei/lottery-backtest) (walk-forward/random/theoretical baseline) and [`ioi-studio/IOI-LottoLab`](https://github.com/ioi-studio/IOI-LottoLab) (seed robustness and paired block bootstrap). No third-party predictor code is copied into this repository.
