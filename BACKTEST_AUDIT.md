# SSQ research audit runner

The isolated `ssq_backtest_audit.py` research runner performs strict walk-forward evaluation, compares the current V3 generator to an explicit single-banker baseline and uniform random tickets, computes exact single-ticket theoretical odds, repeats random controls, reports paired moving-block bootstrap intervals, and Holm-adjusts the comparison p-values.

Run the long-window default (500 out-of-sample periods):

```bash
python ssq_backtest_audit.py --eval 500 --sets 5 --random-replicates 32
```

The report is written to `ssq_backtest_audit.json`. Historical data must contain at least 200 validated draws, and the requested window must leave at least 50 training draws. All schemes are evaluated on the same minimum actual ticket count for each target period; generated shortfalls are recorded separately and cost/ROI use only tickets actually evaluated.

The single-banker comparator uses the current code's `generate_v2_baseline_tickets` implementation. It is an explicit new baseline, **not a faithful historical V2 reconstruction**, because the original V2 implementation is not present in this repository. The random comparator uniformly samples legal SSQ tickets and does not apply the V3 shape filters; therefore it compares with ordinary random play, not a matched-shape control. Ablations compare overlap-cap, same-tail and blue selection variants against the strict overlap-cap baseline.

Prize returns use fixed estimates in `ssq_master_v3.eval_prize`. First and second prizes are floating in real play, so ROI is only an estimate. Read the confidence intervals and Holm-adjusted p-values; do not infer a future predictive advantage from raw historical results. The audit calibrator is research-only and does not silently alter production prediction code.
