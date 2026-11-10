#!/usr/bin/env python3
"""Auditable SSQ walk-forward comparison: heuristic, explicit baseline and random.

This research runner is separate from the production predictor. At each held-out
period it uses only previous draws, generates candidate portfolios, then evaluates
every comparator on the same per-period ticket count. Prize return uses the
project's fixed estimates; it is not actual historical payout or a promise.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Dict, Sequence

import numpy as np
import ssq_master_v3 as engine


class StableAuditCalibrator(engine.OptimalCalibrator):
    """Audit-only calibrator with finite, normalized inverse-loss weights."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_feature_probabilities(draws[:t]) is a pure function of the prefix
        # length t: walk-forward reuses overlapping prefixes across calibration
        # rounds and periods, so memoize per calibrator instance (one per audit).
        self._prob_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _feature_probs(self, draws, t: int):
        cached = self._prob_cache.get(t)
        if cached is None:
            cached = engine.compute_feature_probabilities(draws[:t])
            self._prob_cache[t] = cached
        return cached

    def calibrate(self, draws):
        n = len(draws)
        start = max(20, n - self.window)
        rounds = n - start
        losses = np.full(len(self.features), 1e-4, dtype=float)
        residual = np.zeros(engine.RED_N, dtype=float)
        if rounds >= 3:
            decay = np.exp(np.linspace(-0.8, 0.0, rounds))
            for k, t in enumerate(range(start, n)):
                maps = self._feature_probs(draws, t)
                actual = np.array([float(r in draws[t].reds) for r in range(1, engine.RED_N + 1)])
                scores = [maps[name] / max(float(np.max(maps[name])), 1e-12) for name in self.features]
                for i, score in enumerate(scores):
                    losses[i] += float(np.mean((actual - score) ** 2)) * decay[k]
                residual += (actual - np.mean(scores, axis=0)) * decay[k]
            residual /= rounds
        inverse = 1 / np.maximum(losses, 1e-12)
        weights = inverse / inverse.sum() if np.isfinite(inverse).all() and inverse.sum() else np.full(len(self.features), 1 / len(self.features))
        if self.miss_tilt > 0 and rounds >= 3:
            missed_counts = Counter()
            total_missed = 0
            for t in range(start, n):
                maps = self._feature_probs(draws, t)
                blend = sum(weights[i] * maps[name] / max(float(np.max(maps[name])), 1e-12)
                            for i, name in enumerate(self.features))
                missed = set(draws[t].reds) - set((np.argsort(-blend) + 1)[:12].tolist())
                total_missed += len(missed)
                for r in missed:
                    relative = {name: maps[name][r - 1] / max(float(np.max(maps[name])), 1e-12)
                                for name in self.features}
                    missed_counts[min(relative, key=relative.get)] += 1
            if total_missed:
                weights *= np.array([1 + self.miss_tilt * missed_counts[name] / total_missed
                                     for name in self.features])
                weights /= weights.sum()
        current = self._feature_probs(draws, n)
        logits = [engine.logit(np.clip(current[name] / max(float(np.max(current[name])), 1e-12) * .82 + .03,
                                       .02, .88)) for name in self.features]
        base = engine.norm_expected_6(engine.expit(sum(weights[i] * logits[i] for i in range(len(self.features)))))
        # Feature scores are max-normalised (mean ~0.5) while the empirical hit
        # rate averages 6/33, so the raw residual has a large negative bias.
        # Adding it uncentred pushed every estimate under the floor and collapsed
        # the ranking to a constant; centre it so it only tilts the base.
        residual = residual - residual.mean() if rounds >= 3 else residual
        calibrated = engine.norm_expected_6(np.clip(base + self.lr * residual * engine.PICK, .02, None))
        gaps = np.zeros(engine.RED_N)
        for r in range(1, engine.RED_N + 1):
            for draw in reversed(draws):
                if r in draw.reds:
                    break
                gaps[r - 1] += 1
        return calibrated, current["遗漏反弹"], gaps, dict(zip(self.features, weights))


def theoretical_single_ticket() -> Dict:
    """Exact per-ticket prize probabilities under uniform independent draws."""
    total_red = math.comb(33, 6)
    probability = Counter()
    expected_return = 0.0
    for red_hits in range(7):
        p_red = math.comb(6, red_hits) * math.comb(27, 6 - red_hits) / total_red
        for blue_hit, p_blue in ((0, 15 / 16), (1, 1 / 16)):
            prize, amount = engine.eval_prize(red_hits, blue_hit)
            if prize:
                probability[prize] += p_red * p_blue
                expected_return += p_red * p_blue * amount
    return {"ticket_win_probability": float(sum(probability.values())),
            "expected_fixed_prize_yuan": float(expected_return), "ticket_cost_yuan": 2.0,
            "expected_fixed_prize_roi_percent": float(expected_return / 2 * 100),
            "prize_probabilities": dict(probability),
            "assumption": "uniform draws; fixed prize estimates from engine.eval_prize; variable first/second prizes are approximated"}


def _tickets_from_combos(combos, blues):
    return [{"推荐红球": " ".join(f"{x:02d}" for x in combo),
             "推荐蓝球": f"{blues[i % len(blues)]:02d}"} for i, combo in enumerate(combos)]


def _generate_v3(inputs, actual, n_sets, variant, cand_cache):
    high, mid, out, hot, gold, cold, last, regime, patterns, quota = inputs
    # Candidate pools depend only on (issue, enforce_same_tail); share them
    # across ablation variants — generation rng is seeded by issue, so results
    # are identical to generating per variant, at a fraction of the cost.
    same_tail = variant != "no_same_tail_filter"
    key = (actual.issue, same_tail)
    if key not in cand_cache:
        cand_cache[key] = engine.generate_v3_candidate_combos(
            high, mid, out, last, actual.issue, regime, patterns,
            n_sets=n_sets, enforce_same_tail=same_tail)
    raw_main, raw_hedge, b_primary, b_secondary = cand_cache[key]
    kwargs = {"n_sets": n_sets, "diversity_limit": 3,
              "include_blue_triad": True, "require_min_jaccard": True}
    if variant == "no_overlap_cap":
        kwargs.update(diversity_limit=6, require_min_jaccard=False)
    generated = engine.select_v3_tickets_from_candidates(
        raw_main, raw_hedge, hot, gold, cold, b_primary, b_secondary, quota, **kwargs)
    combos = [sorted(int(x) for x in ticket["推荐红球"].split()) for ticket in generated]
    # V3 generator's internal fallback could relax overlap to four; enforce the
    # baseline cap here and allow the resulting shortfall to remain visible.
    if variant == "baseline":
        accepted = []
        for combo in combos:
            if engine.is_diverse_enough(combo, accepted, max_overlap=3):
                accepted.append(combo)
        combos = accepted
    blues = [hot[0], gold[0], hot[1] if len(hot) > 1 else hot[0], cold[0],
             gold[1] if len(gold) > 1 else gold[0]]
    if variant == "fixed_hot_blue":
        blues = [hot[0]]
    return _tickets_from_combos(combos, blues)


def _uniform_tickets(n_sets, seed):
    rng = random.Random(seed)
    result, seen = [], set()
    while len(result) < n_sets:
        red = tuple(sorted(rng.sample(range(1, 34), 6)))
        blue = rng.randint(1, 16)
        if (red, blue) in seen:
            continue
        seen.add((red, blue))
        result.append({"推荐红球": " ".join(f"{x:02d}" for x in red), "推荐蓝球": f"{blue:02d}"})
    return result


def _score(tickets, actual):
    wins, reward, prizes = 0, 0, Counter()
    for ticket in tickets:
        reds = [int(x) for x in ticket["推荐红球"].split()]
        blue = int(ticket["推荐蓝球"].split()[0])
        name, amount = engine.eval_prize(len(set(reds) & set(actual.reds)), int(blue == actual.blue))
        if name:
            wins += 1
            reward += amount
            prizes[name] += 1
    count = len(tickets)
    return {"tickets": count, "winning_tickets": wins, "reward_yuan": reward,
            "cost_yuan": count * 2, "ticket_win_rate": wins / count if count else 0.0,
            "roi_percent": reward / (count * 2) * 100 if count else 0.0,
            "period_any_prize_rate": float(wins > 0), "prizes": dict(prizes)}


def _common_ticket_count(portfolios, requested):
    """Equalize all compared schemes to the same actual tickets for this period."""
    return min([requested] + [len(tickets) for tickets in portfolios])


def _bootstrap(differences, replicates=4000, block=10, seed=982451653):
    n = len(differences)
    if not n:
        return {"mean_difference": 0.0, "ci95": [0.0, 0.0], "two_sided_p": 1.0,
                "replicates": replicates, "block_length": block}
    rng = random.Random(seed)
    # Pre-materialise every possible moving-block window once; the resampling
    # loop then only copies list slices instead of rebuilding them element-wise.
    repeats = 2 + block // max(n, 1)
    doubled = list(differences) * repeats
    segments = [doubled[start:start + block] for start in range(n)]
    samples = []
    for _ in range(replicates):
        chosen = []
        while len(chosen) < n:
            start = rng.randrange(n)
            chosen.extend(segments[start])
        samples.append(fmean(chosen[:n]))
    samples.sort()
    lower = (sum(x <= 0 for x in samples) + 1) / (replicates + 1)
    upper = (sum(x >= 0 for x in samples) + 1) / (replicates + 1)
    return {"mean_difference": fmean(differences),
            "ci95": [samples[int(.025 * replicates)], samples[min(replicates - 1, int(.975 * replicates))]],
            "two_sided_p": min(1.0, 2 * min(lower, upper)),
            "replicates": replicates, "block_length": block}


def _bootstrap_vs_random(left, random_runs, metric, replicates=4000, block=10, seed=982451653):
    """Bootstrap paired time blocks while resampling the random replicate null."""
    n = len(left)
    if not n or not random_runs:
        return _bootstrap([], replicates, block, seed)
    rng = random.Random(seed)
    # Pre-compute each replicate's moving-block contrasts so the resampling loop
    # only does list copies. Random calls stay in the same order, so the test is
    # numerically identical to recomputing left[i] - rep[i] element by element.
    left_vals = np.array([row[metric] for row in left], dtype=float)
    reps = [np.array([row[metric] for row in run], dtype=float) for run in random_runs]
    repeats = 2 + block // max(n, 1)
    tables = [[np.tile(left_vals - arr, repeats)[start:start + block].tolist()
               for start in range(n)] for arr in reps]
    samples = []
    for _ in range(replicates):
        contrasts = []
        while len(contrasts) < n:
            start = rng.randrange(n)
            rep = rng.randrange(len(random_runs))
            contrasts.extend(tables[rep][start])
        samples.append(fmean(contrasts[:n]))
    samples.sort()
    lower = (sum(x <= 0 for x in samples) + 1) / (replicates + 1)
    upper = (sum(x >= 0 for x in samples) + 1) / (replicates + 1)
    return {"mean_difference": fmean(left[i][metric] - fmean(run[i][metric] for run in random_runs)
                                      for i in range(n)),
            "ci95": [samples[int(.025 * replicates)], samples[min(replicates - 1, int(.975 * replicates))]],
            "two_sided_p": min(1.0, 2 * min(lower, upper)), "replicates": replicates,
            "block_length": block, "random_replicates_resampled": len(random_runs)}


def _holm(pvalues):
    """Holm step-down adjusted p-values, mapped back to original order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _prepare(train, n_sets, calibrator):
    regime = engine.detect_regime_invert(train)
    patterns = {d.zones_3 for d in train[-25:]} if len(train) >= 25 else None
    calpr, rbpr, _, _ = calibrator.calibrate(train)
    ranked = np.argsort(-calpr) + 1
    high, mid = ranked[:12].tolist(), ranked[12:22].tolist()
    out = sorted(ranked[22:].tolist(), key=lambda x: rbpr[x - 1], reverse=True)[:7]
    hot, gold, cold, _, _ = engine.evaluate_blue_triad(train)
    _, pred = engine.analyze_tail_markov_transition(train)
    fatigue, _ = engine.detect_tail_fatigue_inversion(train)
    quota = engine.compute_dynamic_tail_quota(pred, fatigue, n_sets)
    return high, mid, out, hot, gold, cold, train[-1], regime, patterns, quota


_WORKER_STATE = None


def _init_worker(draws, n_sets, random_replicates):
    """Per-process state: draws, ticket count, and a private calibrator cache."""
    global _WORKER_STATE
    _WORKER_STATE = (draws, n_sets, random_replicates,
                     StableAuditCalibrator(window=20, lr=.35, miss_tilt=engine.DEFAULT_TILT), {})


def _audit_period(idx):
    """Score one held-out period. Deterministic: seeds depend only on the issue."""
    draws, n_sets, random_replicates, calibrator, cand_cache = _WORKER_STATE
    actual = draws[idx]
    inputs = _prepare(draws[:idx], n_sets, calibrator)
    high, mid, out, hot, gold, cold, last, regime, patterns, _ = inputs
    ablations = ["baseline", "no_overlap_cap", "no_same_tail_filter", "fixed_hot_blue"]
    names = ["single_banker_baseline"] + ablations
    portfolios = {name: _generate_v3(inputs, actual, n_sets, name, cand_cache) for name in ablations}
    portfolios["single_banker_baseline"] = engine.generate_v2_baseline_tickets(
        high, mid, out, hot, gold, cold, last, actual.issue, regime, patterns, n_sets)
    common = _common_ticket_count(list(portfolios.values()), n_sets)
    period_detail = {"issue": actual.issue, "requested_tickets": n_sets,
                     "common_evaluated_tickets": common, "generated_tickets": {}, "schemes": {}}
    rows = {}
    for name in names:
        generated = portfolios[name]
        period_detail["generated_tickets"][name] = len(generated)
        evaluated = generated[:common]
        result = _score(evaluated, actual)
        result["missing_tickets"] = n_sets - len(generated)
        if name == "baseline":
            result["overlap_max"] = max((len(set(a["推荐红球"].split()) & set(b["推荐红球"].split()))
                                         for j, a in enumerate(evaluated) for b in evaluated[j + 1:]), default=0)
        rows[name] = result
        period_detail["schemes"][name] = result
    random_rows = []
    for rep in range(random_replicates):
        random_portfolio = _uniform_tickets(common, seed=actual.issue * 1009 + rep)
        random_result = _score(random_portfolio, actual)
        random_result["missing_tickets"] = n_sets - common
        random_rows.append(random_result)
    return period_detail, rows, random_rows


def _aggregate(rows):
    tickets = sum(row["tickets"] for row in rows)
    cost = sum(row["cost_yuan"] for row in rows)
    reward = sum(row["reward_yuan"] for row in rows)
    return {"periods": len(rows), "actual_tickets": tickets,
            "missing_tickets": sum(row["missing_tickets"] for row in rows),
            "winning_tickets": sum(row["winning_tickets"] for row in rows),
            "ticket_win_rate": sum(row["winning_tickets"] for row in rows) / tickets if tickets else 0.0,
            "period_any_prize_rate": fmean(row["period_any_prize_rate"] for row in rows) if rows else 0.0,
            "reward_yuan": reward, "cost_yuan": cost,
            "roi_percent": reward / cost * 100 if cost else 0.0,
            "prize_counts": dict(sum((Counter(row["prizes"]) for row in rows), Counter()))}


def run_audit(draws: Sequence, periods=500, n_sets=5, random_replicates=32, bootstrap_replicates=4000,
              workers=1):
    if len(draws) < engine.MIN_HISTORY_DRAWS:
        raise ValueError(f"need at least {engine.MIN_HISTORY_DRAWS} valid historical draws; got {len(draws)}")
    if periods < 10 or periods > len(draws) - 50:
        raise ValueError(f"periods must be between 10 and {len(draws) - 50}, got {periods}")
    if not 1 <= n_sets <= 20 or random_replicates < 2 or bootstrap_replicates < 100:
        raise ValueError("sets must be 1..20, random-replicates >=2, bootstrap-replicates >=100")

    start = len(draws) - periods
    ablations = ["baseline", "no_overlap_cap", "no_same_tail_filter", "fixed_hot_blue"]
    names = ["single_banker_baseline"] + ablations
    rows_by_name = {name: [] for name in names}
    random_runs = [[] for _ in range(random_replicates)]
    detail = []

    # Each period depends only on its own history prefix and its issue-seeded RNG,
    # so periods can be scored in parallel without changing any result. The map
    # preserves index order, keeping aggregation bit-for-bit deterministic.
    indices = list(range(start, len(draws)))
    if workers > 1 and len(indices) >= 20:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, len(indices)),
                      initializer=_init_worker,
                      initargs=(draws, n_sets, random_replicates)) as pool:
            results = pool.map(_audit_period, indices, chunksize=4)
    else:
        _init_worker(draws, n_sets, random_replicates)
        results = [_audit_period(idx) for idx in indices]

    for period_detail, rows, random_rows in results:
        detail.append(period_detail)
        for name in names:
            rows_by_name[name].append(rows[name])
        for rep, random_result in enumerate(random_rows):
            random_runs[rep].append(random_result)

    # Per-period mean over random replicates; all portfolios have equal ticket count.
    random_mean_rows = []
    for i in range(periods):
        source = [run[i] for run in random_runs]
        random_mean_rows.append({"tickets": source[0]["tickets"],
            "missing_tickets": source[0]["missing_tickets"],
            "winning_tickets": fmean(row["winning_tickets"] for row in source),
            "reward_yuan": fmean(row["reward_yuan"] for row in source),
            "cost_yuan": source[0]["cost_yuan"],
            "ticket_win_rate": fmean(row["ticket_win_rate"] for row in source),
            "roi_percent": fmean(row["roi_percent"] for row in source),
            "period_any_prize_rate": fmean(row["period_any_prize_rate"] for row in source), "prizes": {}})

    summary = {name: _aggregate(rows) for name, rows in rows_by_name.items()}
    summary["uniform_random_mean"] = _aggregate(random_mean_rows)
    random_roi = [_aggregate(run)["roi_percent"] for run in random_runs]
    summary["uniform_random_null_roi_percent"] = {"replicates": random_replicates,
        "mean": fmean(random_roi), "q025": float(np.quantile(random_roi, .025)),
        "median": float(np.quantile(random_roi, .5)), "q975": float(np.quantile(random_roi, .975))}

    tests = []
    metrics = ("ticket_win_rate", "roi_percent", "period_any_prize_rate")
    for metric in metrics:
        tests.append({"comparison": f"baseline - random: {metric}", "metric": metric,
                      **_bootstrap_vs_random(rows_by_name["baseline"], random_runs, metric,
                                             bootstrap_replicates, seed=100 + len(tests))})
    for name in ["no_overlap_cap", "no_same_tail_filter", "fixed_hot_blue", "single_banker_baseline"]:
        for metric in metrics:
            diffs = [a[metric] - b[metric] for a, b in zip(rows_by_name[name], rows_by_name["baseline"])]
            tests.append({"comparison": f"{name} - baseline: {metric}", "metric": metric,
                          **_bootstrap(diffs, bootstrap_replicates, seed=100 + len(tests))})
    for test, adjusted in zip(tests, _holm([test["two_sided_p"] for test in tests])):
        test["holm_adjusted_p"] = adjusted
        test["significant_at_0_05_after_holm"] = adjusted < .05

    return {"method": {"walk_forward": True, "periods": periods, "warmup_draws": start,
            "requested_tickets_per_period": n_sets, "same_actual_ticket_count_per_period": True,
            "random_replicates": random_replicates, "bootstrap_replicates": bootstrap_replicates,
            "block_length": 10,
            "single_banker_baseline": "engine.generate_v2_baseline_tickets: current explicit single-banker approximation; not a historical V2 implementation",
            "random_baseline": "uniform independent legal SSQ tickets, repeated seeds",
            "prize_model": "fixed estimates from engine.eval_prize; actual floating jackpots are not modeled",
            "interpretation": "No predictive edge is presumed; raw ROI is volatile and not a guarantee."},
        "issue_range": [draws[start].issue, draws[-1].issue],
        "theoretical_single_ticket": theoretical_single_ticket(), "summary": summary,
        "paired_block_bootstrap_tests": tests, "period_details": detail}


def main():
    parser = argparse.ArgumentParser(description="SSQ walk-forward audit with random and theoretical baselines")
    parser.add_argument("--eval", type=int, default=500, help="out-of-sample periods (default: 500)")
    parser.add_argument("--sets", type=int, default=5)
    parser.add_argument("--random-replicates", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="parallel period workers; results are identical for any value (default: up to 4)")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("ssq_backtest_audit.json"))
    args = parser.parse_args()
    if args.eval < 10 or not 1 <= args.sets <= 20 or args.random_replicates < 2 or args.bootstrap_replicates < 100:
        parser.error("eval>=10, sets=1..20, random-replicates>=2, bootstrap-replicates>=100 are required")
    draws = engine.load_verified_cwl_data(offline=args.offline, force_refresh=args.force_refresh)
    report = run_audit(draws, args.eval, args.sets, args.random_replicates, args.bootstrap_replicates,
                       workers=args.workers)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"回测期号: {report['issue_range'][0]}–{report['issue_range'][1]}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\n区块 bootstrap + Holm 校正:")
    print(json.dumps(report["paired_block_bootstrap_tests"], ensure_ascii=False, indent=2))
    print(f"\nJSON report: {args.output}")


if __name__ == "__main__":
    main()
