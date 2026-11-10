#!/usr/bin/env python3
"""Honest walk-forward SSQ audit: strategy vs uniform random and theory.

This research runner is intentionally separate from the production predictor. It
uses only prior draws for each forecast, equal requested ticket budgets, seeded
random controls, fixed-prize assumptions, block-bootstrap intervals and Holm
correction for rule ablations. It does not claim that any method predicts a fair
lottery.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Sequence, Tuple

import numpy as np

import ssq_master_v3 as engine


PRIZE_RANK = {"一等奖": 1, "二等奖": 2, "三等奖": 3, "四等奖": 4, "五等奖": 5, "六等奖": 6}


class StableAuditCalibrator(engine.OptimalCalibrator):
    """Numerically stable audit variant; inverse-MSE weights are normalized directly."""
    def calibrate(self, draws):
        n = len(draws)
        test_start = max(20, n - self.window)
        rounds = n - test_start
        losses = np.full(len(self.features), 1e-4, dtype=float)
        residual = np.zeros(engine.RED_N, dtype=float)
        if rounds >= 3:
            decay = np.exp(np.linspace(-0.8, 0.0, rounds))
            for k, t in enumerate(range(test_start, n)):
                maps = engine.compute_feature_probabilities(draws[:t])
                y = np.array([float(r in draws[t].reds) for r in range(1, engine.RED_N + 1)])
                scores = [maps[name] / max(float(np.max(maps[name])), 1e-12) for name in self.features]
                for i, score in enumerate(scores):
                    losses[i] += float(np.mean((y - score) ** 2)) * decay[k]
                residual += (y - np.mean(scores, axis=0)) * decay[k]
            residual /= rounds
        inverse = 1.0 / np.maximum(losses, 1e-12)
        weights = inverse / inverse.sum() if np.isfinite(inverse).all() and inverse.sum() else np.full(len(self.features), 1 / len(self.features))

        if self.miss_tilt > 0 and rounds >= 3:
            missed_by_feature = Counter()
            total_missed = 0
            for t in range(test_start, n):
                maps = engine.compute_feature_probabilities(draws[:t])
                blend = sum(weights[i] * maps[name] / max(float(np.max(maps[name])), 1e-12)
                            for i, name in enumerate(self.features))
                top = set((np.argsort(-blend) + 1)[:12].tolist())
                missed = set(draws[t].reds) - top
                total_missed += len(missed)
                for r in missed:
                    relative = {name: maps[name][r - 1] / max(float(np.max(maps[name])), 1e-12)
                                for name in self.features}
                    missed_by_feature[min(relative, key=relative.get)] += 1
            if total_missed:
                weights *= np.array([1.0 + self.miss_tilt * missed_by_feature[name] / total_missed
                                     for name in self.features])
                weights /= weights.sum()

        current = engine.compute_feature_probabilities(draws)
        logits = [engine.logit(np.clip(current[name] / max(float(np.max(current[name])), 1e-12) * 0.82 + 0.03,
                                       0.02, 0.88)) for name in self.features]
        base = engine.norm_expected_6(engine.expit(sum(weights[i] * logits[i] for i in range(len(self.features)))))
        calibrated = engine.norm_expected_6(np.clip(base + self.lr * residual * engine.PICK, 0.02, None))
        gaps = np.zeros(engine.RED_N)
        for r in range(1, engine.RED_N + 1):
            for d in reversed(draws):
                if r in d.reds:
                    break
                gaps[r - 1] += 1
        return calibrated, current["遗漏反弹"], gaps, dict(zip(self.features, weights))


def theoretical_single_ticket() -> Dict[str, float]:
    """Exact SSQ single-ticket prize probabilities under uniform independent draws."""
    total_red = math.comb(33, 6)
    expected_return = 0.0
    prize_probs = Counter()
    for red_hits in range(7):
        p_red = math.comb(6, red_hits) * math.comb(27, 6 - red_hits) / total_red
        for blue_hit, p_blue in ((0, 15 / 16), (1, 1 / 16)):
            prize, amount = engine.eval_prize(red_hits, blue_hit)
            if prize:
                prize_probs[prize] += p_red * p_blue
                expected_return += p_red * p_blue * amount
    return {
        "ticket_win_probability": float(sum(prize_probs.values())),
        "expected_fixed_prize_yuan": float(expected_return),
        "ticket_cost_yuan": 2.0,
        "expected_fixed_prize_roi_percent": float(expected_return / 2 * 100),
        "prize_probabilities": dict(prize_probs),
        "assumption": "uniform independent draws; fixed prize estimates in ssq_master_v3.eval_prize; floating prizes excluded/approximated",
    }


def _tickets_from_combos(combos, blue_roster):
    tickets = []
    for i, combo in enumerate(combos):
        blue = blue_roster[i % len(blue_roster)]
        tickets.append({"推荐红球": " ".join(f"{x:02d}" for x in combo),
                        "推荐蓝球": f"{blue:02d}"})
    return tickets


def _generate_v3(inputs, actual, n_sets, variant, seed):
    high, mid, out, hot, gold, cold, last, regime, patterns, quota = inputs
    kwargs = dict(n_sets=n_sets, diversity_limit=3, enforce_same_tail=True,
                  include_blue_triad=True, require_min_jaccard=True)
    if variant == "no_overlap_cap":
        kwargs.update(diversity_limit=6, require_min_jaccard=False)
    elif variant == "no_same_tail_filter":
        kwargs["enforce_same_tail"] = False
    elif variant == "fixed_hot_blue":
        kwargs["include_blue_triad"] = False

    generated = engine.generate_v3_optimal_tickets(
        high, mid, out, hot, gold, cold, last, actual.issue, regime, patterns, quota,
        **kwargs,
    )
    combos = [sorted(int(x) for x in ticket["推荐红球"].split()) for ticket in generated]
    if variant == "baseline":
        # Enforce the advertised pairwise overlap cap after the legacy generator's
        # quota/fallback selection. Do not silently relax; report resulting shortfall.
        accepted = []
        for combo in combos:
            if engine.is_diverse_enough(combo, accepted, max_overlap=3):
                accepted.append(combo)
        combos = accepted
    elif variant == "no_overlap_cap":
        # Use the same candidate-generation mechanism, but do not apply the final cap.
        pass
    roster = [hot[0], gold[0], hot[1] if len(hot) > 1 else hot[0], cold[0],
              gold[1] if len(gold) > 1 else gold[0]]
    if variant == "fixed_hot_blue":
        roster = [hot[0]]
    return _tickets_from_combos(combos[:n_sets], roster)


def _uniform_tickets(actual, n_sets, seed):
    rng = random.Random(seed)
    combos = []
    seen = set()
    while len(combos) < n_sets:
        red = tuple(sorted(rng.sample(range(1, 34), 6)))
        blue = rng.randint(1, 16)
        key = (red, blue)
        if key in seen:
            continue
        seen.add(key)
        combos.append({"推荐红球": " ".join(f"{x:02d}" for x in red), "推荐蓝球": f"{blue:02d}"})
    return combos


def _score(tickets, actual):
    wins = 0
    reward = 0
    prizes = Counter()
    for ticket in tickets:
        reds = [int(x) for x in ticket["推荐红球"].split()]
        blue = int(ticket["推荐蓝球"].split()[0])
        name, amount = engine.eval_prize(len(set(reds) & set(actual.reds)), int(blue == actual.blue))
        if name:
            wins += 1
            reward += amount
            prizes[name] += 1
    count = len(tickets)
    return {
        "tickets": count,
        "missing": 0,
        "winning_tickets": wins,
        "reward_yuan": reward,
        "cost_yuan": count * 2,
        "ticket_win_rate": wins / count if count else 0.0,
        "roi_percent": reward / (count * 2) * 100 if count else 0.0,
        "any_prize_period": int(wins > 0),
        "prizes": dict(prizes),
    }


def _block_bootstrap(diffs, replicates=4000, block=10, seed=982451653):
    n = len(diffs)
    if not n:
        return {"mean_difference": 0.0, "ci95": [0.0, 0.0], "two_sided_p": 1.0}
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(diffs[(start + j) % n] for j in range(block))
        means.append(fmean(sample[:n]))
    means.sort()
    lo = means[max(0, int(0.025 * replicates))]
    hi = means[min(replicates - 1, int(0.975 * replicates))]
    lower = (sum(x <= 0 for x in means) + 1) / (replicates + 1)
    upper = (sum(x >= 0 for x in means) + 1) / (replicates + 1)
    return {"mean_difference": fmean(diffs), "ci95": [lo, hi],
            "two_sided_p": min(1.0, 2 * min(lower, upper)),
            "replicates": replicates, "block_length": block}


def _holm(pvalues):
    """Holm step-down adjusted p-values, mapped back to original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _prep_round(train, actual, calibrator, n_sets):
    regime = engine.detect_regime_invert(train)
    patterns = {d.zones_3 for d in train[-25:]} if len(train) >= 25 else None
    calpr, rbpr, _, _ = calibrator.calibrate(train)
    ranked = np.argsort(-calpr) + 1
    high = ranked[:12].tolist()
    mid = ranked[12:22].tolist()
    out = sorted(ranked[22:].tolist(), key=lambda x: rbpr[x - 1], reverse=True)
    hot, gold, cold, _, _ = engine.evaluate_blue_triad(train)
    _, pred = engine.analyze_tail_markov_transition(train)
    fatigue, _ = engine.detect_tail_fatigue_inversion(train)
    quota = engine.compute_dynamic_tail_quota(pred, fatigue, n_sets)
    return high, mid, out[:7], hot, gold, cold, train[-1], regime, patterns, quota


def run_audit(draws: Sequence, periods=30, n_sets=5, random_replicates=32, bootstrap_replicates=4000):
    if len(draws) < engine.MIN_HISTORY_DRAWS:
        raise ValueError(f"need at least {engine.MIN_HISTORY_DRAWS} valid historical draws; got {len(draws)}")
    if periods < 10 or periods > len(draws) - 50:
        raise ValueError(f"periods must be between 10 and {len(draws) - 50}, got {periods}")
    if not 1 <= n_sets <= 20 or random_replicates < 2:
        raise ValueError("sets must be 1..20 and random-replicates >= 2")

    start = len(draws) - periods
    calibrator = StableAuditCalibrator(window=20, lr=0.35, miss_tilt=engine.DEFAULT_TILT)
    variants = ["baseline", "no_overlap_cap", "no_same_tail_filter", "fixed_hot_blue"]
    by_variant = {v: [] for v in variants}
    random_by_rep = [[] for _ in range(random_replicates)]
    details = []

    for idx in range(start, len(draws)):
        train, actual = draws[:idx], draws[idx]
        inputs = _prep_round(train, actual, calibrator, n_sets)
        round_record = {"issue": actual.issue, "variants": {}}
        for vi, variant in enumerate(variants):
            tickets = _generate_v3(inputs, actual, n_sets, variant, actual.issue + vi)
            result = _score(tickets, actual)
            result["missing"] = max(0, n_sets - len(tickets))
            result["overlap_max"] = max((len(set(a["推荐红球"].split()) & set(b["推荐红球"].split()))
                                         for i, a in enumerate(tickets) for b in tickets[i + 1:]), default=0)
            by_variant[variant].append(result)
            round_record["variants"][variant] = result
        for rep in range(random_replicates):
            tickets = _uniform_tickets(actual, n_sets, seed=actual.issue * 1009 + rep)
            random_by_rep[rep].append(_score(tickets, actual))
        details.append(round_record)

    def aggregate(rows):
        ticket_count = sum(r["tickets"] for r in rows)
        wins = sum(r["winning_tickets"] for r in rows)
        reward = sum(r["reward_yuan"] for r in rows)
        cost = sum(r["cost_yuan"] for r in rows)
        return {"periods": len(rows), "actual_tickets": ticket_count,
                "missing_tickets": sum(r["missing"] for r in rows),
                "winning_tickets": wins,
                "ticket_win_rate": wins / ticket_count if ticket_count else 0.0,
                "period_any_prize_rate": fmean(r["any_prize_period"] for r in rows) if rows else 0.0,
                "reward_yuan": reward, "cost_yuan": cost,
                "roi_percent": reward / cost * 100 if cost else 0.0,
                "prize_counts": dict(sum((Counter(r["prizes"]) for r in rows), Counter()))}

    random_mean_rows = []
    for i in range(periods):
        run_mean = []
        for rep in range(random_replicates):
            run_mean.append(random_by_rep[rep][i])
        # The random null uses fixed requested tickets per period.
        wins = fmean(r["winning_tickets"] for r in run_mean)
        reward = fmean(r["reward_yuan"] for r in run_mean)
        random_mean_rows.append({"tickets": n_sets, "missing": 0, "winning_tickets": wins,
                                 "reward_yuan": reward, "cost_yuan": n_sets * 2,
                                 "ticket_win_rate": wins / n_sets,
                                 "roi_percent": reward / (n_sets * 2) * 100,
                                 "any_prize_period": fmean(r["any_prize_period"] for r in run_mean),
                                 "prizes": {}})

    summaries = {v: aggregate(rows) for v, rows in by_variant.items()}
    summaries["uniform_random_mean"] = aggregate(random_mean_rows)
    replicates_summary = [aggregate(run) for run in random_by_rep]
    rnd_roi = [s["roi_percent"] for s in replicates_summary]
    summaries["uniform_random_null_roi_percent"] = {
        "replicates": random_replicates, "mean": fmean(rnd_roi),
        "q025": float(np.quantile(rnd_roi, .025)), "median": float(np.quantile(rnd_roi, .5)),
        "q975": float(np.quantile(rnd_roi, .975)),
    }

    tests = []
    base = by_variant["baseline"]
    random_average = random_mean_rows
    for metric in ("ticket_win_rate", "roi_percent", "period_any_prize_rate"):
        left = [r[metric] for r in base]
        right = [r[metric] for r in random_average]
        tests.append({"comparison": f"baseline - random: {metric}", "metric": metric,
                      **_block_bootstrap([a - b for a, b in zip(left, right)], bootstrap_replicates)})
    for variant in variants[1:]:
        for metric in ("ticket_win_rate", "roi_percent", "period_any_prize_rate"):
            diffs = [a[metric] - b[metric] for a, b in zip(by_variant[variant], base)]
            tests.append({"comparison": f"{variant} - baseline: {metric}", "metric": metric,
                          **_block_bootstrap(diffs, bootstrap_replicates)})
    adjusted = _holm([t["two_sided_p"] for t in tests])
    for test, p_adj in zip(tests, adjusted):
        test["holm_adjusted_p"] = p_adj

    return {
        "method": {"walk_forward": True, "periods": periods, "warmup_draws": start,
                   "requested_tickets_per_period": n_sets, "random_replicates": random_replicates,
                   "bootstrap_replicates": bootstrap_replicates, "block_length": 10,
                   "calibrator": "StableAuditCalibrator (inverse-MSE weights normalized directly)",
                   "prize_model": "fixed values from ssq_master_v3.eval_prize; jackpot prizes are estimates, not actual historical payouts",
                   "important": "Lottery prediction edge is not established by this report. Randomness is the null; ROI is sensitive to rare fixed jackpot awards."},
        "issue_range": [draws[start].issue, draws[-1].issue],
        "theoretical_single_ticket": theoretical_single_ticket(),
        "summary": summaries,
        "paired_block_bootstrap_tests": tests,
        "period_details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="SSQ walk-forward/random-baseline/statistical audit")
    parser.add_argument("--eval", type=int, default=30, help="out-of-sample periods")
    parser.add_argument("--sets", type=int, default=5, help="tickets per period")
    parser.add_argument("--random-replicates", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("ssq_backtest_audit.json"))
    args = parser.parse_args()
    if args.eval < 10 or not 1 <= args.sets <= 20 or args.random_replicates < 2 or args.bootstrap_replicates < 100:
        parser.error("eval>=10, sets=1..20, random-replicates>=2, bootstrap-replicates>=100 are required")
    draws = engine.load_verified_cwl_data(offline=args.offline, force_refresh=args.force_refresh)
    report = run_audit(draws, args.eval, args.sets, args.random_replicates, args.bootstrap_replicates)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"回测期号: {report['issue_range'][0]}–{report['issue_range'][1]}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\n配对区块 bootstrap + Holm 校正:")
    print(json.dumps(report["paired_block_bootstrap_tests"], ensure_ascii=False, indent=2))
    print(f"\nJSON report: {args.output}")


if __name__ == "__main__":
    main()
