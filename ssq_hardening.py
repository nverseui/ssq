"""Conservative, auditable hardening for the SSQ V3 research engine.

This module does not claim predictive advantage. It fixes numerical stability,
quota arithmetic and silent portfolio-constraint relaxation, while keeping the
original engine available for comparison.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, Sequence

import numpy as np

import ssq_master_v3 as engine


def stable_softmax(values: Iterable[float]) -> np.ndarray:
    """Finite normalized softmax; invalid or degenerate input gets equal weights."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr -= float(np.max(arr))
    weights = np.exp(np.clip(arr, -700.0, 0.0))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=float)
    return weights / total


def largest_remainder_tail_quota(
    probabilities: Dict[int, float], n_sets: int, fatigue: bool = False
) -> Dict[int, int]:
    """Allocate exactly ``n_sets`` across 3/4/5/6-tail classes."""
    if n_sets < 1:
        raise ValueError("n_sets must be >= 1")
    weights = {k: float(probabilities.get(k, d)) for k, d in
               ((3, 0.05), (4, 0.33), (5, 0.50), (6, 0.12))}
    weights = {k: (v if np.isfinite(v) and v > 0 else 0.0) for k, v in weights.items()}
    if fatigue:
        weights[6] *= 1.8
        weights[5] *= 0.85
        weights[4] *= 0.85
    total = sum(weights.values())
    if total <= 0:
        weights = {3: 0.05, 4: 0.33, 5: 0.50, 6: 0.12}
        total = sum(weights.values())

    exact = {k: n_sets * v / total for k, v in weights.items()}
    quota = {k: int(np.floor(v)) for k, v in exact.items()}
    remaining = n_sets - sum(quota.values())
    for key in sorted(exact, key=lambda k: (exact[k] - quota[k], k), reverse=True)[:remaining]:
        quota[key] += 1
    if sum(quota.values()) != n_sets:
        raise AssertionError("tail quotas do not sum to requested ticket count")
    return quota


def centered_residual_tilt(residual: np.ndarray) -> np.ndarray:
    """Remove a common residual offset so it cannot collapse the whole ranking."""
    arr = np.nan_to_num(np.asarray(residual, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if not arr.size:
        return arr
    arr = arr - float(arr.mean())
    return arr - float(arr.mean())


class HardenedCalibrator(engine.OptimalCalibrator):
    """Walk-forward calibrator with finite loss-based weights and centered residuals."""

    def __init__(self, *args, tau: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = float(tau)
        self._feature_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _features_at(self, draws, end: int):
        result = self._feature_cache.get(end)
        if result is None:
            result = engine.compute_feature_probabilities(draws[:end])
            self._feature_cache[end] = result
        return result

    def calibrate(self, draws: Sequence):
        if not draws:
            raise ValueError("at least one historical draw is required")
        n = len(draws)
        start = max(20, n - self.window)
        rounds = n - start
        losses = np.full(len(self.features), 1e-4, dtype=float)
        residual = np.zeros(engine.RED_N, dtype=float)

        if rounds >= 3:
            decay = np.exp(np.linspace(-0.8, 0.0, rounds))
            for k, t in enumerate(range(start, n)):
                maps = self._features_at(draws, t)
                actual = np.asarray([float(r in draws[t].reds)
                                     for r in range(1, engine.RED_N + 1)])
                scores = []
                for i, name in enumerate(self.features):
                    vector = np.asarray(maps[name], dtype=float)
                    score = vector / max(float(np.max(vector)), 1e-12)
                    losses[i] += float(np.mean((actual - score) ** 2)) * decay[k]
                    scores.append(score)
                residual += (actual - np.mean(scores, axis=0)) * decay[k]
            residual /= rounds

        scaled = losses / max(float(np.mean(losses)), 1e-12)
        weights = stable_softmax(-scaled / self.tau)
        if self.miss_tilt > 0 and rounds >= 3:
            missed_counts = Counter()
            total_missed = 0
            for t in range(start, n):
                maps = self._features_at(draws, t)
                scores = [maps[name] / max(float(np.max(maps[name])), 1e-12)
                          for name in self.features]
                blend = np.sum(np.asarray(scores) * weights[:, None], axis=0)
                missed = set(draws[t].reds) - set((np.argsort(-blend) + 1)[:12].tolist())
                total_missed += len(missed)
                for number in missed:
                    relative = [maps[name][number - 1] / max(float(np.max(maps[name])), 1e-12)
                                for name in self.features]
                    missed_counts[int(np.argmin(relative))] += 1
            if total_missed:
                weights *= np.asarray([1 + self.miss_tilt * missed_counts[i] / total_missed
                                       for i in range(len(self.features))])
                weights /= weights.sum()

        current = self._features_at(draws, n)
        logits = [engine.logit(np.clip(current[name] / max(float(np.max(current[name])), 1e-12)
                                       * 0.82 + 0.03, 0.02, 0.88))
                  for name in self.features]
        base = engine.norm_expected_6(engine.expit(
            sum(weights[i] * logits[i] for i in range(len(self.features)))))
        residual = centered_residual_tilt(residual) if rounds >= 3 else residual
        calibrated = engine.norm_expected_6(
            np.clip(base + self.lr * residual * engine.PICK, 0.02, None))

        gaps = np.zeros(engine.RED_N)
        for number in range(1, engine.RED_N + 1):
            for draw in reversed(draws):
                if number in draw.reds:
                    break
                gaps[number - 1] += 1
        return calibrated, current["遗漏反弹"], gaps, dict(zip(self.features, weights))


def exact_overlap_portfolio(
    candidates: Sequence[Sequence[int]], n_sets: int, max_overlap: int = 3
) -> list[list[int]]:
    """Select unique combinations without ever relaxing the overlap cap."""
    if n_sets < 1:
        raise ValueError("n_sets must be >= 1")
    if not 0 <= max_overlap < engine.PICK:
        raise ValueError(f"max_overlap must be in 0..{engine.PICK - 1}")
    selected: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        combo = sorted(int(x) for x in candidate)
        key = tuple(combo)
        if key in seen or len(combo) != engine.PICK or len(set(combo)) != engine.PICK:
            continue
        if any(x < 1 or x > engine.RED_N for x in combo):
            continue
        if not engine.is_diverse_enough(combo, selected, max_overlap=max_overlap):
            continue
        selected.append(combo)
        seen.add(key)
        if len(selected) == n_sets:
            break
    return selected


def validate_portfolio(
    red_combos: Sequence[Sequence[int]], blues: Sequence[int], n_sets: int,
    max_overlap: int = 3,
) -> Dict:
    """Return validation diagnostics instead of silently accepting malformed tickets."""
    issues: list[str] = []
    if len(red_combos) != n_sets:
        issues.append(f"ticket-count={len(red_combos)} != requested={n_sets}")
    if len(blues) != n_sets:
        issues.append(f"blue-count={len(blues)} != requested={n_sets}")
    seen: set[tuple[int, ...]] = set()
    for i, raw in enumerate(red_combos):
        combo = [int(x) for x in raw]
        key = tuple(sorted(combo))
        if len(combo) != engine.PICK or len(set(combo)) != engine.PICK:
            issues.append(f"ticket-{i}: red count/uniqueness invalid")
        if any(x < 1 or x > engine.RED_N for x in combo):
            issues.append(f"ticket-{i}: red range invalid")
        if key in seen:
            issues.append(f"ticket-{i}: duplicate red combination")
        seen.add(key)
    for i in range(len(red_combos)):
        for j in range(i + 1, len(red_combos)):
            overlap = len(set(red_combos[i]) & set(red_combos[j]))
            if overlap > max_overlap:
                issues.append(f"ticket-pair-{i}-{j}: overlap={overlap} > {max_overlap}")
    for i, blue in enumerate(blues):
        if not 1 <= int(blue) <= engine.BLUE_N:
            issues.append(f"blue-{i}: range invalid")
    return {"valid": not issues, "issues": issues}


def generate_tickets(
    draws: Sequence, n_sets: int = 5, max_overlap: int = 3, seed_offset: int = 0
) -> tuple[list[Dict], Dict]:
    """Generate conservative candidates and explicit diagnostics using this module's fixes."""
    if not 1 <= n_sets <= 20:
        raise ValueError("n_sets must be in 1..20")
    engine.validate_history(draws)
    last = draws[-1]
    target_issue = last.issue + 1
    calibrator = HardenedCalibrator(window=20, lr=0.35, miss_tilt=engine.DEFAULT_TILT)
    calibrated, red_bias, _, weights = calibrator.calibrate(draws)
    ranked = np.argsort(-calibrated) + 1
    high = ranked[:12].tolist()
    mid = ranked[12:22].tolist()
    out = sorted(ranked[22:].tolist(), key=lambda x: red_bias[x - 1], reverse=True)
    hot, gold, cold, _, _ = engine.evaluate_blue_triad(draws)
    _, tail_probs = engine.analyze_tail_markov_transition(draws)
    fatigue, streak = engine.detect_tail_fatigue_inversion(draws)
    quota = largest_remainder_tail_quota(tail_probs, n_sets, fatigue)
    regime = engine.detect_regime_invert(draws)
    patterns = {d.zones_3 for d in draws[-25:]} if len(draws) >= 25 else None
    main, hedge, primary, secondary = engine.generate_v3_candidate_combos(
        high, mid, out[:7], last, target_issue + seed_offset, regime, patterns,
        n_sets=n_sets, enforce_same_tail=True)

    buckets = defaultdict(list)
    for combo in main + hedge:
        buckets[len({x % 10 for x in combo})].append(combo)
    selected: list[list[int]] = []
    quota_selected: Dict[int, int] = {}
    for tails, count in quota.items():
        picked = exact_overlap_portfolio(buckets[tails], count, max_overlap)
        selected.extend(picked)
        quota_selected[tails] = len(picked)
    if len(selected) < n_sets:
        remaining = [combo for tails in (5, 4, 6, 3) for combo in buckets[tails]
                     if combo not in selected]
        selected.extend(exact_overlap_portfolio(remaining, n_sets - len(selected), max_overlap)
                        if not selected else _pick_additional(remaining, selected,
                                                              n_sets - len(selected), max_overlap))

    blue_roster = []
    for pool in (hot, gold, cold, range(1, engine.BLUE_N + 1)):
        for blue in pool:
            if int(blue) not in blue_roster:
                blue_roster.append(int(blue))
    blue_roster = blue_roster or [1]
    blues = [blue_roster[i % len(blue_roster)] for i in range(len(selected))]
    validation = validate_portfolio(selected, blues, len(selected), max_overlap)
    tickets = []
    for i, (combo, blue) in enumerate(zip(selected, blues), start=1):
        tickets.append({"序号": f"第{i:02d}注",
                        "推荐红球": " ".join(f"{x:02d}" for x in combo),
                        "推荐蓝球": f"{blue:02d}",
                        "尾数": f"{len({x % 10 for x in combo})}尾"})
    diagnostics = {
        "requested_tickets": n_sets,
        "generated_tickets": len(tickets),
        "missing_tickets": n_sets - len(tickets),
        "target_tail_quota": quota,
        "selected_by_tail_quota": quota_selected,
        "max_pair_overlap": max((len(set(selected[i]) & set(selected[j]))
                                  for i in range(len(selected)) for j in range(i + 1, len(selected))),
                                 default=0),
        "overlap_limit": max_overlap,
        "tail_fatigue_streak": streak,
        "calibration_weights": weights,
        "portfolio_valid": validation["valid"],
        "validation_issues": validation["issues"],
    }
    return tickets, diagnostics


def _pick_additional(candidates, selected, count, max_overlap):
    result = []
    seen = {tuple(c) for c in selected}
    for candidate in candidates:
        combo = sorted(int(x) for x in candidate)
        key = tuple(combo)
        if key in seen or not engine.is_diverse_enough(combo, list(selected) + result, max_overlap):
            continue
        result.append(combo)
        seen.add(key)
        if len(result) >= count:
            break
    return result
