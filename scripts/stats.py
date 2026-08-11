"""Statistics for comparing two models on the same benchmark items.

Baseline and tuned are scored on an identical, fixed set of tasks, so the two
score vectors are paired. Reporting two independent Wald intervals and eyeballing
whether they overlap is the wrong test here -- it throws away the pairing and is
badly conservative. What matters is the distribution of the per-item *difference*,
so the delta CI is a paired bootstrap over items, and significance is McNemar's
exact test on the discordant pairs.

Everything is exact or resampled; no normal approximation, because per-category
counts here are in the low hundreds and pass rates run near the 0/1 boundaries
where the Wald interval misbehaves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict


@dataclass
class Interval:
    point: float
    low: float
    high: float

    def as_pct(self) -> str:
        return f"{self.point:6.2%}  [{self.low:6.2%}, {self.high:6.2%}]"

    def to_dict(self) -> dict:
        return {k: round(v, 6) for k, v in asdict(self).items()}


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> Interval:
    """Wilson score interval for a single pass rate.

    Preferred over Wald because BFCL category sizes are small and rates sit close
    to 0 or 1, where Wald produces intervals that run outside [0, 1].
    """
    if n == 0:
        return Interval(0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_delta(
    baseline: list[int],
    tuned: list[int],
    iterations: int = 10000,
    seed: int = 20260809,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap CI for (tuned - baseline) pass rate, resampling items.

    Resamples item indices, not the two vectors independently -- the pairing is
    the whole point. Each draw keeps a task's baseline and tuned outcome together.
    """
    if len(baseline) != len(tuned):
        raise ValueError(f"unpaired vectors: {len(baseline)} vs {len(tuned)}")
    n = len(baseline)
    if n == 0:
        return Interval(0.0, 0.0, 0.0)

    observed = (sum(tuned) - sum(baseline)) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            i = rng.randrange(n)
            total += tuned[i] - baseline[i]
        deltas.append(total / n)
    deltas.sort()
    lo = deltas[int((alpha / 2) * iterations)]
    hi = deltas[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return Interval(observed, lo, hi)


def mcnemar_exact(baseline: list[int], tuned: list[int]) -> dict:
    """Exact McNemar test on paired binary outcomes.

    Only discordant pairs carry information: items both models pass, or both
    fail, say nothing about which is better. Uses the exact binomial rather than
    the chi-square approximation, which is unreliable when discordant counts are
    small -- a realistic case for the smaller BFCL categories.
    """
    gained = sum(1 for b, t in zip(baseline, tuned) if t == 1 and b == 0)
    lost = sum(1 for b, t in zip(baseline, tuned) if t == 0 and b == 1)
    n_disc = gained + lost

    if n_disc == 0:
        return {"gained": 0, "lost": 0, "discordant": 0, "p_value": 1.0}

    k = min(gained, lost)
    tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2**n_disc)
    return {
        "gained": gained,
        "lost": lost,
        "discordant": n_disc,
        "p_value": min(1.0, 2 * tail),
    }


def compare(name: str, baseline: list[int], tuned: list[int]) -> dict:
    n = len(baseline)
    base_ci = wilson_interval(sum(baseline), n)
    tuned_ci = wilson_interval(sum(tuned), n)
    delta = paired_bootstrap_delta(baseline, tuned)
    mcnemar = mcnemar_exact(baseline, tuned)
    return {
        "category": name,
        "n": n,
        "baseline": base_ci.to_dict(),
        "tuned": tuned_ci.to_dict(),
        "delta": delta.to_dict(),
        "mcnemar": mcnemar,
    }


def format_row(r: dict, labels: tuple[str, str] = ("base", "tuned")) -> str:
    b, t, d, m = r["baseline"], r["tuned"], r["delta"], r["mcnemar"]
    sig = "*" if m["p_value"] < 0.05 else " "
    lo, hi = labels
    return (
        f"{r['category']:<22} n={r['n']:>4}  "
        f"{lo} {b['point']:6.2%}  {hi} {t['point']:6.2%}  "
        f"delta {d['point']:+6.2%} [{d['low']:+6.2%}, {d['high']:+6.2%}]  "
        f"+{m['gained']}/-{m['lost']}  p={m['p_value']:.4g}{sig}"
    )


if __name__ == "__main__":
    # Sanity check the machinery on a known-signal synthetic pair.
    rng = random.Random(0)
    base = [1 if rng.random() < 0.30 else 0 for _ in range(400)]
    tune = [1 if (b or rng.random() < 0.40) else 0 for b in base]
    r = compare("synthetic_check", base, tune)
    print(format_row(r))
    print("\nself-check: a model that never regresses should show lost=0")
    assert r["mcnemar"]["lost"] == 0, "monotone improvement must not lose items"
    print("ok")
