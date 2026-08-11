"""Aggregate the 3-seed replication of Ablation B (calls-only vs swapped).

The single-seed run left one claim without an interval: with total rows held
fixed, the four call categories moved -1 to -3% and none reached significance,
so the tax was reported as "real but below this benchmark's resolution". That is
a statement about variance, and it needs more than one draw.

Two views are printed, because they fail in opposite directions:

1. PER-SEED. Each seed is analysed on its own, then the three deltas are
   summarised by mean and range. This respects the fact that a seed is the unit
   of replication, but n=3 is far too small for a real interval -- the range is
   descriptive, not a confidence statement.

2. POOLED. All 3x1,240 paired outcomes stacked into one comparison. This has
   real power, and it OVERSTATES it: the same 1,240 tasks recur in every seed, so
   the observations are clustered rather than independent, and the interval is
   narrower than a cluster-robust one would be. It is reported as an upper bound
   on the evidence, not as the headline.

Where the two disagree, the per-seed view wins. Sign consistency across seeds is
also reported: for a real effect the direction should hold, and 3/3 in the same
direction is weak-but-real evidence (sign test p=0.25 one-sided) in a way that
2/3 is not.
"""

import json
from pathlib import Path

from stats import compare, format_row

RESULTS = Path(r"D:\ToolCallFT\results")
RUNS = Path(r"D:\ToolCallFT\runs")

SEEDS = [20260809, 20260810, 20260811]
# arm A = calls only; arm B = 1,200 calls swapped for refusals, rows held fixed
ARM_A = {s: f"tuned_s{s}" for s in SEEDS}
ARM_B = {
    20260809: "tuned_fixed",  # already ran under the unified memory config
    20260810: "tuned_fixed_s20260810",
    20260811: "tuned_fixed_s20260811",
}

CALL_CATS = ["simple_python", "multiple", "parallel", "parallel_multiple"]


def load(tag: str) -> dict | None:
    p = RESULTS / f"scored_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def vectors(left: dict, right: dict, cat: str) -> tuple[list[int], list[int]]:
    l_items, r_items = left[cat], right[cat]
    if set(l_items) != set(r_items):
        raise SystemExit(f"{cat}: task ids differ between arms; pairing is broken")
    ids = sorted(l_items)
    return [int(l_items[i]) for i in ids], [int(r_items[i]) for i in ids]


def summarise(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    mean = sum(values) / len(values)
    return f"mean {mean:+6.2%}  range [{lo:+6.2%}, {hi:+6.2%}]"


def main() -> int:
    arms = {}
    for seed in SEEDS:
        a, b = load(ARM_A[seed]), load(ARM_B[seed])
        if a is None or b is None:
            missing = ARM_A[seed] if a is None else ARM_B[seed]
            print(f"skipping seed {seed}: missing scored_{missing}.json")
            continue
        arms[seed] = (a, b)

    if not arms:
        raise SystemExit("no complete seed pairs found -- run run_multiseed.ps1 first")
    print(f"complete seed pairs: {sorted(arms)}\n")

    cats = sorted(next(iter(arms.values()))[0])

    # ---- view 1: per seed --------------------------------------------------
    print("=" * 110)
    print("PER-SEED  (arm A = calls only  ->  arm B = composition swapped)")
    print("=" * 110)
    per_cat: dict[str, list[float]] = {c: [] for c in cats}
    for seed, (a, b) in sorted(arms.items()):
        print(f"\n--- seed {seed} ---")
        pooled_a, pooled_b = [], []
        for cat in cats:
            av, bv = vectors(a, b, cat)
            pooled_a.extend(av)
            pooled_b.extend(bv)
            row = compare(cat, av, bv)
            per_cat[cat].append(row["delta"]["point"])
            print(format_row(row, labels=("callsonly", "swapped")))
        print(format_row(compare("OVERALL", pooled_a, pooled_b), labels=("callsonly", "swapped")))

    # ---- across-seed summary ----------------------------------------------
    print("\n" + "=" * 110)
    print("ACROSS SEEDS  (n=3 -- range is descriptive, NOT a confidence interval)")
    print("=" * 110)
    for cat in cats:
        vals = per_cat[cat]
        if not vals:
            continue
        neg = sum(1 for v in vals if v < 0)
        pos = sum(1 for v in vals if v > 0)
        consistent = "all negative" if neg == len(vals) else (
            "all positive" if pos == len(vals) else f"MIXED ({pos}+/{neg}-)")
        print(f"{cat:<20} {summarise(vals)}   sign: {consistent}")

    tax = [v for c in CALL_CATS for v in per_cat.get(c, [])]
    if tax:
        neg = sum(1 for v in tax if v < 0)
        print(f"\ncall categories pooled across seeds ({len(tax)} deltas): "
              f"{summarise(tax)}")
        print(f"  {neg}/{len(tax)} negative")

    # ---- view 2: pooled ----------------------------------------------------
    print("\n" + "=" * 110)
    print("POOLED  (all seeds stacked; power OVERSTATED -- tasks repeat across seeds)")
    print("=" * 110)
    pooled_rows = []
    for cat in cats:
        av_all, bv_all = [], []
        for _, (a, b) in sorted(arms.items()):
            av, bv = vectors(a, b, cat)
            av_all.extend(av)
            bv_all.extend(bv)
        row = compare(cat, av_all, bv_all)
        pooled_rows.append(row)
        print(format_row(row, labels=("callsonly", "swapped")))

    av_all, bv_all = [], []
    for cat in cats:
        for _, (a, b) in sorted(arms.items()):
            av, bv = vectors(a, b, cat)
            av_all.extend(av)
            bv_all.extend(bv)
    overall = compare("OVERALL", av_all, bv_all)
    pooled_rows.append(overall)
    print(format_row(overall, labels=("callsonly", "swapped")))

    # ---- config-inertness check -------------------------------------------
    old, new = load("tuned"), load(ARM_A[20260809])
    config_check = None
    if old and new:
        print("\n" + "=" * 110)
        print("CONFIG CHECK  original `tuned` vs same seed under --pad-multiple 256"
              " --eval-strategy no")
        print("  These settings are claimed to be memory-side only. Same seed, same data,")
        print("  so a real difference here would falsify that claim.")
        print("=" * 110)
        av_all, bv_all = [], []
        for cat in sorted(set(old) & set(new)):
            av, bv = vectors(old, new, cat)
            av_all.extend(av)
            bv_all.extend(bv)
            print(format_row(compare(cat, av, bv), labels=("orig_cfg", "new_cfg")))
        config_check = compare("OVERALL", av_all, bv_all)
        print(format_row(config_check, labels=("orig_cfg", "new_cfg")))

    out = {
        "seeds": sorted(arms),
        "per_category_deltas": per_cat,
        "pooled": pooled_rows,
        "config_check": config_check,
        "training": {
            d.parent.name: json.loads(d.read_text(encoding="utf-8"))
            for d in sorted(RUNS.glob("*/train_summary.json"))
        },
    }
    (RESULTS / "multiseed_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'multiseed_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
