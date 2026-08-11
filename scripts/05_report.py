"""Turn per-item BFCL outcomes into the results tables, with paired statistics.

Input:  results/scored_<model>.json  -- {category: {task_id: 0|1}}
Output: results/report.json and results/report.md

Three paired comparisons are reported. The first two measure each fine-tune
against the untouched base model. The third is the ablation proper: `tuned` vs
`tuned_refusal` isolates the effect of adding refusal examples, since the base
model, LoRA config, learning rate, seed, and decoding are identical between them
and the training file is the only thing that changed.

Every comparison requires the two sides to cover exactly the same task ids. If
they do not, the pairing is broken and every interval below it is wrong, so this
refuses to proceed rather than silently intersecting them.
"""

import json
from pathlib import Path

from stats import compare, format_row

RESULTS = Path(r"D:\ToolCallFT\results")
RUNS = Path(r"D:\ToolCallFT\runs")

COMPARISONS = [
    ("base", "tuned", "Run 1 - ToolACE as shipped (every target is a tool call)"),
    ("base", "tuned_refusal", "Run 2 - ToolACE + 1,200 constructed refusals (6,116 rows)"),
    ("base", "tuned_fixed", "Run 3 - 1,200 calls REPLACED by refusals (4,920 rows)"),
    ("tuned", "tuned_refusal", "Ablation A - adding refusals (confounded with +24% data)"),
    # The clean one: identical row count to run 1, so the only thing that moved
    # is what those rows contain.
    ("tuned", "tuned_fixed", "Ablation B - composition only, total rows held fixed"),
    ("tuned_refusal", "tuned_fixed", "Does the extra 24% of call data matter?"),
]


def load(name: str) -> dict | None:
    path = RESULTS / f"scored_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(left: dict, right: dict, tag: str) -> list[dict]:
    cats = sorted(set(left) | set(right))
    missing = [c for c in cats if c not in left or c not in right]
    if missing:
        raise SystemExit(f"{tag}: categories missing from one side: {missing}")

    rows, pooled_l, pooled_r = [], [], []
    for cat in cats:
        l_items, r_items = left[cat], right[cat]
        if set(l_items) != set(r_items):
            only_l = len(set(l_items) - set(r_items))
            only_r = len(set(r_items) - set(l_items))
            raise SystemExit(
                f"{tag}/{cat}: task ids differ (left-only {only_l}, right-only {only_r}); "
                "pairing is broken, refusing to report"
            )
        ids = sorted(l_items)
        lv = [int(l_items[i]) for i in ids]
        rv = [int(r_items[i]) for i in ids]
        pooled_l.extend(lv)
        pooled_r.extend(rv)
        rows.append(compare(cat, lv, rv))

    rows.append(compare("OVERALL", pooled_l, pooled_r))
    return rows


def md_table(rows: list[dict], left: str, right: str) -> list[str]:
    md = [
        f"| category | n | {left} | {right} | delta (95% CI) | gained/lost | McNemar p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b, t, d, m = r["baseline"], r["tuned"], r["delta"], r["mcnemar"]
        name = f"**{r['category']}**" if r["category"] == "OVERALL" else r["category"]
        md.append(
            f"| {name} | {r['n']} | {b['point']:.2%} | {t['point']:.2%} | "
            f"{d['point']:+.2%} [{d['low']:+.2%}, {d['high']:+.2%}] | "
            f"+{m['gained']}/-{m['lost']} | {m['p_value']:.3g} |"
        )
    return md


def main() -> int:
    scored = {name: load(name) for name in ("base", "tuned", "tuned_refusal", "tuned_fixed")}
    available = [k for k, v in scored.items() if v is not None]
    print(f"scored models available: {available}")
    for name, data in scored.items():
        if data is None:
            print(f"  (missing scored_{name}.json -- comparisons using it are skipped)")

    # Cost is keyed by run directory. The previous version overwrote a single
    # dict inside the glob loop, so with more than one run present it reported
    # whichever summary the filesystem happened to yield last -- numbers from one
    # experiment attached to another.
    costs = {}
    for summary in sorted(RUNS.glob("*/train_summary.json")):
        s = json.loads(summary.read_text(encoding="utf-8"))
        costs[summary.parent.name] = {
            "train_file": s.get("train_file", "train.jsonl"),
            "train_composition": s.get("train_composition"),
            # Recorded because the two runs differ here: the refusal run needed
            # bucketed padding and no in-training eval to fit in 8 GiB. Both are
            # memory-side only, but a config difference between compared runs
            # belongs in the artifact rather than in someone's memory.
            "pad_multiple": s.get("pad_multiple", 0),
            "eval_strategy": s.get("eval_strategy", "steps"),
            "train_rows": s.get("train_rows"),
            "train_runtime_min": s.get("train_runtime_min"),
            "peak_vram_gib": s.get("peak_vram_gib"),
            "final_train_loss": s.get("final_train_loss"),
            "lora_r": s.get("lora_r"),
            "base_model": s.get("base_model"),
        }

    report = {"comparisons": {}, "training_cost": costs}
    md: list[str] = []

    for left, right, headline in COMPARISONS:
        if scored[left] is None or scored[right] is None:
            continue
        tag = f"{left}_vs_{right}"
        rows = build_rows(scored[left], scored[right], tag)
        report["comparisons"][tag] = rows

        print(f"\n### {headline}   [{left} -> {right}]")
        print("-" * 118)
        for r in rows:
            print(format_row(r, labels=(left, right)))
        print("-" * 118)

        md += [f"### {headline}", "", *md_table(rows, left, right), ""]

    print("* = McNemar exact p < 0.05.  delta CI = paired bootstrap, 10k resamples.")

    if costs:
        md += ["### Training cost", "", "| run | rows | composition | min | peak VRAM |", "|---|---|---|---|---|"]
        for name, c in costs.items():
            comp = c.get("train_composition") or {}
            comp_str = ", ".join(f"{k}={v:,}" for k, v in sorted(comp.items())) or "calls only"
            md.append(
                f"| {name} | {c.get('train_rows'):,} | {comp_str} | "
                f"{c.get('train_runtime_min')} | {c.get('peak_vram_gib')} GiB |"
            )
        md.append("")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULTS / 'report.json'} and {RESULTS / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
