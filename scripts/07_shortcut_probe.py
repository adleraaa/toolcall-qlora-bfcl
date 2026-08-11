"""Did refusal training teach judgement, or a lexical shortcut?

THE WORRY
`01b_build_refusals.py` only keeps a cross-paired refusal when the offered tool
names share ZERO word tokens with the request. That filter exists for a good
reason -- it stops serviceable pairs from being mislabelled as refusals, and a
false refusal label would teach the model to decline things it can do. But it
has a side effect: every refusal the model ever saw had disjoint vocabulary. A
model can fit that signal perfectly without learning anything about
applicability, by learning "tool words and request words don't overlap -> don't
call".

THE TEST
If it learned the shortcut, it should over-refuse on tasks where the vocabulary
happens not to overlap but the correct action is still to call. BFCL's call
categories already contain such tasks. So: split every call task by whether its
tool names share a token with its request, and compare how often each model
emits a call in each bucket.

The diagnostic quantity is EMISSION RATE (did the model produce a parseable
call at all), not accuracy. Accuracy conflates "declined when it should have
called" with "called the wrong thing"; only the first is evidence of the
shortcut.

Reading the result:
  - A refusal-trained model that emits calls at the same rate in both buckets
    learned something better than the lexical rule.
  - A large drop confined to the disjoint bucket, and absent from the
    calls-only models, is the shortcut showing itself.
  - A drop present in the calls-only baselines too means the bucket is simply
    harder, and says nothing about refusal training.

The calls-only arm is the control that makes the difference interpretable: it
never saw a refusal, so any bucket gap it shows is a property of the tasks.
"""

import json
import re
from pathlib import Path

from bfcl_eval.constants.enums import ReturnFormat
from bfcl_eval.model_handler.utils import default_decode_ast_prompting

from stats import compare, format_row

RESULTS = Path(r"D:\ToolCallFT\results")
PROMPTS = Path(r"D:\ToolCallFT\prompts")

CALL_CATS = ["simple_python", "multiple", "parallel", "parallel_multiple"]

# Same tokenisation the refusal builder used, so the probe splits tasks on
# exactly the criterion the training filter applied.
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_][\w.\-]*)"')
WORD_RE = re.compile(r"[a-z]+")

# Models to compare. The calls-only arm is the control.
REFUSAL_MODELS = ["tuned_refusal", "tuned_fixed", "tuned_fixed_s20260810", "tuned_fixed_s20260811"]
CONTROL_MODELS = ["tuned", "tuned_s20260809", "tuned_s20260810", "tuned_s20260811"]


def tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_.\-]", name.lower()) if len(t) > 2}


def emitted_call(raw: str) -> bool:
    try:
        decoded = default_decode_ast_prompting(raw, ReturnFormat.PYTHON, False)
    except Exception:
        return False
    return bool(decoded) and not (isinstance(decoded, list) and len(decoded) == 0)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n")]


def build_buckets() -> dict[str, str]:
    """task_id -> 'disjoint' | 'overlapping', by tool-name vs request vocabulary."""
    buckets = {}
    for cat in CALL_CATS:
        for row in read_jsonl(PROMPTS / f"{cat}.json"):
            sys_txt = row["messages"][0]["content"]
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            tool_tok = {t for n in TOOL_NAME_RE.findall(sys_txt) for t in tokens(n)}
            user_tok = {w for w in WORD_RE.findall(user.lower()) if len(w) > 2}
            buckets[row["id"]] = "overlapping" if (tool_tok & user_tok) else "disjoint"
    return buckets


def emission_by_bucket(model: str, buckets: dict[str, str]) -> dict[str, list[int]]:
    out = {"disjoint": [], "overlapping": []}
    for cat in CALL_CATS:
        rf = RESULTS / model / f"BFCL_v4_{cat}_result.json"
        if not rf.exists():
            return {}
        for row in read_jsonl(rf):
            b = buckets.get(row["id"])
            if b:
                out[b].append(int(emitted_call(row["result"])))
    return out


def main() -> int:
    buckets = build_buckets()
    n_dis = sum(1 for v in buckets.values() if v == "disjoint")
    print(f"call tasks: {len(buckets)}  disjoint {n_dis}  overlapping {len(buckets) - n_dis}")
    if n_dis < 30:
        print("WARNING: disjoint bucket is small; treat the comparison as indicative")

    print("\n" + "=" * 96)
    print("CALL EMISSION RATE on call tasks (higher = model still calls; lower = it declined)")
    print("=" * 96)
    print(f"{'model':<26} {'disjoint':>18} {'overlapping':>18} {'gap':>10}")
    print("-" * 96)

    gaps = {}
    for group, models in (("calls-only (control)", CONTROL_MODELS), ("refusal-trained", REFUSAL_MODELS)):
        print(f"\n[{group}]")
        for m in models:
            e = emission_by_bucket(m, buckets)
            if not e:
                print(f"{m:<26} {'(no results)':>18}")
                continue
            d = sum(e["disjoint"]) / max(1, len(e["disjoint"]))
            o = sum(e["overlapping"]) / max(1, len(e["overlapping"]))
            gaps.setdefault(group, []).append(d - o)
            print(f"{m:<26} {d:>17.2%} {o:>17.2%} {d - o:>+9.2%}")

    print("\n" + "-" * 96)
    for group, vals in gaps.items():
        mean = sum(vals) / len(vals)
        print(f"{group:<26} mean gap {mean:+.2%}  range [{min(vals):+.2%}, {max(vals):+.2%}]")

    if len(gaps) == 2:
        ctrl = sum(gaps["calls-only (control)"]) / len(gaps["calls-only (control)"])
        ref = sum(gaps["refusal-trained"]) / len(gaps["refusal-trained"])
        excess = ref - ctrl
        print(f"\nEXCESS gap attributable to refusal training: {excess:+.2%}")
        print("  (control gap is a property of the tasks; only the excess implicates training)")
        verdict = (
            "no evidence of the lexical shortcut" if excess > -0.02 else
            "possible shortcut -- refusal models decline disproportionately on disjoint tasks"
        )
        print(f"  verdict: {verdict}")

    # Paired test on the one comparison that matters most: best refusal model vs
    # its own control, restricted to disjoint tasks.
    print("\n" + "=" * 96)
    print("PAIRED on disjoint call tasks only: tuned_s20260809 -> tuned_fixed (same seed)")
    print("=" * 96)
    a, b = emission_by_bucket("tuned_s20260809", buckets), emission_by_bucket("tuned_fixed", buckets)
    if a and b and len(a["disjoint"]) == len(b["disjoint"]):
        print(format_row(compare("disjoint_emission", a["disjoint"], b["disjoint"]),
                         labels=("callsonly", "swapped")))
        print(format_row(compare("overlap_emission", a["overlapping"], b["overlapping"]),
                         labels=("callsonly", "swapped")))

    out = {"buckets": {"disjoint": n_dis, "overlapping": len(buckets) - n_dis}, "gaps": gaps}
    (RESULTS / "shortcut_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'shortcut_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
