"""Score generated responses with BFCL's own checker. Runs in .venv-bfcl.

Reuses BFCL's `default_decode_ast_prompting` and `ast_checker` directly rather
than reimplementing AST comparison, so the numbers mean the same thing they mean
on the public leaderboard. Argument matching in particular is subtler than it
looks -- ground truth lists several acceptable values per parameter, including ""
for "may be omitted" -- and a hand-rolled comparison would quietly disagree.

Emits per-item 0/1 so the paired statistics in 05_report.py have something to
pair, plus a format-compliance count that keeps "could not parse at all" visible
as its own failure mode instead of blending into the accuracy number.

`--self-test` checks the wiring before trusting any real score: feeding the
ground truth back in must score ~100%, and feeding prose in must score 0% on
call categories and 100% on irrelevance. A pipeline that cannot do that is
measuring nothing.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.eval_config import PROMPT_PATH
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.model_handler.utils import default_decode_ast_prompting

RESULTS = Path(r"D:\ToolCallFT\results")
DATA = Path(PROMPT_PATH)

CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]

# ast_checker looks this up in MODEL_CONFIG_MAPPING purely to read one flag,
# `underscore_to_dot`. An unregistered name raises KeyError *inside* the checker,
# which -- if swallowed -- silently scores every single entry as wrong while the
# run still looks healthy. So this must be a registered prompting (non-FC) OSS
# model, matched to how we actually prompt: same family, same non-FC style.
BFCL_MODEL_TAG = "qwen3-1.7b"

# Counts checker-internal failures separately from genuine wrong answers. A
# nonzero value here means the harness is broken, not that the model is bad, and
# the two must never be allowed to look alike.
CHECKER_ERRORS: dict[str, int] = defaultdict(int)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n")]


def is_empty_output(decoded) -> bool:
    return not decoded or (isinstance(decoded, list) and len(decoded) == 0)


def score_entry(raw: str, prompt_entry: dict, answer_item, category: str) -> tuple[int, bool]:
    """Return (correct, parsed_ok)."""
    try:
        decoded = default_decode_ast_prompting(raw, ReturnFormat.PYTHON, False)
        parsed_ok = not is_empty_output(decoded)
    except Exception:
        # A model answering in prose fails to decode; for irrelevance that IS
        # the correct behaviour, everywhere else it is a miss.
        return (1 if "irrelevance" in category else 0, False)

    if "irrelevance" in category:
        # Correct precisely when the model declined to emit a call.
        return (0 if parsed_ok else 1, parsed_ok)

    if not parsed_ok:
        return 0, False

    try:
        verdict = ast_checker(
            prompt_entry["function"],
            decoded,
            answer_item,
            Language.PYTHON,
            category,
            BFCL_MODEL_TAG,
        )
        return (1 if verdict.get("valid") else 0), True
    except Exception as exc:
        CHECKER_ERRORS[f"{type(exc).__name__}: {exc}"] += 1
        return 0, True


def synth_perfect(answer_item) -> str:
    """Rebuild a canonical call string from ground truth, for the self-test."""
    calls = []
    for call in answer_item:
        for fname, params in call.items():
            args = []
            for pname, values in params.items():
                if not values:
                    continue
                val = values[0]
                if val == "" and len(values) > 1:
                    val = values[1]
                if val == "":
                    continue  # optional parameter, omit it
                args.append(f"{pname}={val!r}")
            calls.append(f"{fname}({', '.join(args)})")
    return "[" + ", ".join(calls) + "]"


def run_self_test() -> int:
    print("=== self-test: ground truth fed back in should score ~100% ===")
    ok = True
    for cat in CATEGORIES:
        prompts = read_jsonl(DATA / f"BFCL_v4_{cat}.json")
        if "irrelevance" in cat:
            prose = "None of the provided functions can answer this question."
            hits = sum(score_entry(prose, p, None, cat)[0] for p in prompts)
            rate = hits / len(prompts)
            print(f"  {cat:<20} prose -> {rate:.1%} correct (expect 100%)")
            ok &= rate > 0.99
            continue

        answers = {a["id"]: a["ground_truth"] for a in read_jsonl(DATA / "possible_answer" / f"BFCL_v4_{cat}.json")}
        hits = sum(score_entry(synth_perfect(answers[p["id"]]), p, answers[p["id"]], cat)[0] for p in prompts)
        rate = hits / len(prompts)
        flag = "" if rate >= 0.90 else "   <-- TOO LOW"
        print(f"  {cat:<20} ground truth -> {rate:.1%} correct (expect >=90%){flag}")
        # The upper bound matters as much as the lower one: a harness that scores
        # the correct answer as wrong is broken in the direction that looks like
        # "the model underperformed", which is exactly the failure that survives
        # review. Assert it explicitly instead of only checking prose -> 0%.
        ok &= rate >= 0.90

        prose = "I am not able to help with that request."
        prose_hits = sum(score_entry(prose, p, answers[p["id"]], cat)[0] for p in prompts)
        print(f"  {cat:<20} prose        -> {prose_hits / len(prompts):.1%} correct (expect 0%)")
        ok &= prose_hits == 0

    if CHECKER_ERRORS:
        ok = False
        print("\nCHECKER ERRORS (harness bug, not model error):")
        for msg, n in sorted(CHECKER_ERRORS.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {n:>6}x  {msg[:120]}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    # Free-form to accept multi-seed tags; it only selects a results/ subdir.
    ap.add_argument("--model", help="tag under results/ to score")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.model:
        raise SystemExit("--model required unless --self-test")

    res_dir = RESULTS / args.model
    scored: dict[str, dict[str, int]] = {}
    compliance: dict[str, dict] = {}

    for cat in CATEGORIES:
        rf = res_dir / f"BFCL_v4_{cat}_result.json"
        if not rf.exists():
            print(f"SKIP {cat}: no result file")
            continue
        results = read_jsonl(rf)
        prompts = {p["id"]: p for p in read_jsonl(DATA / f"BFCL_v4_{cat}.json")}
        answers = {}
        pa = DATA / "possible_answer" / f"BFCL_v4_{cat}.json"
        if pa.exists():
            answers = {a["id"]: a["ground_truth"] for a in read_jsonl(pa)}

        per_item, parsed = {}, 0
        for row in results:
            correct, parsed_ok = score_entry(
                row["result"], prompts[row["id"]], answers.get(row["id"]), cat
            )
            per_item[row["id"]] = correct
            parsed += int(parsed_ok)

        scored[cat] = per_item
        compliance[cat] = {"n": len(results), "parsed": parsed, "parse_rate": parsed / len(results)}
        acc = sum(per_item.values()) / len(per_item)
        print(f"{cat:<20} n={len(per_item):>4}  acc={acc:7.2%}  parse_rate={parsed / len(results):7.2%}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"scored_{args.model}.json").write_text(json.dumps(scored, indent=2), encoding="utf-8")
    comp_path = RESULTS / "format_compliance.json"
    existing = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else {}
    existing[args.model] = compliance
    comp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwrote scored_{args.model}.json and format_compliance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
