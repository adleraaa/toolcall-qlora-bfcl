"""Mix refusal examples into the SFT set, built entirely from ToolACE itself.

WHY THIS EXISTS
The first run gained +8.5% / +8.0% on the call categories and lost 55.4 points on
`irrelevance`, with a +0/-133 flip pattern -- pure one-way collapse. The cause is
visible in the corpus: every one of the 4,916 training targets is a tool call,
because that is what a tool-calling dataset contains. The model learned the
unconditional prior "functions are present, therefore call one".

The telling detail is that ToolACE's own system prompt already says:

    "If none of the function can be used, point it out."

So the protocol always specified a refusal branch; the data just never
demonstrated it. This script supplies the missing half. It is not teaching new
behaviour -- it is making the demonstrations agree with the instructions the
model is already given.

HOW THE REFUSALS ARE BUILT (and why not some other way)
Constructed by CROSS-PAIRING inside ToolACE: take example i's system block (its
tool schemas) and example j's user request, keeping the pair only when j's
request provably cannot be served by i's tools. The assistant target becomes a
short natural-language "these tools don't cover it".

No external dataset, and above all NOTHING from BFCL. Sourcing refusals from the
eval set -- or from anything derived from it -- would be train/test contamination
and would make the recovered irrelevance score meaningless.

A general-purpose chat corpus (Dolly et al.) was considered and rejected on
distribution grounds: BFCL `irrelevance` prompts are *task* requests aimed at a
mismatched tool ("compute the area of a triangle", offered only a BMI function),
not open-domain chitchat. Training on trivia would teach "decline small talk",
which is a different behaviour than the one being measured.

MIXING RATIO
1,200 refusals against 4,920 call examples => 19.6% of the mixed set. Chosen to
approximate the eval set's own irrelevance share (240/1240 = 19.4%), so the
training prior roughly matches the deployment frequency rather than being tuned
to whatever number happens to score best.

Train refusals are paired only from train rows and val refusals only from val
rows, so val stays genuinely held out.
"""

import argparse
import json
import random
import re
from pathlib import Path

DATA = Path(r"D:\ToolCallFT\data")
SEED = 20260809
N_TRAIN_REFUSALS = 1200
VAL_REFUSAL_FRACTION = 0.196  # match the train mix

# Tool names as advertised in the system block. Requires a STRING value, so a
# parameter literally named `name` (which appears as `"name": {`) cannot match.
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_][\w.\-]*)"')
# Top-level calls in `[f(...), g(...)]` -- anchored to `[` or `,` so nested
# parentheses inside argument values are not mistaken for callees.
CALLEE_RE = re.compile(r'(?:^\s*\[|,)\s*([A-Za-z_][\w.\-]*)\s*\(')

# Splitting names into word tokens lets us reject pairs that merely *look*
# unrelated. `calculate_discounted_tax` vs `calculate_tax_refund` share
# {calculate, tax}; that pair is plausibly serviceable and must not be labelled
# a refusal. Conservative on purpose: a false refusal label is worse than a
# smaller dataset, since it would teach the model to decline things it can do.
def tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_.\-]", name.lower()) if len(t) > 2}


REFUSAL_TEMPLATES = [
    "None of the available functions can do that. {tools} {does} not cover this request.",
    "I can't complete this with the functions provided -- {tools} {is_} unrelated to what you're asking.",
    "That request is outside what these tools support. The only function available {is_} {tools}.",
    "No suitable function here. {tools} {does} not address this request.",
    "I don't have a function for that. What's available {is_} {tools}, which {does} not apply.",
    "This can't be answered with the given tools; {tools} {is_} for a different purpose.",
    "None of the provided functions match this request. Available: {tools}.",
    "The available function set doesn't cover this. {tools} {does} not do what you're asking.",
    "I'm unable to handle that with the current tools -- {tools} {is_} not applicable.",
    "That falls outside these functions' scope. {tools} cannot serve this request.",
    "No function available for this. {tools} {does} not fit the request.",
    "This request doesn't map to any provided function; {tools} {is_} the only option and {does} not apply.",
]


# Part of ToolACE flattens a multi-turn thread into one user string prefixed with
# "Role definition:" / "Historical dialogue:". Those are multi-turn examples in a
# single-turn wrapper -- outside the scope 01_prepare_data.py declares, and
# nothing like the short standalone requests being scored. They are excluded as a
# SOURCE OF REFUSAL REQUESTS only; the call examples are passed through untouched,
# so the sole variable in this ablation stays "refusals added".
MULTITURN_MARKERS = ("Role definition:", "Historical dialogue:", "Inquirer:")
MAX_REQUEST_CHARS = 400


def is_standalone_request(user: str) -> bool:
    return len(user) <= MAX_REQUEST_CHARS and not any(m in user for m in MULTITURN_MARKERS)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path.name:<20} {len(rows):>6,} rows")


def phrase(tool_names: list[str]) -> tuple[str, str, str]:
    """Render tool names plus verb agreement, so targets read as real sentences."""
    quoted = [f"`{t}`" for t in tool_names]
    if len(quoted) == 1:
        return quoted[0], "does", "is"
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}", "do", "are"
    return f"{', '.join(quoted[:-1])}, and {quoted[-1]}", "do", "are"


def build_refusals(rows: list[dict], n_wanted: int, rng: random.Random, label: str) -> list[dict]:
    enriched, self_consistent = [], 0
    for r in rows:
        tools = sorted(set(TOOL_NAME_RE.findall(r["system"])))
        callees = set(CALLEE_RE.findall(r["assistant"]))
        if not tools or not callees:
            continue
        # Self-check on the extractors: in a well-formed example every function
        # the target calls must be one the system block advertises. A low rate
        # here means the regexes are wrong and every downstream pair is suspect.
        if callees <= set(tools):
            self_consistent += 1
        enriched.append({"row": r, "tools": tools, "callees": callees,
                         "tok": {t for c in callees for t in tokens(c)},
                         "standalone": is_standalone_request(r["user"])})

    rate = self_consistent / max(1, len(enriched))
    n_standalone = sum(1 for e in enriched if e["standalone"])
    print(f"[{label}] usable rows {len(enriched):,} | extractor self-consistency {rate:.1%}")
    print(f"[{label}] standalone requests eligible as refusal sources: {n_standalone:,}")
    if rate < 0.90:
        raise SystemExit(
            f"FAIL: only {rate:.1%} of targets call a function their own system block lists. "
            "Tool-name/callee extraction is unreliable; refusal labels would be noise."
        )

    refusals, seen_users, attempts = [], set(), 0
    max_attempts = n_wanted * 200
    while len(refusals) < n_wanted and attempts < max_attempts:
        attempts += 1
        host = rng.choice(enriched)   # supplies the tools
        guest = rng.choice(enriched)  # supplies the request
        if host is guest or not guest["standalone"]:
            continue
        # The request must need a function the host does not advertise...
        if guest["callees"] & set(host["tools"]):
            continue
        # ...and must not be merely a renamed sibling of one it does.
        host_tok = {t for name in host["tools"] for t in tokens(name)}
        if host_tok & guest["tok"]:
            continue
        user = guest["row"]["user"]
        if user in seen_users:  # don't let one request dominate the refusal set
            continue
        seen_users.add(user)

        tools_str, does, is_ = phrase(host["tools"])
        refusals.append(
            {
                "system": host["row"]["system"],
                "user": user,
                "assistant": rng.choice(REFUSAL_TEMPLATES).format(
                    tools=tools_str, does=does, is_=is_
                ),
                "kind": "refusal",
            }
        )

    if len(refusals) < n_wanted:
        print(f"[{label}] WARNING: only built {len(refusals):,}/{n_wanted:,} "
              f"after {attempts:,} attempts (filters are strict)")
    return refusals


def main() -> int:
    ap = argparse.ArgumentParser()
    # Appending refusals grows the training set by 24%, so the first ablation
    # moved two things at once: composition AND volume. The irrelevance recovery
    # is immune to that -- no quantity of call-only data teaches declining, and
    # run 1 proves it by regressing on more of exactly that data -- but the
    # call-category gains cannot be attributed cleanly between the two.
    #
    # --fixed-total drops an equal number of call examples so the row count
    # matches run 1 exactly. Then the only difference between the two training
    # sets is that 1,200 call examples became 1,200 refusals, and any delta is
    # attributable to composition alone.
    ap.add_argument("--fixed-total", action="store_true",
                    help="replace calls with refusals instead of appending")
    # Seed drives refusal pairing AND which calls get dropped, so each seed is a
    # fully independent replicate rather than the same dataset trained twice.
    # Treating dataset construction as fixed would understate the variance the
    # multi-seed run exists to measure.
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    suffix = "fixed" if args.fixed_total else "mixed"
    if args.seed != SEED:
        suffix = f"{suffix}_s{args.seed}"
    rng = random.Random(args.seed)
    train = load_jsonl(DATA / "train.jsonl")
    val = load_jsonl(DATA / "val.jsonl")
    print(f"call examples: train {len(train):,} | val {len(val):,}")
    print(f"mode: {'replace (total held fixed)' if args.fixed_total else 'append'}\n")

    train_ref = build_refusals(train, N_TRAIN_REFUSALS, rng, "train")
    n_val_ref = max(1, round(len(val) * VAL_REFUSAL_FRACTION / (1 - VAL_REFUSAL_FRACTION)))
    val_ref = build_refusals(val, n_val_ref, rng, "val")

    for r in train + val:
        r.setdefault("kind", "call")

    if args.fixed_total:
        # Drop calls at random (fixed seed) rather than from either end -- the
        # file is already shuffled, but slicing would still correlate the dropped
        # set with position, and position correlates with nothing we want to bet
        # on. The refusals themselves are byte-identical to the append run, since
        # they are drawn from the same seeded stream before this branch.
        keep_train = len(train) - len(train_ref)
        keep_val = len(val) - len(val_ref)
        if keep_train <= 0 or keep_val <= 0:
            raise SystemExit("refusal count exceeds the call set; nothing left to train on")
        train = rng.sample(train, keep_train)
        val = rng.sample(val, keep_val)
        print(f"dropped {len(train_ref):,} train / {len(val_ref):,} val calls to hold the total fixed")

    mixed_train = train + train_ref
    mixed_val = val + val_ref
    # Shuffle so refusals are not clustered at the end of the epoch, where they
    # would act as a short fine-tune on top of the call examples rather than
    # being learned jointly with them.
    rng.shuffle(mixed_train)
    rng.shuffle(mixed_val)

    write_jsonl(DATA / f"train_{suffix}.jsonl", mixed_train)
    write_jsonl(DATA / f"val_{suffix}.jsonl", mixed_val)

    share = len(train_ref) / len(mixed_train)
    print(f"\ntrain mix: {len(train):,} calls + {len(train_ref):,} refusals "
          f"= {len(mixed_train):,} ({share:.1%} refusal)")
    print(f"val   mix: {len(val):,} calls + {len(val_ref):,} refusals "
          f"= {len(mixed_val):,} ({len(val_ref)/len(mixed_val):.1%} refusal)")
    print("eval-set irrelevance share for reference: 240/1240 = 19.4%")
    if args.fixed_total:
        print("run 1 trained on 4,920 rows -- this set must match for the "
              "comparison to isolate composition")

    print("\n--- sample refusal ---")
    s = train_ref[0]
    print("tools in system:", sorted(set(TOOL_NAME_RE.findall(s["system"]))))
    print("user:", s["user"][:160])
    print("target:", s["assistant"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
