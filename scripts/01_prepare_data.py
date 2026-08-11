"""Build a single-turn tool-calling SFT set from ToolACE (Apache-2.0).

Why ToolACE and not xLAM: xLAM-60k is gated (needs a HF account to accept terms)
and CC-BY-4.0, which complicates publishing the adapter. ToolACE is Apache-2.0
and ungated, and its assistant targets are already written as
`[func(arg=val), ...]` -- the exact surface the BFCL AST checker parses. Training
target and eval protocol therefore share one format, so a score change measures
the model, not a format-conversion bug.

Scope: single-turn only (one system, one user, one assistant tool call). That
matches the BFCL non-live categories we score (simple / multiple / parallel /
parallel_multiple), and keeps sequences short enough for 8 GiB of VRAM.
"""

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\ToolCallFT\hf_cache")

from datasets import load_dataset  # noqa: E402

OUT = Path(r"D:\ToolCallFT\data")
SEED = 20260809
VAL_FRACTION = 0.05

# A ToolACE assistant turn is a tool call only when it looks like `[name(...)]`.
# Plain prose replies exist in the corpus too; those teach chat, not tool use.
CALL_RE = re.compile(r"^\s*\[\s*[A-Za-z_][\w.\-]*\s*\(.*\)\s*\]\s*$", re.DOTALL)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading Team-ACE/ToolACE ...")
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    print(f"raw rows: {len(ds):,}")

    # ToolACE conversations are multi-turn (user -> asst -> tool -> asst -> ...).
    # Only the OPENING (user, assistant) pair is usable as a standalone example:
    # every later user turn is conditioned on tool results earlier in the thread,
    # so lifting one out without that context would pair a prompt with an answer
    # it cannot justify -- mislabelled data that quietly teaches guessing.
    # Taking the first pair yields one clean example per conversation and matches
    # the single-turn BFCL categories being scored.
    reasons: Counter[str] = Counter()
    kept = []
    for row in ds:
        convo = row.get("conversations") or []
        if len(convo) < 2:
            reasons["too_few_turns"] += 1
            continue
        first, second = convo[0], convo[1]
        if first.get("from") != "user" or second.get("from") != "assistant":
            reasons["does_not_open_user_then_assistant"] += 1
            continue
        if not CALL_RE.match(second.get("value", "")):
            reasons["opening_reply_is_not_a_tool_call"] += 1
            continue
        system = (row.get("system") or "").strip()
        if not system:
            reasons["no_system_prompt"] += 1
            continue
        kept.append(
            {
                "system": system,
                "user": first["value"].strip(),
                "assistant": second["value"].strip(),
                "source_turns": len(convo),
            }
        )

    print(f"kept: {len(kept):,}  ({len(kept) / len(ds):.1%} of corpus)")
    print("dropped:")
    for reason, n in reasons.most_common():
        print(f"  {reason:<20} {n:>8,}")

    if not kept:
        print("FAIL: filter kept nothing -- schema probably changed.")
        return 1

    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * VAL_FRACTION))
    val, train = kept[:n_val], kept[n_val:]

    for name, rows in (("train", train), ("val", val)):
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path}  ({len(rows):,} rows)")

    # Length stats drive max_seq_len; guessing it wastes VRAM or truncates targets.
    lens = [len(r["system"]) + len(r["user"]) + len(r["assistant"]) for r in kept]
    lens.sort()
    print("\nchar length percentiles (system+user+assistant):")
    for p in (50, 90, 95, 99, 100):
        print(f"  p{p:<3} {lens[min(len(lens) - 1, p * len(lens) // 100)]:>7,}")

    print("\nexample target:")
    print(" ", kept[0]["assistant"][:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
