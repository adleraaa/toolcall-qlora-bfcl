"""Build BFCL prompts using BFCL's OWN prompt construction. Runs in .venv-bfcl.

Why this is a separate step in a separate interpreter: prompt construction needs
`bfcl_eval`, generation needs torch, and letting pip resolve both in one
environment risks bfcl's pins downgrading the transformers/torch build that was
just verified on this Blackwell GPU. So the two never share an interpreter --
they hand off through a JSON file.

Why not hand-roll the prompt: the score is only comparable to the public
leaderboard if the model sees exactly what BFCL would have shown it. This calls
`system_prompt_pre_processing_chat_model`, the same function BFCL's own
prompting handlers use, so the system prompt is byte-identical to an official run.
"""

import json
from pathlib import Path

from bfcl_eval.constants.eval_config import PROMPT_PATH
from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

OUT = Path(r"D:\ToolCallFT\prompts")

# Single-turn, AST-scored, Python-only. These need no execution sandbox and no
# multi-turn state machine, which is what makes them runnable off a local
# generation loop. `irrelevance` is included on purpose: it scores whether the
# model correctly declines to call anything, and a model fine-tuned to always
# emit a call can regress there -- a real risk this setup should be able to see
# rather than hide.
CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = Path(PROMPT_PATH)
    print(f"bfcl data dir: {data_dir}")

    grand_total = 0
    for cat in CATEGORIES:
        src = data_dir / f"BFCL_v4_{cat}.json"
        if not src.exists():
            print(f"SKIP {cat}: {src} not found")
            continue

        rows = []
        for line in src.read_text(encoding="utf-8").strip().split("\n"):
            entry = json.loads(line)

            # question is [turn][message]; single-turn categories have one turn.
            turns = entry["question"]
            assert len(turns) == 1, f"{entry['id']} is not single-turn ({len(turns)})"

            # This mutates its input, so hand it a copy -- otherwise the entry's
            # own question list gets a system prompt spliced into it.
            messages = system_prompt_pre_processing_chat_model(
                [dict(m) for m in turns[0]],
                entry["function"],
                entry["id"],
            )
            rows.append({"id": entry["id"], "messages": messages})

        dst = OUT / f"{cat}.json"
        with dst.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        grand_total += len(rows)
        print(f"{cat:<20} {len(rows):>5} prompts -> {dst.name}")

    print(f"\ntotal {grand_total} prompts")

    sample = json.loads((OUT / CATEGORIES[0]).with_suffix(".json").read_text(encoding="utf-8").split("\n")[0])
    print(f"\nroles in first entry: {[m['role'] for m in sample['messages']]}")
    print("--- system prompt (first 700 chars) ---")
    print(sample["messages"][0]["content"][:700])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
