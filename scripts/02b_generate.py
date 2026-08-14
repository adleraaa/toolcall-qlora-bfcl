"""Generate BFCL responses locally. Runs in .venv (torch). Consumes 02a's JSON.

Writes result files in BFCL's own layout so the official evaluator can score
them: result/<model_tag>/BFCL_v4_<category>_result.json, one
{"id": ..., "result": ...} object per line.
"""

import argparse
import json
import time
from pathlib import Path

from model_runner import generate, load

PROMPTS = Path(r"D:\ToolCallFT\prompts")
RESULTS = Path(r"D:\ToolCallFT\results")

# `base` loads no adapter. The two tuned variants differ only in training data:
# `tuned` saw 4,920 tool calls, `tuned_refusal` saw those plus 1,200 refusals.
ADAPTERS = {
    "tuned": Path(r"D:\ToolCallFT\runs\qwen2.5-1.5b-toolace-qlora\adapter"),
    "tuned_refusal": Path(r"D:\ToolCallFT\runs\qwen2.5-1.5b-toolace-refusal-qlora\adapter"),
    # Same row count as `tuned` (4,920), with 1,200 calls swapped for the same
    # 1,200 refusals `tuned_refusal` saw -- isolates composition from volume.
    "tuned_fixed": Path(r"D:\ToolCallFT\runs\qwen2.5-1.5b-toolace-fixed-qlora\adapter"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    # Free-form so multi-seed runs can use their own tags without editing this
    # file. "base" means no adapter; a name in ADAPTERS resolves to its path;
    # anything else must supply --adapter explicitly.
    ap.add_argument("--model", required=True, help="output tag under results/")
    ap.add_argument("--adapter", default="", help="adapter dir; overrides ADAPTERS lookup")
    # Every run so far has been NF4 on both sides, which controls quantisation
    # away. Running the same weights in bf16 measures it instead.
    ap.add_argument("--precision", default="nf4", choices=["nf4", "bf16"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="per-category cap, for smoke tests")
    ap.add_argument("--categories", default="")
    args = ap.parse_args()

    # `base` and any `base_*` variant (e.g. base_bf16) run without an adapter.
    # Anything else must resolve to one, so a mistyped tag fails loudly instead
    # of silently regenerating the baseline under a new name.
    if args.model == "base" or args.model.startswith("base_"):
        adapter = None
    elif args.adapter:
        adapter = Path(args.adapter)
    else:
        adapter = ADAPTERS.get(args.model)
        if adapter is None:
            raise SystemExit(
                f"unknown model tag '{args.model}' -- pass --adapter, or use one of: "
                f"base, {', '.join(ADAPTERS)}"
            )
    if adapter is not None and not adapter.exists():
        raise SystemExit(f"no adapter at {adapter} -- run 03_train_qlora.py first")

    files = sorted(PROMPTS.glob("*.json"))
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise SystemExit("no prompt files -- run 02a_build_prompts.py first")

    model, tok = load(str(adapter) if adapter else None, precision=args.precision)
    out_dir = RESULTS / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = time.time()
    for path in files:
        cat = path.stem
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n")]
        if args.limit:
            rows = rows[: args.limit]

        # Sorting by length groups similar sizes into a batch, so padding is not
        # dominated by one long prompt. Order is restored before writing.
        texts = [
            tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            for r in rows
        ]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        ordered_texts = [texts[i] for i in order]

        print(f"\n[{cat}] {len(rows)} prompts, model={args.model}")
        started = time.time()
        outs_sorted = generate(
            model,
            tok,
            ordered_texts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        outs = [None] * len(rows)
        for slot, text in zip(order, outs_sorted):
            outs[slot] = text

        dst = out_dir / f"BFCL_v4_{cat}_result.json"
        with dst.open("w", encoding="utf-8") as fh:
            for row, text in zip(rows, outs):
                fh.write(json.dumps({"id": row["id"], "result": text.strip()}, ensure_ascii=False) + "\n")
        mins = (time.time() - started) / 60
        print(f"[{cat}] done in {mins:.1f} min -> {dst.name}")

    print(f"\nall categories done in {(time.time() - overall) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
