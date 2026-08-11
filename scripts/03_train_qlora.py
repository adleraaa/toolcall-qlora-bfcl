"""QLoRA SFT of Qwen2.5-1.5B-Instruct for tool calling, on one 8 GiB laptop GPU.

Stack is deliberately plain: transformers + peft + bitsandbytes + trl. Unsloth is
faster and lighter, but it is a wrapper with its own kernel patches; on a
Blackwell/Windows box the plain stack has fewer moving parts to debug, and 1.5B
in NF4 fits with room to spare.

Loss is computed on the assistant turn only. Training on the prompt would let the
model spend capacity memorising the (very long) tool schemas in the system block
instead of learning to emit the call.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\ToolCallFT\hf_cache")

# NOT setting expandable_segments here, deliberately. It was tried as an OOM fix
# and made things worse in a way worth recording: on this Windows/WDDM box it let
# the allocator reserve far beyond physical VRAM (19 GiB reserved on an 8 GiB
# card) by spilling into host memory, so the run stopped crashing and started
# crawling -- 38 s/step against a normal 5 s/step, turning a 1-hour job into a
# 6.6-hour one. Silent 7x slowdowns are worse than loud failures; the crash at
# least announced itself.
#
# The headroom problem is instead handled by not running evaluation during
# training (see --eval-strategy) and by returning cached blocks periodically.

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class VramGuard(TrainerCallback):
    """Log VRAM every N steps. Observes only -- it does not touch the allocator.

    Two refusal-mixed runs died in backward at step 190 and 193 on a config that
    completed at 615 steps. Sequence length was ruled out by measurement: the
    mixed set's length distribution matches run 1's almost exactly (max 2855 in
    both) and the rows around the crash top out at 1617 tokens. What is left is
    headroom -- run 1 peaked at 5.91 GiB allocated on a card with ~7.6 GiB free,
    and reserved runs above allocated, so it finished with almost nothing spare.

    This callback used to also call `empty_cache()` every N steps, on the theory
    that returning free blocks would relieve pressure. The readings say it did
    the opposite. Reserved memory sawtoothed in time with the callback --
    9.6, 16.2, 9.1, 12.5, 7.5, 16.4 GiB on an 8 GiB card -- because releasing
    blocks forces the allocator to re-request them, and on WDDM the driver is
    free to satisfy the new request out of shared host memory. Run 1 completed
    without this call; the run that added it died at step 193. So the "safety
    measure" was manufacturing the fragmentation it was meant to prevent, and
    it is gone.

    What stays is the logging, which has already paid for itself twice: it
    caught expandable_segments quietly paging through host memory (19 GiB
    reserved, 38 s/step against a normal 5 s/step -- a 6.6-hour job wearing the
    face of a healthy one), and then caught this. Neither failure announced
    itself in the loss curve.
    """

    def __init__(self, every: int = 50):
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every:
            return
        print(
            f"  [vram] step {state.global_step:>4}  "
            f"alloc {torch.cuda.memory_allocated() / 1024**3:.2f}  "
            f"reserved {torch.cuda.memory_reserved() / 1024**3:.2f}  "
            f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
            flush=True,
        )

DATA = Path(r"D:\ToolCallFT\data")
RUNS = Path(r"D:\ToolCallFT\runs")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_dataset(rows: list[dict], tok, max_len: int) -> Dataset:
    """Tokenise into input_ids + labels, masking everything before the answer."""

    def encode(row: dict) -> dict | None:
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = row["assistant"] + tok.eos_token

        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]

        # Truncating the answer would train the model to emit broken calls, so
        # drop the example instead of silently corrupting the target.
        if len(p_ids) + len(a_ids) > max_len:
            return None

        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids[:]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    encoded, dropped = [], 0
    for row in rows:
        item = encode(row)
        if item is None:
            dropped += 1
        else:
            encoded.append(item)
    print(f"  encoded {len(encoded):,}, dropped {dropped:,} over max_len={max_len}")
    return Dataset.from_list(encoded)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="qwen2.5-1.5b-toolace-qlora")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--train-rows", type=int, default=8000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    # Swappable so the refusal-mixed ablation runs through this exact script with
    # every other hyperparameter untouched -- the dataset stays the one variable.
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--val-file", default="val.jsonl")
    # Held-out evaluation during training is pure instrumentation: it runs under
    # no_grad and never touches the weights, so disabling it cannot change the
    # adapter this script produces. It can, however, decide whether the run
    # finishes -- a forward pass over the eval set allocates a logits tensor of
    # seq_len x 151,936 vocab on top of the training peak, which is what pushed
    # the refusal-mixed run over an 8 GiB card twice. Kept as a flag rather than
    # deleted so run 1's exact configuration stays reproducible.
    ap.add_argument("--eval-strategy", default="steps", choices=["steps", "no"])
    # Round every batch up to a multiple of this many tokens.
    #
    # With batch size 1 and no rounding, every step allocates a logits tensor of
    # seq_len x 151,936 -- and seq_len is different almost every step, so the
    # caching allocator accumulates thousands of mutually unusable free blocks.
    # Measured: reserved climbed 14.0 -> 16.9 -> 19.8 GiB on an 8 GiB card while
    # peak live allocation never exceeded 5.34 GiB, then died. Rounding to a
    # multiple collapses those thousands of shapes into a handful that can
    # actually be reused.
    #
    # Padding is mathematically inert here: pad positions carry attention_mask 0
    # and label -100, so they contribute nothing to the loss and nothing to any
    # gradient. It costs a little compute on short sequences and changes no
    # learned weight.
    ap.add_argument("--pad-multiple", type=int, default=0, help="0 = pad to exact batch max")
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    out_dir = RUNS / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_rows = load_jsonl(DATA / args.train_file)[: args.train_rows]
    val_rows = load_jsonl(DATA / args.val_file)[:500]
    kinds = Counter(r.get("kind", "call") for r in train_rows)
    print(f"train rows {len(train_rows):,} | val rows {len(val_rows):,}")
    print(f"train composition: {dict(kinds)}")
    train_ds = build_dataset(train_rows, tok, args.max_len)
    val_ds = build_dataset(val_rows, tok, args.max_len)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy=args.eval_strategy,
            eval_steps=100,
            save_strategy="epoch",
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=0.3,
            report_to=[],
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(
            tok,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=args.pad_multiple or None,
        ),
        callbacks=[VramGuard(every=25)],
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    minutes = (time.time() - started) / 60
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    # Cost is part of the result, not a footnote.
    summary = {
        "base_model": BASE_MODEL,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "eval_strategy": args.eval_strategy,
        "pad_multiple": args.pad_multiple,
        "seed": args.seed,
        "train_composition": dict(kinds),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "max_len": args.max_len,
        "effective_batch": args.grad_accum,
        "train_runtime_min": round(minutes, 2),
        "peak_vram_gib": round(peak_gib, 2),
        "final_train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\ndone in {minutes:.1f} min | peak VRAM {peak_gib:.2f} GiB")
    print(f"adapter -> {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
