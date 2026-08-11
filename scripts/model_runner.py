"""Local generation for BFCL, in the same 4-bit regime the model was trained in.

Kept separate from BFCL specifics so the model side can be tested on its own.

The base model is loaded in NF4 rather than bf16 on purpose. The tuned model is a
LoRA adapter over an NF4 base, so scoring the baseline in full precision would
confound the comparison: part of any measured delta would just be the difference
between a quantised and an unquantised model. Both sides run 4-bit; the only
thing that changes between them is the adapter.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HOME", r"D:\ToolCallFT\hf_cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load(adapter_dir: str | None = None):
    """Load the base model in NF4, optionally with a LoRA adapter attached."""
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"  # required for correct batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"loaded adapter: {adapter_dir}")
    else:
        print("no adapter: baseline")

    model.eval()
    model.config.use_cache = True
    return model, tok


@torch.no_grad()
def generate(
    model,
    tok,
    prompts: list[str],
    *,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    progress_every: int = 20,
) -> list[str]:
    """Greedy-decode a list of already-templated prompts.

    Greedy, not sampled: the benchmark is scored on exact structural match, so
    sampling would add run-to-run variance that has nothing to do with the model
    difference we are trying to measure. Same seed, same decode, both sides.
    """
    outputs: list[str] = []
    started = time.time()

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        try:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            # One oversized batch should not kill a 40-minute run.
            torch.cuda.empty_cache()
            print(f"  OOM at {start}, retrying one at a time")
            out_texts = []
            for single in chunk:
                e = tok([single], return_tensors="pt", truncation=True, max_length=4096)
                e = {k: v.to(model.device) for k, v in e.items()}
                o = model.generate(
                    **e,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                out_texts.append(tok.decode(o[0][e["input_ids"].shape[1] :], skip_special_tokens=True))
                torch.cuda.empty_cache()
            outputs.extend(out_texts)
            continue

        prompt_len = enc["input_ids"].shape[1]
        for row in out:
            outputs.append(tok.decode(row[prompt_len:], skip_special_tokens=True))

        done = len(outputs)
        if done % progress_every < batch_size:
            rate = done / max(1e-6, time.time() - started)
            eta = (len(prompts) - done) / max(1e-6, rate) / 60
            print(f"  {done}/{len(prompts)}  {rate:.2f} it/s  eta {eta:.1f} min", flush=True)

    return outputs


def apply_chat(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
