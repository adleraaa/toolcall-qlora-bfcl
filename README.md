# Tool-Calling QLoRA: does fine-tuning a 1.5B model actually help on BFCL?

Fine-tunes `Qwen2.5-1.5B-Instruct` for function calling with QLoRA on a single
8 GiB laptop GPU, and measures whether it helped on the
[Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
using BFCL's own evaluator — with paired confidence intervals, a significance
test, and the training cost reported alongside the win.

**Headline:** the obvious fine-tune gained +8.5% on tool calling and lost 55
points of the model's ability to *decline* — a net **−7.10%**. Mixing in 1,200
constructed refusals recovers ~49 of those points, reproduced across three seeds
(+105/−1, +115/−1, +133/−1) with **no detectable cost to tool calling**, ending
at **+4.1% over base on average** (+4.92 / +4.35 / +3.06 by seed). A
volume-controlled ablation shows the safety recovery is entirely the refusal
data and the capability gain is entirely the data volume — the two are
orthogonal, so refusals should be **appended, not substituted** (3/3 seeds
favour appending, mean 2.42 points).

> Every number below is produced by `05_report.py` from `results/`; none is
> hand-typed. Paired bootstrap CIs, McNemar exact tests, per-item outcomes in
> `results/scored_*.json`.

## Why this is set up the way it is

**The format trap.** A base instruct model asked to call a function will happily
answer in JSON, or in prose, or in Python. BFCL's AST checker parses
`[func(arg=val)]`. So a naive baseline scores near zero, the fine-tune scores far
above it, and the headline delta is mostly *format compliance* wearing the
costume of *capability*. That is the single easiest way to produce an impressive
and meaningless number.

The fix here: baseline and fine-tuned model are prompted with the **same system
prompt protocol**, the one ToolACE uses, which states the output format
explicitly. Both models are told exactly how to answer. The delta that survives
is a capability delta, not a parsing artifact. `results/format_compliance.json`
reports what fraction of each model's outputs were even parseable, so the two
effects stay separable instead of being silently blended.

**Pairing.** Both models are scored on an identical, fixed task set, so the
per-item outcomes are paired. Reporting two independent intervals and checking
whether they overlap discards that pairing and is needlessly conservative.
Instead:

- per-model pass rate → **Wilson** interval (category sizes are in the hundreds
  and rates sit near 0/1, where Wald runs outside `[0,1]`)
- the delta → **paired bootstrap** over items, resampling task indices so each
  draw keeps a task's two outcomes together
- significance → **McNemar's exact test** on discordant pairs, binomial rather
  than chi-square because discordant counts get small per category

**Cost is a result.** Wall-clock, peak VRAM, and rows consumed are recorded in
`runs/*/train_summary.json` and reported next to the accuracy delta. A win that
costs an hour of a laptop GPU is a different claim than a win that costs a
cluster.

## Stack

| | |
|---|---|
| Base model | `Qwen2.5-1.5B-Instruct` (ungated) |
| Training data | [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), Apache-2.0, ungated |
| Method | QLoRA — NF4, double quant, bf16 compute, LoRA r=16 on all attention + MLP projections |
| Framework | PyTorch + transformers + peft + bitsandbytes |
| Benchmark | BFCL v4 — scored with `bfcl_eval`'s official `ast_checker`, imported directly rather than via the `bfcl evaluate` CLI (vLLM has no Windows build). `irrelevance` is scored by this repo's own rule: correct iff the model emits nothing parseable. |
| Hardware | RTX 5070 Laptop (Blackwell sm_120), 8 GiB VRAM |

ToolACE was chosen over the more common xLAM-60k for two reasons: xLAM is gated
and CC-BY-4.0, which complicates redistributing an adapter, and ToolACE's
assistant targets are already written in the `[func(arg=val)]` surface the BFCL
checker parses — so the training target and the eval protocol share one format
and a score change cannot be a conversion bug.

## Notes on the hardware

Blackwell (sm_120) needs a `cu128`/`cu129` PyTorch build; stock wheels detect the
GPU and then fail every kernel launch with *no kernel image is available*.
`00_env_check.py` exists because of exactly that: it refuses to pass on
`torch.cuda.is_available()` alone and runs real matmuls in fp32/fp16/bf16 first.
The build is pinned to **cu129, not cu130**, because cu130 breaks the
bitsandbytes wheels that QLoRA depends on.

vLLM has no native Windows support, which rules out BFCL's usual generation
backend. BFCL splits `generate` from `evaluate`, so generation runs locally
through transformers, with the models loaded in NF4 4-bit (bf16 only in the
quantisation comparison below), and the resulting response files are scored
with `bfcl_eval`'s `ast_checker`; `irrelevance` is scored by this repo's own
rule (see **Stack**). These numbers come from this local setup, not from a
leaderboard submission, so do not compare them directly with the public
leaderboard.

### Fitting it in 8 GiB: a fragmentation story

The first run completed. The second — same model, same hyperparameters, only
1,200 extra rows — died of CUDA OOM at optimizer step 193, five separate times,
three of them at *exactly* that step. Getting it to finish took six attempts and
cost more wall-clock than the training itself, so the diagnosis is worth writing
down.

Three hypotheses were tested and killed by evidence:

1. **Random fragmentation.** Killed by the crash landing on the same step
   repeatedly — the shuffle seed is fixed, so the data order is identical run to
   run. Something deterministic was happening.
2. **Cross-pairing manufactured giant sequences.** Plausible: refusal examples
   splice one example's tool schemas onto another's user turn. Killed by
   measuring token lengths — the refusals came out *shorter* (p99 1,250 vs 1,443
   for calls), the mixed set's distribution matched run 1's almost exactly, and
   the rows around the crash topped out at 1,617 tokens.
3. **My own "fix" was the problem.** `empty_cache()` every 25 steps made reserved
   memory sawtooth in lockstep with the callback, so it looked causal. Killed by
   removing it: reserved then climbed *monotonically* instead — 14.0 → 16.9 →
   19.8 GiB — and the run still died at 193.

The answer was sitting in the instrumentation the whole time:

```
peak allocated   5.34 GiB        <- live tensors
reserved        19.79 GiB        <- pool held from the driver, on an 8 GiB card
```

A 4x gap between live and reserved is not "out of memory", it is *"no two
allocations are the same shape"*. With `batch_size=1` and padding to each batch's
own maximum, every step allocates a logits tensor of `seq_len × 151,936` — and
`seq_len` differs almost every step. The pool fills with thousands of mutually
unusable free blocks, and on WDDM the driver keeps satisfying new requests out of
shared host memory until it won't.

The fix is one argument, `--pad-multiple 256`, which collapses thousands of
distinct shapes into eight reusable ones. Reserved memory went flat:

```
step  25-175   reserved 12.20 GiB   (unchanged for 150 steps)
step 200-225   reserved 15.68 GiB   (one step up at a larger bucket, then flat)
```

Padding is mathematically inert: pad positions carry `attention_mask=0` and
label `-100`, so they enter neither the loss nor any gradient. That mattered for
more than tidiness — the obvious OOM fixes (shrink `max_len`, shrink the batch,
drop LoRA rank) would each have changed the data or the optimization, and with
them the only variable this ablation is allowed to have.

**The instrumentation is the lesson.** An earlier attempt set
`expandable_segments:True` and looked *healthy* — no crash, loss coming down
normally. The VRAM log showed 19 GiB reserved on an 8 GiB card at 38 s/step
against a normal 5 s/step: it had silently started paging through host memory,
turning a 1-hour job into a 6.6-hour one. Nothing in the loss curve would have
said so. A 7x slowdown wearing the face of a healthy run is worse than a crash,
because a crash at least tells you.

## Trusting the measuring instrument

The evaluation harness is self-tested before any real score is believed
(`04_evaluate.py --self-test`). It feeds the ground truth back in, which must
score ≥90%, and feeds prose in, which must score 0% on call categories and 100%
on irrelevance.

That test earned its keep immediately. BFCL's `ast_checker` takes a `model_name`
and uses it to look up one flag in its model registry; an unregistered name
raises `KeyError` *inside* the checker. Caught by a broad `except`, every entry
scores as wrong — baseline 0%, tuned 0%, delta 0% — and nothing crashes. The
conclusion on offer would have been "a 1.5B model is too small for this to help",
when in fact the checker never ran.

This is the failure mode worth designing against: not a crash, but a plausible
number. Hence two rules the harness follows — checker-internal exceptions are
counted and surfaced separately from wrong answers, and the self-test asserts the
upper bound as well as the lower one. A harness that scores correct answers as
wrong looks exactly like a model that underperformed.

## Pipeline

Two interpreters, handing off through JSON — `bfcl_eval`'s dependency pins must
never be allowed to renegotiate the torch build verified on this GPU.

```
.venv        00_env_check.py      prove the GPU runs real kernels, in every dtype
.venv        00b_quant_check.py   prove bitsandbytes NF4 + LoRA backward works here
.venv        01_prepare_data.py   ToolACE -> single-turn SFT set, with drop reasons
.venv        01b_build_refusals.py cross-paired refusals -> train_mixed.jsonl
.venv-bfcl   02a_build_prompts.py BFCL's own prompt constructor -> prompts/*.json
.venv        02b_generate.py      local generation -> BFCL-format result files
.venv        03_train_qlora.py    QLoRA SFT, assistant-only loss, cost recorded
.venv-bfcl   04_evaluate.py       BFCL's own AST checker -> per-item 0/1
.venv-bfcl   05_report.py         Wilson / paired bootstrap / McNemar -> tables
.venv-bfcl   06_multiseed_report.py  per-seed + pooled replication of ablation B
.venv-bfcl   07_shortcut_probe.py    did refusal training learn a lexical rule?
```

## Results

Three models, identical prompts, greedy decoding, the same 1,240 BFCL tasks
scored by BFCL's own checker:

- **base** — `Qwen2.5-1.5B-Instruct` in NF4, no adapter
- **base_bf16** — the same weights unquantised, to price the quantisation itself
- **tuned** — LoRA on 4,916 ToolACE examples, every target a tool call
- **tuned_refusal** — those same examples **plus** 1,200 refusals (6,116 rows)
- **tuned_fixed** — 1,200 calls **replaced** by those refusals (4,917 rows)

The third model exists to separate "refusals help" from "more data helps", which
the second one alone cannot do.

### Run 1 — ToolACE as shipped

| category | n | base | tuned | delta (95% CI) | gained/lost | McNemar p |
|---|---|---|---|---|---|---|
| multiple | 200 | 81.00% | 89.50% | **+8.50%** [+4.00, +13.50] | +21/−4 | 0.00091 |
| parallel | 200 | 64.00% | 72.00% | **+8.00%** [+2.00, +14.50] | +29/−13 | 0.020 |
| simple_python | 400 | 83.50% | 85.50% | +2.00% [−1.25, +5.25] | +28/−20 | 0.31 |
| parallel_multiple | 200 | 56.00% | 58.00% | +2.00% [−5.50, +9.00] | +30/−26 | 0.69 |
| irrelevance | 240 | 73.75% | 18.33% | **−55.42%** [−61.67, −49.17] | **+0/−133** | 1.8e−40 |
| **OVERALL** | **1240** | **73.63%** | **66.53%** | **−7.10%** [−9.84, −4.27] | +108/−196 | 5.1e−07 |

The fine-tune worked on what it was trained for — `multiple` +8.5%, `parallel`
+8.0%, both significant — and made the model worse overall. `simple_python` and
`parallel_multiple` moved +2.0% each with intervals crossing zero; those are not
improvements, and reporting them as such would be reading noise.

Then `irrelevance` collapses by 55 points, and the *shape* of the collapse is
the finding: **+0/−133**. Not one task flipped from wrong to right. 133 flipped
from right to wrong, one direction, no offsetting gain. A coin does not do that.

The mechanism is in `results/format_compliance.json`: the share of irrelevance
prompts where the model emitted a parseable call went **26.25% → 81.67%**. All
4,916 training targets are tool calls, because that is what a tool-calling corpus
is, so the model learned an unconditional prior — *functions are present,
therefore call one*. It did not get better at choosing tools. It got worse at
declining.

Scoring only the call categories would have produced a clean, publishable,
entirely misleading headline: *"+8.5% and +8.0% on tool calling."* Both numbers
are real. The aggregate is −7.10%. `irrelevance` was in the eval set from the
start for exactly this reason, not added after the regression was suspected.

### Run 2 — the same recipe plus 1,200 refusals

| category | n | base | tuned_refusal | delta (95% CI) | gained/lost | McNemar p |
|---|---|---|---|---|---|---|
| multiple | 200 | 81.00% | 93.00% | **+12.00%** [+7.00, +17.00] | +27/−3 | 8.4e−06 |
| parallel | 200 | 64.00% | 76.50% | **+12.50%** [+6.00, +19.00] | +36/−11 | 0.00035 |
| simple_python | 400 | 83.50% | 87.50% | **+4.00%** [+0.75, +7.25] | +32/−16 | 0.029 |
| parallel_multiple | 200 | 56.00% | 60.50% | +4.50% [−3.00, +12.00] | +34/−25 | 0.30 |
| irrelevance | 240 | 73.75% | 68.33% | **−5.42%** [−9.58, −1.25] | +6/−19 | 0.015 |
| **OVERALL** | **1240** | **73.63%** | **78.55%** | **+4.92%** [+2.66, +7.18] | +135/−74 | 2.9e−05 |

### Run 3 — 1,200 calls *replaced* by refusals, total held fixed

Run 2 confounds two changes: it adds refusals *and* 24% more data. Run 3 removes
the confound by swapping 1,200 call examples for the same 1,200 refusals, landing
at 4,917 rows against run 1's 4,916.

| category | n | base | tuned_fixed | delta (95% CI) | gained/lost | McNemar p |
|---|---|---|---|---|---|---|
| multiple | 200 | 81.00% | 88.50% | **+7.50%** [+2.50, +12.50] | +22/−7 | 0.0081 |
| parallel | 200 | 64.00% | 71.00% | **+7.00%** [+1.00, +13.00] | +26/−12 | 0.034 |
| simple_python | 400 | 83.50% | 83.50% | 0.00% [−3.75, +3.75] | +29/−29 | 1.0 |
| parallel_multiple | 200 | 56.00% | 55.00% | −1.00% [−9.00, +7.00] | +32/−34 | 0.90 |
| irrelevance | 240 | 73.75% | 68.75% | **−5.00%** [−9.17, −1.25] | +6/−18 | 0.023 |
| **OVERALL** | **1240** | **73.63%** | **74.84%** | +1.21% [−1.13, +3.55] | +115/−100 | 0.34 |

### Ablation A vs B — separating composition from volume

**A. Adding refusals** (`tuned` → `tuned_refusal`, +24% data — confounded):

| category | delta | gained/lost | p |
|---|---|---|---|
| irrelevance | **+50.00%** | **+120/−0** | 1.5e−36 |
| parallel | +4.50% | +11/−2 | 0.022 |
| multiple | +3.50% | +8/−1 | 0.039 |
| simple_python | +2.00% | +12/−4 | 0.077 |
| parallel_multiple | +2.50% | +17/−12 | 0.46 |
| **OVERALL** | **+12.02%** | +168/−19 | 5.4e−31 |

**B. Replacing calls with refusals** (`tuned` → `tuned_fixed`, rows held fixed —
composition is the only variable):

| category | delta | gained/lost | p |
|---|---|---|---|
| irrelevance | **+50.42%** | **+121/−0** | 7.5e−37 |
| multiple | −1.00% | +6/−8 | 0.79 |
| parallel | −1.00% | +8/−10 | 0.81 |
| simple_python | −2.00% | +6/−14 | 0.12 |
| parallel_multiple | −3.00% | +17/−23 | 0.43 |
| **OVERALL** | **+8.31%** | +158/−55 | 9.9e−13 |

**Ablation A's call-side gains were the extra data, not the refusals.** Under A
every call category moved up and it was tempting to conclude that refusal data
*helps* tool calling. B says otherwise: with volume held fixed, all four call
categories move slightly **down** (−1.0 to −3.0%), none significantly.

That downward pattern is itself a single-seed artifact — see
[the three-seed replication](#three-seed-replication-of-ablation-b), where half
of those deltas flip sign. The durable claim from B is the +50 recovery, not the
small negatives.

What survives both ablations untouched is `irrelevance`: **+50.00% at +120/−0**
and **+50.42% at +121/−0**, independently reproduced, mirroring run 1's +0/−133.

**C. The volume term, isolated** (`tuned_refusal` → `tuned_fixed`, identical
refusals, 1,200 fewer calls):

| category | delta | p |
|---|---|---|
| simple_python | **−4.00%** | 0.00086 |
| parallel | **−5.50%** | 0.013 |
| multiple | **−4.50%** | 0.022 |
| parallel_multiple | −5.50% | 0.117 |
| irrelevance | +0.42% | **1.0** |
| **OVERALL** | **−3.71%** | 9.1e−06 |

The two effects are almost perfectly orthogonal. Removing call data costs three
call categories significantly and moves irrelevance by 0.42 points at p=1.0.
Refusal data buys declining behaviour; call data buys calling behaviour; neither
substitutes for the other.

**Practical consequence: append refusals, do not substitute them.** Both fixes
recover the safety regression equally; only the additive one also keeps the
capability win. Replicated at three seeds, appending beats replacing on OVERALL
by **1.13 / 2.42 / 3.71 points — 3/3 in the same direction, 2/3 significant,
mean 2.42**. The direction holds everywhere; the magnitude is roughly half what
the first seed alone suggested.

`format_compliance.json` shows the mechanism reversing cleanly. Parseable calls
emitted on irrelevance prompts: base **26.25%**, run 1 **81.67%**, run 2
**31.67%**, run 3 **31.25%**. Run 1 taught the model to always call; runs 2 and 3
restored something close to the base model's willingness to decline, while
keeping the call-formatting skill that made the fine-tune worth doing.

### Three-seed replication of Ablation B

One seed cannot support a claim about variance, and "the tax is below
resolution" is exactly such a claim. Ablation B was rerun end to end at three
seeds, with the seed driving refusal pairing, which calls get dropped, *and*
training order — so each is a full independent replicate, not the same dataset
retrained. All six runs share one memory configuration.

| category | seed 09 | seed 10 | seed 11 | mean | sign |
|---|---|---|---|---|---|
| **irrelevance** | +43.33% | +47.50% | +55.00% | **+48.61%** | **3/3 +** |
| multiple | +1.00% | 0.00% | +1.50% | +0.83% | no negatives |
| parallel | −1.00% | +4.50% | −0.50% | +1.00% | mixed |
| parallel_multiple | −2.50% | +7.00% | +4.00% | +2.83% | mixed |
| simple_python | −1.75% | −0.50% | −0.25% | −0.83% | 3/3 −, all tiny |
| **OVERALL** | +7.42% | +10.89% | +11.37% | **+9.89%** | 3/3 + |

Per-seed McNemar on irrelevance: **+105/−1, +115/−1, +133/−1**. Pooled across
seeds it is +353/−3 at n=720.

**There is no detectable alignment tax.** Across the four call categories and
three seeds, the twelve deltas split **5 positive, 6 negative, and one exactly
zero**, mean +0.96%. The uniformly-negative pattern that one seed showed does
not survive replication: on other seeds five of those twelve land positive.
Swapping 1,200 calls for 1,200 refusals leaves call-category performance
unchanged within noise, and buys ~49 points of declining behaviour.

This is the second correction on this point. The first ablation suggested
refusals *improve* tool calling (wrong — that was data volume); the
volume-controlled single seed then suggested a small consistent tax (also wrong
— that was sampling noise). Only the replication separates them.

### The variance is the other finding

`irrelevance` is far less stable in one arm than the other:

| arm | seed 09 | seed 10 | seed 11 | range |
|---|---|---|---|---|
| calls only | 25.42% | 18.33% | 10.00% | **15.42 pts** |
| swapped | 68.75% | 65.83% | 65.00% | **3.75 pts** |

Refusal data does not only raise irrelevance by ~49 points — it cuts the
seed-to-seed spread by roughly four fifths. A model trained only on successful
calls sits right on the call/decline boundary and lands anywhere in a 15-point
window depending on nothing but the seed. Adding refusals moves it off that
boundary and holds it there.

The append arm is steadier still on OVERALL — 78.55 / 77.98 / 76.69 — against
64.19–67.42 for calls-only. Refusal data narrows the spread in both mixing
strategies.

Two consequences for how the headline numbers should be read:

- **Run 1's −55.42% is a point in a range.** The same regression measured at
  three seeds is −48.33 / −55.42 / −63.75 points. Its existence is not in doubt
  at any seed; its magnitude carries ~15 points of uncertainty.
- **The config check clears.** Rerunning seed 20260809 under
  `--pad-multiple 256 --eval-strategy no` shifted OVERALL by **+0.89%
  (p=0.20)** — no significant difference, supporting the claim that both
  settings are memory-side only. The one category that moved, irrelevance
  (+7.08%), moved by less than half that arm's own seed-to-seed range, so it is
  not separable from noise.

### What does the 4-bit quantisation itself cost?

Every comparison above runs NF4 on both sides, which *controls* quantisation
away. This one measures it: same weights, same prompts, same greedy decode,
same batch size — only the numeric format moves.

| category | bf16 | NF4 | delta (95% CI) | gained/lost | McNemar p |
|---|---|---|---|---|---|
| parallel_multiple | 64.00% | 56.00% | **−8.00%** [−13.00, −3.00] | +7/−23 | 0.0052 |
| parallel | 70.50% | 64.00% | **−6.50%** [−11.50, −1.50] | +8/−21 | 0.024 |
| multiple | 84.50% | 81.00% | −3.50% [−8.00, +1.00] | +7/−14 | 0.19 |
| simple_python | 86.25% | 83.50% | −2.75% [−5.75, +0.25] | +14/−25 | 0.11 |
| irrelevance | 68.75% | 73.75% | **+5.00%** [+1.67, +8.33] | +15/−3 | 0.0075 |
| **OVERALL** | **76.45%** | **73.63%** | **−2.82%** [−4.68, −0.97] | +51/−86 | 0.0035 |

**The cost is real but concentrated, and one category moves the other way.**
4-bit costs ~2.8 points overall, but the damage is not spread evenly: the two
categories that require composing *several* calls in one response lose 6.5 and
8.0 points, while the single-call categories lose an amount this benchmark
cannot separate from zero. Difficulty and quantisation sensitivity track each
other.

`irrelevance` gains 5 points from being quantised, which sounds like quantisation
improving safety. It is not. The parse rates show one mechanism with two signs:

| category | parse rate, bf16 | parse rate, NF4 |
|---|---|---|
| simple_python | 99.50% | 98.50% |
| multiple | 100.00% | 99.00% |
| parallel | 98.00% | 96.00% |
| parallel_multiple | 96.00% | 95.50% |
| irrelevance | 31.25% | 26.25% |

NF4 emits a parseable call *less often in every single category*. Where calling
is correct that is a loss; on `irrelevance`, where declining is correct, the same
weakened impulse scores as a gain. A quantised model is not more principled about
declining — it is just less able to produce a call, and one category happens to
reward that.

Note the magnitudes, though: `parallel_multiple` loses 8.0 accuracy points while
its parse rate drops only 0.5. So the dominant failure is **not** malformed
output — it is well-formed calls with the wrong function or wrong arguments.
Quantisation degrades selection and argument binding considerably more than it
degrades format compliance, and a format-compliance metric alone would have
missed almost all of it.

**Limits of this arm, stated plainly.**

- **Only the untuned model has a full-precision arm.** The tuned model's bf16 run
  did not finish: bf16 weights plus a LoRA adapter plus the KV cache for the
  longer prompt categories exceeds 8 GiB, and the run degraded from 1.37 it/s to
  0.52 it/s and then died — the same WDDM host-memory spill documented above.
  Lowering the batch size would have fit, and that is exactly what makes it
  unusable; see below.
- **Batch size changes greedy outputs, so it cannot be varied between arms.**
  Re-running 40 `irrelevance` prompts at batch 1 instead of batch 4 changed 4 of
  them — including a flip from `[math.sum(...)]` to *"None of the provided
  functions can solve a quadratic equation"*, i.e. from wrong to right. Padding
  differences inside a batch perturb the logits enough to cross the
  call/decline boundary. Every arm in this project therefore uses batch 4, and
  the tuned bf16 comparison is left undone rather than run at a batch size that
  would have made it incomparable.

That last point generalises past this project: on a benchmark with a decision
boundary, batch size is not a performance knob, it is an experimental variable.

### Did it learn judgement, or a lexical shortcut?

The refusal builder only accepts a cross-paired example when the offered tool
names share **zero** word tokens with the request. That filter prevents
serviceable pairs from being mislabelled as refusals — a false refusal label
would teach the model to decline things it can do — but it has a side effect:
every refusal the model saw had disjoint vocabulary. A model can fit that
signal perfectly without learning anything about applicability, by learning
*"tool words and request words don't overlap → don't call"*.

If it learned that rule, it should over-refuse on tasks whose vocabulary happens
not to overlap but where calling is still correct. BFCL's call categories
already contain 86 such tasks, and every model's outputs on them are already
scored, so this costs no new training. The diagnostic is **call emission rate**
— did the model produce a parseable call at all — not accuracy, since accuracy
conflates "declined when it should have called" with "called the wrong thing".

| arm | disjoint (n=86) | overlapping (n=914) | gap |
|---|---|---|---|
| calls-only (control, 4 models) | 100.00% | 99.89–100% | +0.05% |
| refusal-trained (4 models) | **100.00%** | 99.89–100% | +0.08% |

**Excess gap attributable to refusal training: +0.03%.** Not one of the four
refusal-trained models declined a single disjoint call task. Paired at matched
seed, the emission delta is +0/−0 at p=1. The lexical shortcut is not what they
learned.

The control arm is what makes this readable: it never saw a refusal, so its
bucket gap is a property of the tasks. Only the excess implicates training.

**What this does not show.** BFCL's "disjoint" means disjoint *tokens*; the
tool is often still obviously related in meaning (`get_weather` against "will I
need an umbrella"). The refusals in training were semantically unrelated
(traffic lookup against Japanese tax rates). So this rules out the crudest
shortcut and says nothing about the middle ground — a tool that roughly fits but
lacks a required parameter, or that serves a near-variant of what was asked.
BFCL has no such category, and the safety claim here is correspondingly narrower
than "the model understands applicability".

One mechanism note in passing: emission rate on call tasks sits at ~100% for
every refusal-trained model. Whatever else they learned, **they did not become
reluctant to call** — behavioural support for the statistical finding that no
capability cost is detectable.

### What none of them fixed

Runs 2 and 3 are still **−5.42%** and **−5.00%** below base on irrelevance
(p=0.015, p=0.023). The regression is ~90% gone, not gone. A fifth of the
training set being refusals buys back most of the declining behaviour but does
not fully restore it, and the honest headline is "largely mitigated", not
"solved".

### Why this matters beyond the benchmark

An agent that cannot decline is the failure mode action-level safety evaluation
exists to catch. In a live API environment, "always call something" is how a
write lands on the wrong resource, or a permission boundary gets crossed, on a
request that should have been refused. Run 1's capability gain and its safety
regression came from the same 54 minutes of training, and only one of them was
visible in the loss curve — which, as the cost table shows, ranked the three
models in exactly the wrong order.

Two things here generalise past this model and this benchmark. First, the
regression is a **property of the corpus, not of the method**: any SFT set built
from successful tool calls teaches "call something" as an unconditional prior,
because refusals are not what a tool-calling dataset collects. Second, the fix is
cheap, but *which* fix you pick matters — substituting refusals recovers the
safety behaviour and throws away the capability gain, while appending them keeps
both. Ablation A alone would have gotten the reason right by accident and the
mechanism wrong.

### Cost

| run | rows | composition | steps | wall-clock | peak VRAM | final loss |
|---|---|---|---|---|---|---|
| tuned | 4,916 | calls only | 615 | 54.3 min | 5.91 GiB | 0.0558 |
| tuned_refusal | 6,116 | 4,920 calls + 1,200 refusals raw, 4 over max_len dropped | 765 | 82.0 min | 5.94 GiB | 0.0751 |
| tuned_fixed | 4,917 | 3,720 calls + 1,200 refusals raw, 3 over max_len dropped | 615 | 46.5 min | 5.94 GiB | 0.0800 |

One RTX 5070 Laptop (8 GiB), LoRA r=16, one epoch each, ~70 MB adapter each.

Note the loss column against the results: final training loss rises monotonically
with refusal share (0% → 19.6% → 24.4%), while the *best* model on BFCL is the
middle one and the *worst* is the one with the lowest loss. Loss on this corpus
ranks the models backwards. It is measuring how closely the output matches the
corpus's dominant style, and the corpus's dominant style is the failure being
fixed.

### Honest limits

- **Ablation B is replicated at three seeds; runs 1 and 2 are single-seed.** The
  irrelevance results are reproduced across every configuration and seed tried
  (+0/−133 down, then +120/−0, +121/−0, +105/−1, +115/−1, +133/−1 back up). The
  ±1–4.5% call-side numbers from the single-seed runs are not reproduced and
  should be read as noise.
- **"No detectable tax" is not "no tax".** Twelve deltas splitting 5 positive,
  6 negative and one exactly zero, with mean +0.96%, rules out an effect large
  enough for this setup to see; at n=200 per category per seed for three of the
  four call categories (`simple_python` is n=400), a true cost of ~1% would
  still be invisible. Three seeds bound it, they do not eliminate it.
- **Run 3 removes the volume confound but not every difference.** It has 150
  fewer optimizer steps than run 2 and a higher refusal share (24.4% vs 19.6%),
  since holding rows fixed necessarily changes the ratio.
- **The quantisation arm covers the untuned model only**, and at one batch size.
  See that section for why the tuned bf16 run was abandoned rather than run at a
  smaller batch.
- **Refusal targets are templated** — twelve phrasings with the tool names
  substituted in, not human-written declines. BFCL's irrelevance check only asks
  whether a parseable call was emitted, so it cannot distinguish a model that
  reasons about applicability from one that has learned a refusal register.
  [The shortcut probe](#did-it-learn-judgement-or-a-lexical-shortcut) rules out
  the crudest version of this (declining on vocabulary mismatch) but not the
  middle ground, which no BFCL category covers.
- **The first seed flattered every effect.** Seed 20260809 produced the largest
  magnitude in each analysis it appears in: the append-vs-replace gap (3.71 vs a
  3-seed mean of 2.42), the apparent alignment tax (a uniform negative that half
  the seeds reverse), and the gain over base (+4.92 vs a mean of +4.11). Every
  headline number in the first version of this README came from it. Directions
  survived replication; magnitudes shrank.
- **Run 2 needed two memory-side settings run 1 did not** — `--pad-multiple 256`
  and no in-training eval — because run 1's configuration OOM'd five times on
  this card. Padding positions are masked out of the loss and gradients, and
  eval runs under `no_grad`, so neither changes the learned weights; but the
  configurations were not byte-identical and pretending otherwise would be
  dishonest.
- **Refusal pairs are heuristically labelled.** A pair is kept when the request's
  required functions are absent from the offered tools *and* the two share no
  name tokens. That is conservative but not a semantic guarantee; a genuinely
  serviceable pair mislabelled as a refusal would teach the model to decline
  work it could do.
- **One epoch, one hyperparameter setting.** `grad_norm` peaked around 0.93
  against `max_grad_norm=0.3`, so gradients were clipped hard throughout — a
  natural next ablation, along with LoRA rank and the refusal mixing ratio
  (fixed here at 19.6% to match the eval set's irrelevance share, not tuned).
- **Single-turn Python categories only.** Multi-turn, Java/JS, and the
  executable categories are untested here.
- **Loss predicted nothing.** Run 1's training loss fell to ~0.05 within 3% of
  an epoch while the model was busy losing 55 points of the ability to decline.
  Low loss on a tool-calling corpus mostly measures style agreement.

## Reproducing

```bash
.venv/Scripts/python       scripts/00_env_check.py
.venv/Scripts/python       scripts/00b_quant_check.py
.venv/Scripts/python       scripts/01_prepare_data.py
.venv-bfcl/Scripts/python  scripts/02a_build_prompts.py
.venv-bfcl/Scripts/python  scripts/04_evaluate.py --self-test
.venv/Scripts/python       scripts/02b_generate.py --model base
.venv/Scripts/python       scripts/03_train_qlora.py
.venv/Scripts/python       scripts/02b_generate.py --model tuned
.venv-bfcl/Scripts/python  scripts/04_evaluate.py --model base
.venv-bfcl/Scripts/python  scripts/04_evaluate.py --model tuned

# ablation A: append refusals (6,116 rows)
.venv/Scripts/python       scripts/01b_build_refusals.py
.venv/Scripts/python       scripts/03_train_qlora.py \
    --run-name qwen2.5-1.5b-toolace-refusal-qlora \
    --train-file train_mixed.jsonl --val-file val_mixed.jsonl \
    --eval-strategy no --pad-multiple 256
.venv/Scripts/python       scripts/02b_generate.py --model tuned_refusal
.venv-bfcl/Scripts/python  scripts/04_evaluate.py --model tuned_refusal

# ablation B: replace calls with refusals, rows held fixed (4,917 rows)
.venv/Scripts/python       scripts/01b_build_refusals.py --fixed-total
.venv/Scripts/python       scripts/03_train_qlora.py \
    --run-name qwen2.5-1.5b-toolace-fixed-qlora \
    --train-file train_fixed.jsonl --val-file val_fixed.jsonl \
    --eval-strategy no --pad-multiple 256
.venv/Scripts/python       scripts/02b_generate.py --model tuned_fixed
.venv-bfcl/Scripts/python  scripts/04_evaluate.py --model tuned_fixed

.venv-bfcl/Scripts/python  scripts/05_report.py

# three-seed replication: ablation B (~6 h), then the append arm (~2.8 h)
powershell -File scripts/run_multiseed.ps1
powershell -File scripts/run_append_seeds.ps1
.venv-bfcl/Scripts/python  scripts/06_multiseed_report.py

# shortcut probe -- pure analysis over outputs that already exist
.venv-bfcl/Scripts/python  scripts/07_shortcut_probe.py

# quantized vs full-precision (~12 min for the baseline arm)
powershell -File scripts/run_precision.ps1
```

`--eval-strategy no --pad-multiple 256` are there for the 8 GiB card, not for the
experiment; see [Fitting it in 8 GiB](#fitting-it-in-8-gib-a-fragmentation-story).
Neither changes what the model learns.

## How this was built

Developed in August 2026 with Claude Code (an AI coding agent) as a pair
programmer; the commits are dated 2026-08-11 to 2026-08-20, and 6 of 6 commits
carry a Co-Authored-By trailer. Training and the eval generations ran on a
single 8 GiB laptop GPU (RTX 5070). This section was added on 2026-10-22, also
with Claude Code.

## Licences

Code Apache-2.0. ToolACE is Apache-2.0. Qwen2.5-1.5B-Instruct is under its own
licence. BFCL data and evaluator belong to the Gorilla project.
