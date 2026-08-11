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
at **+4.92%** over base. A volume-controlled ablation shows the safety recovery
is entirely the refusal data and the capability gain is entirely the data volume
— the two are orthogonal, so refusals should be **appended, not substituted**.

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
| Framework | PyTorch + transformers + peft + bitsandbytes + trl |
| Benchmark | BFCL, scored with the official `bfcl evaluate` |
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
through transformers and the resulting response files are scored by the official
evaluator — the scores stay comparable to the public leaderboard.

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
```

## Results

Three models, identical prompts, greedy decoding, the same 1,240 BFCL tasks
scored by BFCL's own checker:

- **base** — `Qwen2.5-1.5B-Instruct` in NF4, no adapter
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
| irrelevance | +0.42% | **1.0** |
| **OVERALL** | **−3.71%** | 9.1e−06 |

The two effects are almost perfectly orthogonal. Removing call data costs three
call categories significantly and moves irrelevance by 0.42 points at p=1.0.
Refusal data buys declining behaviour; call data buys calling behaviour; neither
substitutes for the other.

**Practical consequence: append refusals, do not substitute them.** Both fixes
recover the safety regression equally. Only the additive one also keeps the
capability win — +4.92% over base and significant, against +1.21% at p=0.34 for
the replacement, which is statistically indistinguishable from not fine-tuning
at all.

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
three seeds, the twelve deltas split **6 positive / 6 negative**, mean +0.96%.
The uniformly-negative pattern that one seed showed does not survive
replication — half of it changes sign. Swapping 1,200 calls for 1,200 refusals
leaves call-category performance unchanged within noise, and buys ~49 points of
declining behaviour.

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
| tuned_refusal | 6,116 | 4,920 calls + 1,200 refusals | 765 | 82.0 min | 5.94 GiB | 0.0751 |
| tuned_fixed | 4,917 | 3,720 calls + 1,200 refusals | 615 | 46.5 min | 5.94 GiB | 0.0800 |

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
- **"No detectable tax" is not "no tax".** Twelve deltas splitting 6/6 with mean
  +0.96% rules out an effect large enough for this setup to see; at n=200 per
  category per seed, a true cost of ~1% would still be invisible. Three seeds
  bound it, they do not eliminate it.
- **Run 3 removes the volume confound but not every difference.** It has 150
  fewer optimizer steps than run 2 and a higher refusal share (24.4% vs 19.6%),
  since holding rows fixed necessarily changes the ratio.
- **Refusal targets are templated** — twelve phrasings with the tool names
  substituted in, not human-written declines. BFCL's irrelevance check only asks
  whether a parseable call was emitted, so it cannot distinguish a model that
  reasons about applicability from one that has learned a refusal register. That
  distinction matters for the safety claim and is not measured here.
- **Run 2 needed two memory-side settings run 1 did not** — `--pad-multiple 256`
  and no in-training eval — because run 1's configuration OOM'd five times on
  this card. Padding positions are masked out of the loss and gradients, and
  eval runs under `no_grad`, so neither changes the learned weights; but the
  configurations were not byte-identical and pretending otherwise would be
  dishonest.
- **Refusal targets are templated.** Twelve phrasings with the tool names
  interpolated, not natural human refusals. The model may have learned the
  template as much as the judgement, and BFCL's irrelevance check only asks
  whether a parseable call was emitted — it cannot tell a well-reasoned decline
  from a canned one.
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

# three-seed replication of ablation B (~6 h on one 8 GiB GPU)
powershell -File scripts/run_multiseed.ps1
.venv-bfcl/Scripts/python  scripts/06_multiseed_report.py
```

`--eval-strategy no --pad-multiple 256` are there for the 8 GiB card, not for the
experiment; see [Fitting it in 8 GiB](#fitting-it-in-8-gib-a-fragmentation-story).
Neither changes what the model learns.

## Licences

Code Apache-2.0. ToolACE is Apache-2.0. Qwen2.5-1.5B-Instruct is under its own
licence. BFCL data and evaluator belong to the Gorilla project.
