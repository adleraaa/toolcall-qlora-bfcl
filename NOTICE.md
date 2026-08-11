# Third-party components

This project's own code is Apache-2.0 (see `LICENSE`). It trains on and evaluates
against work owned by others, none of which is redistributed here:

| Component | Owner | Licence | How it is used |
|---|---|---|---|
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Team-ACE | Apache-2.0 | Training data, downloaded at runtime by `01_prepare_data.py`. The refusal set in `01b_build_refusals.py` is derived from it by recombining existing examples. |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Alibaba Cloud | Apache-2.0 | Base model, downloaded at runtime. |
| [BFCL / Gorilla](https://github.com/ShishirPatil/gorilla) | UC Berkeley | Apache-2.0 | Benchmark data, prompt construction, and the AST checker that produces every score reported. |

`data/` and `prompts/` are gitignored specifically so that no reshaped copy of
ToolACE or BFCL is redistributed through this repository — run the scripts and
they are rebuilt from the original sources.

Trained LoRA adapters are also not published here. They are ~70 MB each, eight
of them exist, and every one is reproducible from the commands in the README.
