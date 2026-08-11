# Multi-seed replication of Ablation B: calls-only vs composition-swapped.
#
# WHY ONLY ABLATION B
# The +50-point irrelevance recovery is already reproduced across two independent
# configurations (+120/-0 and +121/-0); more seeds there buy little. The claim
# that still has no interval is the *call-side* one: with rows held fixed, the
# four call categories moved -1 to -3% and none reached significance, so the
# honest reading was "a tax exists but sits below this benchmark's resolution".
# That sentence is a statement about variance, and one seed cannot support it.
#
# DESIGN
# 3 seeds x {calls-only, composition-swapped}. The seed drives refusal pairing,
# which calls get dropped, and training order -- each seed is a full independent
# replicate, not the same dataset retrained.
#
# Every run here uses the SAME memory configuration (--pad-multiple 256,
# --eval-strategy no). The original `tuned` used neither, so it is rerun at seed
# 20260809 rather than reused: comparing across a config boundary is exactly the
# sloppiness this experiment exists to remove. That rerun doubles as a check on
# the claim that those two settings are inert -- same seed, same data, only the
# memory settings differ, so the scores should land on top of each other.
#
# `tuned_fixed` at seed 20260809 already ran under this exact configuration and
# is reused as-is.

$ErrorActionPreference = "Stop"
Set-Location "D:\ToolCallFT\scripts"

$venv = "D:\ToolCallFT\.venv\Scripts\python.exe"
$bfcl = "D:\ToolCallFT\.venv-bfcl\Scripts\python.exe"
$runs = "D:\ToolCallFT\runs"

$allSeeds = @(20260809, 20260810, 20260811)
$newSeeds = @(20260810, 20260811)   # 20260809's fixed run already exists

function Invoke-Step {
    param([string]$Label, [scriptblock]$Action)
    Write-Output ""
    Write-Output "=================================================================="
    Write-Output "  $Label"
    Write-Output "  $(Get-Date -Format 'HH:mm:ss')"
    Write-Output "=================================================================="
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

# ---- arm A: calls only, all three seeds -----------------------------------
foreach ($seed in $allSeeds) {
    $tag = "tuned_s$seed"
    $run = "qwen2.5-1.5b-callsonly-s$seed"

    Invoke-Step "TRAIN $tag (calls only, 4,920 rows)" {
        & $venv "03_train_qlora.py" --run-name $run `
            --train-file "train.jsonl" --val-file "val.jsonl" `
            --eval-strategy no --pad-multiple 256 --seed $seed
    }
    Invoke-Step "GENERATE $tag" {
        & $venv "02b_generate.py" --model $tag --adapter "$runs\$run\adapter"
    }
    Invoke-Step "SCORE $tag" {
        & $bfcl "04_evaluate.py" --model $tag
    }
}

# ---- arm B: composition swapped, two new seeds ----------------------------
foreach ($seed in $newSeeds) {
    $tag = "tuned_fixed_s$seed"
    $run = "qwen2.5-1.5b-fixed-s$seed"

    Invoke-Step "BUILD refusal set (seed $seed, total held fixed)" {
        & $venv "01b_build_refusals.py" --fixed-total --seed $seed
    }
    Invoke-Step "TRAIN $tag (3,720 calls + 1,200 refusals)" {
        & $venv "03_train_qlora.py" --run-name $run `
            --train-file "train_fixed_s$seed.jsonl" --val-file "val_fixed_s$seed.jsonl" `
            --eval-strategy no --pad-multiple 256 --seed $seed
    }
    Invoke-Step "GENERATE $tag" {
        & $venv "02b_generate.py" --model $tag --adapter "$runs\$run\adapter"
    }
    Invoke-Step "SCORE $tag" {
        & $bfcl "04_evaluate.py" --model $tag
    }
}

Write-Output ""
Write-Output "=== multiseed complete at $(Get-Date -Format 'HH:mm:ss') ==="
Write-Output "arms: tuned_s{20260809,20260810,20260811}"
Write-Output "      tuned_fixed{,_s20260810,_s20260811}"
