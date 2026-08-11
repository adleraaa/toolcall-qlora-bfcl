# Replicate the APPEND arm at the two seeds the replace arm already has.
#
# WHY
# The practical recommendation this project ends on -- append refusals rather
# than substitute them -- rests on comparing two configurations. After
# run_multiseed.ps1 the replace arm has three seeds and the append arm has one,
# so the recommendation is a claim about a difference where one side is a single
# draw. Two more append runs put both arms on the same footing.
#
# Same memory configuration as every other run in the replication
# (--pad-multiple 256, --eval-strategy no), and the seed again drives refusal
# pairing as well as training order.

$ErrorActionPreference = "Stop"
Set-Location "D:\ToolCallFT\scripts"

$venv = "D:\ToolCallFT\.venv\Scripts\python.exe"
$bfcl = "D:\ToolCallFT\.venv-bfcl\Scripts\python.exe"
$runs = "D:\ToolCallFT\runs"

foreach ($seed in @(20260810, 20260811)) {
    $tag = "tuned_refusal_s$seed"
    $run = "qwen2.5-1.5b-append-s$seed"

    Write-Output ""
    Write-Output "=============================================================="
    Write-Output "  BUILD append set, seed $seed   $(Get-Date -Format 'HH:mm:ss')"
    Write-Output "=============================================================="
    & $venv "01b_build_refusals.py" --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "build failed ($seed)" }

    Write-Output ""
    Write-Output "  TRAIN $tag  $(Get-Date -Format 'HH:mm:ss')"
    & $venv "03_train_qlora.py" --run-name $run `
        --train-file "train_mixed_s$seed.jsonl" --val-file "val_mixed_s$seed.jsonl" `
        --eval-strategy no --pad-multiple 256 --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "train failed ($seed)" }

    Write-Output ""
    Write-Output "  GENERATE $tag  $(Get-Date -Format 'HH:mm:ss')"
    & $venv "02b_generate.py" --model $tag --adapter "$runs\$run\adapter"
    if ($LASTEXITCODE -ne 0) { throw "generate failed ($seed)" }

    Write-Output ""
    Write-Output "  SCORE $tag  $(Get-Date -Format 'HH:mm:ss')"
    & $bfcl "04_evaluate.py" --model $tag
    if ($LASTEXITCODE -ne 0) { throw "score failed ($seed)" }
}

Write-Output ""
Write-Output "=== append replication complete at $(Get-Date -Format 'HH:mm:ss') ==="
