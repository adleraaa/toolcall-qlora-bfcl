# Quantized vs full-precision arms.
#
# WHY
# Every run in this project so far loaded NF4 on both sides. That was the right
# call for measuring the adapter -- it controls quantisation away -- but it means
# the project cannot say anything about what quantisation itself costs. These two
# runs measure it instead, holding weights, prompts, and greedy decode fixed and
# moving only the numeric format.
#
# Two arms, because they answer different questions:
#   base_bf16          -- what does 4-bit cost an untuned model?
#   tuned_refusal_bf16 -- and does a LoRA adapter TRAINED against NF4 base
#                         weights still behave when those weights are restored to
#                         bf16? (QLoRA adapters are routinely served this way;
#                         whether the adapter compensates for quantisation error
#                         it no longer faces is an empirical question.)

$ErrorActionPreference = "Stop"
Set-Location "D:\ToolCallFT\scripts"

$venv = "D:\ToolCallFT\.venv\Scripts\python.exe"
$bfcl = "D:\ToolCallFT\.venv-bfcl\Scripts\python.exe"
$runs = "D:\ToolCallFT\runs"

Write-Output "=== [1/4] generate base @ bf16  $(Get-Date -Format 'HH:mm:ss') ==="
& $venv "02b_generate.py" --model base_bf16 --precision bf16
if ($LASTEXITCODE -ne 0) { throw "base bf16 generation failed" }

Write-Output ""
Write-Output "=== [2/4] generate tuned_refusal @ bf16  $(Get-Date -Format 'HH:mm:ss') ==="
& $venv "02b_generate.py" --model tuned_refusal_bf16 --precision bf16 `
    --adapter "$runs\qwen2.5-1.5b-toolace-refusal-qlora\adapter"
if ($LASTEXITCODE -ne 0) { throw "tuned bf16 generation failed" }

Write-Output ""
Write-Output "=== [3/4] score both  $(Get-Date -Format 'HH:mm:ss') ==="
& $bfcl "04_evaluate.py" --model base_bf16
if ($LASTEXITCODE -ne 0) { throw "scoring base_bf16 failed" }
& $bfcl "04_evaluate.py" --model tuned_refusal_bf16
if ($LASTEXITCODE -ne 0) { throw "scoring tuned_refusal_bf16 failed" }

Write-Output ""
Write-Output "=== [4/4] full report  $(Get-Date -Format 'HH:mm:ss') ==="
& $bfcl "05_report.py"
if ($LASTEXITCODE -ne 0) { throw "report failed" }

Write-Output ""
Write-Output "=== precision arms complete at $(Get-Date -Format 'HH:mm:ss') ==="
