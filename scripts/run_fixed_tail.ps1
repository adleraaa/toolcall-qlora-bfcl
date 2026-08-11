# Post-training steps for the total-fixed composition ablation (run 3).
# Same two-interpreter handoff as run_ablation_tail.ps1; stops at first failure.

$ErrorActionPreference = "Stop"
Set-Location "D:\ToolCallFT\scripts"

$venv = "D:\ToolCallFT\.venv\Scripts\python.exe"
$bfcl = "D:\ToolCallFT\.venv-bfcl\Scripts\python.exe"

Write-Output "=== [1/3] generating with the total-fixed adapter ==="
& $venv "02b_generate.py" --model tuned_fixed
if ($LASTEXITCODE -ne 0) { throw "generation failed with exit code $LASTEXITCODE" }

Write-Output ""
Write-Output "=== [2/3] scoring ==="
& $bfcl "04_evaluate.py" --model tuned_fixed
if ($LASTEXITCODE -ne 0) { throw "scoring failed with exit code $LASTEXITCODE" }

Write-Output ""
Write-Output "=== [3/3] full report across all four models ==="
& $bfcl "05_report.py"
if ($LASTEXITCODE -ne 0) { throw "report failed with exit code $LASTEXITCODE" }

Write-Output ""
Write-Output "=== fixed-total tail complete ==="
