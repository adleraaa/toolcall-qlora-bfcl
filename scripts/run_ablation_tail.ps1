# Chain the post-training steps for the refusal-mixed ablation.
#
# Generation needs .venv (torch); scoring and reporting need .venv-bfcl. They are
# separate interpreters on purpose -- letting pip resolve bfcl-eval's pins next
# to the verified Blackwell torch build risks downgrading it -- so the handoff is
# through files on disk, and this script just sequences the two.
#
# Stops at the first failure rather than reporting a partial comparison as if it
# were complete.

$ErrorActionPreference = "Stop"
Set-Location "D:\ToolCallFT\scripts"

$venv = "D:\ToolCallFT\.venv\Scripts\python.exe"
$bfcl = "D:\ToolCallFT\.venv-bfcl\Scripts\python.exe"

Write-Output "=== [1/3] generating BFCL responses with the refusal-mixed adapter ==="
& $venv "02b_generate.py" --model tuned_refusal
if ($LASTEXITCODE -ne 0) { throw "generation failed with exit code $LASTEXITCODE" }

Write-Output ""
Write-Output "=== [2/3] scoring with BFCL's own checker ==="
& $bfcl "04_evaluate.py" --model tuned_refusal
if ($LASTEXITCODE -ne 0) { throw "scoring failed with exit code $LASTEXITCODE" }

Write-Output ""
Write-Output "=== [3/3] paired statistics across all three models ==="
& $bfcl "05_report.py"
if ($LASTEXITCODE -ne 0) { throw "report failed with exit code $LASTEXITCODE" }

Write-Output ""
Write-Output "=== ablation tail complete ==="
