# Run the full pipeline end-to-end (Windows PowerShell).
# Usage:  .\run_all.ps1
$ErrorActionPreference = "Stop"
Write-Host "==> 01 prepare data"
python scripts/01_prepare_data.py
Write-Host "==> 02 collect activations"
python scripts/02_collect_activations.py
Write-Host "==> 03b LLM verbalize (replaces 03)"
python scripts/03b_llm_verbalize.py
Write-Host "==> 04 train reconstructor"
python scripts/04_train_reconstructor.py
Write-Host "==> 05 evaluate"
python scripts/05_evaluate.py
Write-Host "==> 06 shuffled-explanations control"
python scripts/06_control_shuffled.py
Write-Host ""
Write-Host "Done. See results/eval_results.json and results/*.png"
