#!/usr/bin/env bash
# Run the full pipeline end-to-end (bash, including Colab/Linux/Mac).
# Usage:  bash run_all.sh
set -euo pipefail

echo "==> 01 prepare data"
python scripts/01_prepare_data.py
echo "==> 02 collect activations"
python scripts/02_collect_activations.py
echo "==> 03b LLM verbalize (replaces 03)"
python scripts/03b_llm_verbalize.py
echo "==> 04 train reconstructor"
python scripts/04_train_reconstructor.py
echo "==> 05 evaluate"
python scripts/05_evaluate.py
echo "==> 06 shuffled-explanations control"
python scripts/06_control_shuffled.py

echo
echo "Done. See results/eval_results.json and results/*.png"
