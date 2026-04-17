#!/bin/bash
# Post-eval pipeline: runs automatically after eval completes.
# Usage: bash scripts/post_eval.sh
set -e

echo "=== POST-EVAL PIPELINE ==="
echo "$(date)"

# 1. Snapshot with full tables, multi-hop breakdown, significance
echo ""
echo ">>> Step 1: Full results snapshot"
python scripts/snapshot_results.py --save

# 2. Extract qualitative examples
echo ""
echo ">>> Step 2: Qualitative examples"
python scripts/extract_examples.py

# 3. Populate paper tables
echo ""
echo ">>> Step 3: Populate paper tables & figures"
python scripts/populate_paper.py

# 4. Claude judge verification pass (~$2, ~15 min)
echo ""
echo ">>> Step 4: Claude judge verification"
echo "    (Run manually: python scripts/claude_judge_pass.py)"
echo "    Estimated cost: ~\$2"

echo ""
echo "=== DONE ==="
echo "Results in paper/results/"
echo "Figures in paper/figures/"
