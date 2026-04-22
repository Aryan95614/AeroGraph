#!/usr/bin/env bash
# End-to-end held-out pipeline after the initial partial extraction.
# Assumes data_heldout/processed/extractions.jsonl contains only non-empty
# records (run the cleanup Python block first). Will finish the missing
# extractions, then run build/normalize/embed/community/eval sequentially.
set -e
cd "$(dirname "$0")/.."

set -a
source .env
set +a

export AEROGRAPH_DATA_DIR=data_heldout

STAMP() { date +"%H:%M:%S"; }
echo "[$(STAMP)] === EXTRACT (resume missing) ==="
python3 -m aerograph extract 2>&1 | tail -5

echo "[$(STAMP)] === Cleaning empties (round 2) ==="
python3 <<'PY'
import json
kept = []
with open('data_heldout/processed/extractions.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('entities') or d.get('relations'):
            kept.append(d)
with open('data_heldout/processed/extractions.jsonl', 'w') as f:
    for d in kept:
        f.write(json.dumps(d) + '\n')
remaining = 2000 - len(kept)
print(f'Non-empty: {len(kept)}  To retry: {remaining}')
PY

echo "[$(STAMP)] === EXTRACT (retry round 2) ==="
python3 -m aerograph extract 2>&1 | tail -5

echo "[$(STAMP)] === BUILD GRAPH ==="
python3 -m aerograph build 2>&1 | tail -5

echo "[$(STAMP)] === NORMALIZE ==="
python3 -m aerograph normalize 2>&1 | tail -8

echo "[$(STAMP)] === EMBED ==="
python3 -m aerograph embed 2>&1 | tail -3

echo "[$(STAMP)] === COMMUNITY ==="
python3 -m aerograph community 2>&1 | tail -5

echo "[$(STAMP)] === HELDOUT EVAL ==="
python3 scripts/run_heldout_eval.py 2>&1 | tail -12

echo "[$(STAMP)] === DONE ==="
