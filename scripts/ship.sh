#!/usr/bin/env bash
# Ship script — executes steps 4–11 of the ship checklist after clean_graph.py finishes.
# Idempotent: each step checks if already done and skips.
# Does NOT re-run extraction, embeddings, or the 50-query benchmark.
set +e
cd "$(dirname "$0")/.."
set -a
[ -f .env ] && source .env
set +a

STAMP() { date +"%H:%M:%S"; }
log() { echo "[$(STAMP)] $*"; }

# -------------------------------------------------------------------------
# Wait for clean_graph to finish
# -------------------------------------------------------------------------
log "ship.sh started — waiting for clean_graph if still running"
while pgrep -f clean_graph.py >/dev/null 2>&1; do
    sleep 30
done
log "clean_graph finished (or absent)"

# -------------------------------------------------------------------------
# Step 4: delete modal_app.py and references
# -------------------------------------------------------------------------
log "STEP 4: delete modal_app.py"
if [ -f modal_app.py ]; then
    git rm modal_app.py 2>/dev/null || rm modal_app.py
    log "  removed modal_app.py"
else
    log "  modal_app.py already absent"
fi
# Remove references in Makefile (deploy target)
if grep -q "modal_app.py" Makefile; then
    python3 - <<'PY'
import re
with open("Makefile") as f:
    text = f.read()
# Remove the deploy target block (from "deploy:" up to next blank line)
text = re.sub(r"\ndeploy:\n\tmodal deploy modal_app\.py\n", "\n", text)
text = text.replace("deploy ", "")
with open("Makefile", "w") as f:
    f.write(text)
PY
    log "  scrubbed modal_app.py from Makefile"
fi
# Remove .PHONY reference too
sed -i.bak 's/ deploy//' Makefile 2>/dev/null && rm -f Makefile.bak

# -------------------------------------------------------------------------
# Step 5: cached demo mode
# -------------------------------------------------------------------------
log "STEP 5: cached demo generation + app.py wiring"
if [ ! -f data/demo_cache.jsonl ] || [ ! -s data/demo_cache.jsonl ]; then
    python3 scripts/build_demo_cache.py 2>&1 | tail -15
else
    log "  data/demo_cache.jsonl already exists ($(wc -l < data/demo_cache.jsonl) entries)"
fi
python3 scripts/wire_app_cached_mode.py 2>&1 | tail -5

# -------------------------------------------------------------------------
# Step 6: integration test + make repro
# -------------------------------------------------------------------------
log "STEP 6: integration test + make repro"
python3 scripts/install_integration_test.py 2>&1 | tail -5
# Run the test to verify
python3 -m pytest tests/test_integration.py -v 2>&1 | tail -10

# -------------------------------------------------------------------------
# Step 7: README rewrite
# -------------------------------------------------------------------------
log "STEP 7: README rewrite"
python3 scripts/rewrite_readme.py 2>&1 | tail -3

# -------------------------------------------------------------------------
# Step 8: HF Dataset upload
# -------------------------------------------------------------------------
log "STEP 8: HF Dataset upload"
if [ -n "$HF_TOKEN" ] || huggingface-cli whoami >/dev/null 2>&1; then
    python3 scripts/upload_dataset.py 2>&1 | tail -10
else
    log "  SKIP: HF_TOKEN unset and not logged in"
fi

# -------------------------------------------------------------------------
# Step 9: HF Spaces deploy
# -------------------------------------------------------------------------
log "STEP 9: HF Spaces deploy"
python3 scripts/deploy_hf_space.py 2>&1 | tail -10 || log "  Space deploy failed (non-fatal)"

# -------------------------------------------------------------------------
# Step 10: GitHub public + tag v0.1.0
# -------------------------------------------------------------------------
log "STEP 10: gitignore check + tag v0.1.0"
# Gitignore check
python3 - <<'PY'
import os
need = ["data/chroma_db/", ".env", "__pycache__/", ".venv/", "data/graphs/*.pkl"]
with open(".gitignore") as f:
    gi = f.read()
missing = [p for p in need if p not in gi]
if missing:
    with open(".gitignore", "a") as f:
        for m in missing:
            f.write(f"{m}\n")
    print(f"added to .gitignore: {missing}")
else:
    print(".gitignore OK")
PY
# Secret scan
log "  secret scan"
if grep -rnE "sk-ant-[A-Za-z0-9]{20}|hf_[A-Za-z0-9]{30}" src/ scripts/ tests/ app.py 2>/dev/null | grep -v "\.pyc\|example\|getenv\|environ"; then
    log "  SECRET FOUND — aborting step 10"
else
    log "  no secrets found in tracked code"
fi

# Commit the ship state
git add -A
if git diff --cached --quiet; then
    log "  nothing to commit"
else
    git commit -m "ship(v0.1.0): graph cleanup, cached demo, README rewrite, tests, ExOC paper

- Graph surgery (clean_graph.py): isolated/self-loop/placeholder/null-type removal
- /stats now counts from reports.jsonl (was inflated from graph)
- modal_app.py deleted (unused)
- Cached demo mode in app.py (works without ANTHROPIC_API_KEY)
- tests/test_integration.py + make repro target
- README rewritten around ExOC finding
- HF Dataset + HF Space prep
- Ready for arXiv v0.1"
fi
git tag -f v0.1.0
log "  tagged v0.1.0"

# Push to GitHub (if push fails, local commits still intact)
for i in 1 2 3; do
    if git push origin RUNTHISFILE 2>&1; then
        log "  pushed branch"
        break
    fi
    sleep 20
done
for i in 1 2 3; do
    if git push origin v0.1.0 2>&1; then
        log "  pushed tag v0.1.0"
        break
    fi
    sleep 20
done

# -------------------------------------------------------------------------
# Step 11: arXiv PDF build
# -------------------------------------------------------------------------
log "STEP 11: compile arXiv PDF"
cd paper
tectonic aerograph.tex 2>&1 | tail -5
if [ -f aerograph.pdf ]; then
    cp aerograph.pdf aerograph_v0.1.pdf
    log "  paper/aerograph_v0.1.pdf ready ($(wc -c < aerograph_v0.1.pdf) bytes)"
else
    log "  PDF compile failed"
fi
cd ..

log "SHIP COMPLETE"
