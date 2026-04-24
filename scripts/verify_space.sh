#!/usr/bin/env bash
# Pre-deploy verification for the HF Space. Catches the two classes of
# failure that shipped uncaught to production:
#   (1) app.py syntax errors under the Space's Python version
#   (2) metadata drift between README frontmatter and requirements.txt
# Plus sanity: valid cached demo data, clean venv install, gradio launch.
set -e
cd "$(dirname "$0")/.."

STAMP() { date +"%H:%M:%S"; }
log() { echo "[$(STAMP)] $*"; }

fail() {
    log "VERIFY FAILED: $*"
    exit 1
}

log "verify-space started"

# ---- 1. Byte-compile on local Python ----
log "[1/5] byte-compile app.py"
python3 -m py_compile app.py || fail "app.py has a SyntaxError on $(python3 --version)"

# ---- 2. Metadata parity check ----
log "[2/5] metadata parity (README + requirements)"
python3 scripts/check_space_metadata.py || fail "metadata parity check"

# ---- 3. Cached demo JSON validation ----
log "[3/5] cached demo JSON validation"
python3 - <<'PY' || exit 1
import json, sys
from pathlib import Path
p = Path("data/demo_cache.jsonl")
if not p.exists():
    print("FAIL: data/demo_cache.jsonl missing")
    sys.exit(1)
bad = []
n = 0
with open(p) as f:
    for ln in f:
        if not ln.strip():
            continue
        n += 1
        try:
            r = json.loads(ln)
        except json.JSONDecodeError as e:
            print(f"FAIL: invalid JSON line {n}: {e}")
            sys.exit(1)
        if len(r.get("answer","")) < 50:
            bad.append(r.get("question","?")[:60])
if bad:
    print(f"FAIL: {len(bad)} cached answers < 50 chars:")
    for q in bad: print(f"  - {q}")
    sys.exit(1)
if n < 5:
    print(f"FAIL: only {n} cached entries; expected >=5")
    sys.exit(1)
print(f"OK: {n} cached entries, all answers >= 50 chars")
PY

# ---- 4. Clean-venv install of deploy requirements ----
log "[4/5] clean-venv install of deploy requirements"
REQS=$(python3 - <<'PY'
import re, pathlib
src = pathlib.Path("scripts/deploy_hf_space.py").read_text()
m = re.search(r'requirements = """(.*?)"""', src, re.S)
print(m.group(1) if m else "")
PY
)
if [ -z "$REQS" ]; then
    fail "could not extract requirements from deploy_hf_space.py"
fi
VENV=$(mktemp -d)/venv
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "$REQS" > /tmp/verify_space_reqs.txt
pip install --quiet --disable-pip-version-check -r /tmp/verify_space_reqs.txt >/tmp/verify_space_pip.log 2>&1 || {
    deactivate || true
    tail -20 /tmp/verify_space_pip.log
    fail "pip install failed"
}
python -c "import gradio; print(f'imports OK (gradio {gradio.__version__})')" || {
    deactivate || true
    fail "core imports failed in clean venv"
}
deactivate || true
log "  clean venv install + core imports OK"

# ---- 5. Gradio smoke test in the clean venv ----
log "[5/5] gradio headless smoke test"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# Also install the project deps needed by app.py (sentence-transformers, networkx, ...)
pip install --quiet -e . >/tmp/verify_space_editable.log 2>&1 || {
    deactivate || true
    tail -20 /tmp/verify_space_editable.log
    fail "editable install failed"
}
unset ANTHROPIC_API_KEY
# smoke_gradio.py has its own timeout logic; don't require GNU coreutils.
python scripts/smoke_gradio.py --timeout 60 || {
    deactivate || true
    fail "gradio smoke test failed"
}
deactivate || true

log "VERIFY OK"
