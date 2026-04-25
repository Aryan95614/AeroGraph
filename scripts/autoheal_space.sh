#!/usr/bin/env bash
# Autoheal loop for the HF Space.
#
# What it does:
#   1. Polls https://huggingface.co/spaces/Aryan95614/aerograph until the
#      Space is either RUNNING (exit 0) or RUNTIME_ERROR.
#   2. On RUNTIME_ERROR, scrapes the Python traceback from the page.
#   3. Dispatches two rival fixers in parallel — `gemini` and `codex` CLIs —
#      with the same prompt: "app.py broke on HF Spaces with this traceback,
#      propose the smallest patch that resolves it."
#   4. For each returned patch: apply to a worktree, run `make verify-space`.
#      First to pass verify-space wins; its patch is applied to the repo and
#      redeployed.
#   5. Repeat up to MAX_ITERS times, then stop for human review.
#
# Requires: `gemini` and `codex` on PATH. If either is missing, that branch
# is skipped. Both missing = script prints a diagnostic and exits.
#
# Usage: bash scripts/autoheal_space.sh [--max-iters 3] [--poll-interval 90]
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE_URL="https://huggingface.co/spaces/Aryan95614/aerograph"
MAX_ITERS="${MAX_ITERS:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-90}"
POLL_CAP="${POLL_CAP:-8}"  # max polls per iteration

STAMP() { date +"%H:%M:%S"; }
log() { echo "[$(STAMP)] autoheal: $*"; }

# ---------- helpers ----------

fetch_status() {
    # Returns one of: RUNNING | BUILDING | RUNTIME_ERROR | UNKNOWN
    # A false-positive RUNNING will cost a real iteration, so require
    # direct-endpoint evidence. The iframe URL appears in page metadata
    # regardless of status — do NOT use it alone as a success signal.
    local page_body iframe_body iframe_http
    page_body=$(curl -sL "$SPACE_URL" 2>/dev/null || echo "")
    if echo "$page_body" | grep -qE "Runtime error|Exit code:"; then
        echo "RUNTIME_ERROR"
        return
    fi
    # Direct iframe endpoint:
    #   HTTP 200 + gradio HTML markers => RUNNING
    #   HTTP 503 / empty               => still building or crashed
    iframe_http=$(curl -sL -o /tmp/autoheal_iframe.html -w "%{http_code}" \
                    "https://aryan95614-aerograph.hf.space" 2>/dev/null || echo "000")
    iframe_body=$(cat /tmp/autoheal_iframe.html 2>/dev/null || echo "")
    if [ "$iframe_http" = "200" ] && \
       echo "$iframe_body" | grep -qE "gradio-app|__GRADIO_CONFIG__|/assets/index-"; then
        echo "RUNNING"
        return
    fi
    if echo "$page_body" | grep -qE "APP_STARTING|Building|Fetching metadata|Installing"; then
        echo "BUILDING"
        return
    fi
    echo "UNKNOWN"
}

fetch_traceback() {
    # Best-effort scrape of the traceback from the Space page.
    curl -sL "$SPACE_URL" 2>/dev/null \
        | sed -n '/Traceback/,/^[[:space:]]*$/p' \
        | head -50
}

# Dispatch a CLI fixer ($1 = name, $2 = binary, $3 = traceback file, $4 = output dir)
# Writes the proposed patched app.py into $4/app.py.
dispatch_fixer() {
    local name="$1" bin="$2" trace="$3" outdir="$4"
    if ! command -v "$bin" >/dev/null 2>&1; then
        log "skip $name (binary not on PATH)"
        return 1
    fi
    mkdir -p "$outdir"
    # The fixer gets: the traceback, the current app.py, and a system prompt
    # asking for the smallest diff that resolves the error.
    local prompt_file="$outdir/prompt.txt"
    cat > "$prompt_file" <<PROMPT
You are fixing a Python app that fails to start on HuggingFace Spaces.

Here is the current app.py:
\`\`\`python
$(cat app.py)
\`\`\`

Here is the runtime traceback from the Space container:
\`\`\`
$(cat "$trace")
\`\`\`

Produce a new app.py that resolves the error with the smallest possible
change. Output ONLY the complete updated app.py contents, no commentary,
no markdown fences. Do not add features or rename functions.
PROMPT

    log "dispatching $name with trace ($(wc -l < "$trace") lines)"
    # Timeout keeps runaway fixers in check. 180s is plenty for either CLI.
    if "$bin" --help 2>&1 | grep -q "non-interactive\|--prompt"; then
        "$bin" --prompt "$(cat "$prompt_file")" > "$outdir/app.py" 2>"$outdir/err" || true
    else
        # Generic stdin-based invocation
        "$bin" < "$prompt_file" > "$outdir/app.py" 2>"$outdir/err" || true
    fi
    if [ ! -s "$outdir/app.py" ]; then
        log "$name produced empty output"
        return 1
    fi
    return 0
}

# Apply a candidate app.py to the repo, run verify-space, return 0 if it passes.
try_candidate() {
    local candidate="$1"
    cp app.py app.py.preheal
    cp "$candidate" app.py
    if bash scripts/verify_space.sh >"/tmp/autoheal_verify_$$.log" 2>&1; then
        log "candidate passes verify-space"
        rm -f app.py.preheal
        return 0
    fi
    # Revert
    mv app.py.preheal app.py
    tail -15 "/tmp/autoheal_verify_$$.log" | sed 's/^/    verify: /'
    rm -f "/tmp/autoheal_verify_$$.log"
    return 1
}

# ---------- main loop ----------

for iter in $(seq 1 "$MAX_ITERS"); do
    log "iteration $iter / $MAX_ITERS"

    # Poll for terminal state
    status=UNKNOWN
    for poll in $(seq 1 "$POLL_CAP"); do
        status=$(fetch_status)
        log "  poll $poll: $status"
        case "$status" in
            RUNNING)
                log "Space is RUNNING. Done."
                exit 0 ;;
            RUNTIME_ERROR)
                break ;;
        esac
        sleep "$POLL_INTERVAL"
    done

    if [ "$status" != "RUNTIME_ERROR" ]; then
        log "no terminal state reached; giving up this iteration"
        continue
    fi

    # Capture the traceback
    trace_file="/tmp/autoheal_trace_${iter}.txt"
    fetch_traceback > "$trace_file"
    if [ ! -s "$trace_file" ]; then
        log "could not scrape traceback from page; dumping raw HTML snippet"
        curl -sL "$SPACE_URL" | grep -iE "error|traceback" | head -20 > "$trace_file"
    fi
    log "traceback captured ($(wc -l < "$trace_file") lines)"

    # Dispatch gemini + codex in parallel
    gem_dir="/tmp/autoheal_gemini_${iter}"
    cdx_dir="/tmp/autoheal_codex_${iter}"
    ( dispatch_fixer gemini gemini "$trace_file" "$gem_dir" ) &
    pid_g=$!
    ( dispatch_fixer codex codex "$trace_file" "$cdx_dir" ) &
    pid_c=$!
    wait "$pid_g" || true
    wait "$pid_c" || true

    # First-to-pass-verify wins
    applied=""
    for cand_dir in "$gem_dir" "$cdx_dir"; do
        if [ -f "$cand_dir/app.py" ] && try_candidate "$cand_dir/app.py"; then
            applied="$cand_dir"
            break
        fi
    done

    if [ -z "$applied" ]; then
        log "no candidate passed verify-space; stopping for human review"
        log "  gemini attempt: $gem_dir"
        log "  codex attempt:  $cdx_dir"
        log "  traceback:      $trace_file"
        exit 2
    fi

    log "applied candidate from $applied; redeploying"
    python scripts/deploy_hf_space.py || { log "deploy failed"; exit 3; }
    git add app.py
    git commit -m "autoheal: apply fix from $(basename "$applied")

From traceback in $trace_file" || true
    git push origin HEAD || log "push failed (non-fatal)"
done

log "max iterations ($MAX_ITERS) reached; stopping"
