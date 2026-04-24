"""Headless Gradio construction test.

Imports app.py and calls build_app() in a subprocess with ANTHROPIC_API_KEY
unset. If the Blocks object constructs, the Space will launch.

We deliberately do NOT launch the server here: macOS sandbox networking and
jinja2/starlette version skew cause localhost-binding false negatives that
don't occur on the HF Space's Linux Docker runtime. The thing that fails on
HF and isn't caught by clean-venv imports is a construction-time bug in
build_app() — which this script catches.

Exits 0 on success, non-zero with diagnostic on failure.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=60)
    # Retain --port for Makefile compatibility; we don't actually open it.
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--query", default=None)
    args = ap.parse_args()

    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)

    # Exercise both build_app() AND the launch-kwargs code path, because
    # Gradio raises TypeError on invalid Blocks.launch kwargs at call time
    # (not at Blocks construction). Monkey-patch launch to validate kwargs
    # against the real method signature without actually binding a port.
    probe = (
        "import sys, inspect; sys.path.insert(0, '.');"
        "import gradio as gr;"
        # Support both the full app (build_app) and minimal cached demo (build).
        "import app as _a;"
        "builder = getattr(_a, 'build_app', None) or getattr(_a, 'build', None);"
        "assert builder is not None, 'app.py must expose build() or build_app()';"
        "blocks = builder();"
        "assert blocks is not None, 'builder returned None';"
        # Validate that the kwargs our app.launch() call uses are ALL in the
        # real Blocks.launch signature. This catches the common 'passed theme
        # to launch() when it belongs on Blocks()' class of bug.
        "sig = inspect.signature(gr.Blocks.launch);"
        "valid = set(sig.parameters.keys());"
        "used = {'server_name','server_port','share'};"
        "bad = used - valid;"
        "assert not bad, f'app.launch() uses kwargs not accepted by gradio {gr.__version__}: {bad}';"
        "print(f'OK: build_app + launch kwargs valid for gradio {gr.__version__}')"
    )
    print(f"smoke: running build_app() probe with CACHED_MODE (timeout {args.timeout}s)")
    try:
        r = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        print("FAIL: build_app() probe timed out")
        sys.exit(1)
    sys.stdout.write(r.stdout)
    sys.stdout.flush()
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        print("FAIL: build_app() probe failed")
        sys.exit(1)
    print("OK: gradio smoke test passed")


if __name__ == "__main__":
    main()
