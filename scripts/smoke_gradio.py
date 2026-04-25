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

    probe = (
        "import sys; sys.path.insert(0, '.');"
        "from app import build_app, CACHED_MODE;"
        "assert CACHED_MODE, 'expected CACHED_MODE=True with no API key';"
        "blocks = build_app();"
        "assert blocks is not None, 'build_app returned None';"
        "print('OK: build_app constructed Blocks cleanly')"
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
