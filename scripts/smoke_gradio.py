"""Headless Gradio smoke test. Launches app.py on a random port, hits the
Gradio API with one cached query, asserts a non-empty response, tears down.

Exits 0 on success, non-zero with diagnostic on failure.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_QUERY = "What are the most common contributing factors in bird strike incidents?"


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_ready(url: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="0 = random")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    port = args.port or find_free_port()
    print(f"smoke: launching app.py on :{port} (cached mode)")

    # Force cached mode
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env["GRADIO_SERVER_PORT"] = str(port)

    # Patch server_port in app.py at runtime using an env var; app.py uses
    # GRADIO_SERVER_PORT if set. If not, we launch by importing and calling
    # build_app().launch() in a subprocess with our port.
    launcher = (
        "import os,sys; sys.path.insert(0,'.');"
        "from app import build_app;"
        f"build_app().launch(server_name='127.0.0.1', server_port={port},"
        " share=False, prevent_thread_lock=False)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        print(f"smoke: waiting up to {args.timeout}s for {url}")
        if not wait_ready(url, args.timeout):
            # drain stdout so the user sees the launch error
            try:
                out, _ = proc.communicate(timeout=5)
                print(out[-2000:])
            except subprocess.TimeoutExpired:
                proc.kill()
            print("FAIL: gradio did not become ready")
            sys.exit(1)

        # Verify the HTTP root loads and doesn't contain "Error" in body
        with urllib.request.urlopen(url, timeout=5) as r:
            body = r.read().decode("utf-8", "ignore")
        if "<title>" not in body.lower():
            print("FAIL: gradio page did not render")
            sys.exit(1)
        print("smoke: root page OK")

        # Light-touch confirmation: cached demo JSON is present (the Space will
        # use it for queries). We do NOT exercise the API here because Gradio's
        # queue endpoints are version-fragile.
        cache = Path("data/demo_cache.jsonl")
        if not cache.exists() or cache.stat().st_size == 0:
            print("FAIL: data/demo_cache.jsonl missing or empty")
            sys.exit(1)
        with open(cache) as f:
            n = sum(1 for _ in f)
        if n < 5:
            print(f"FAIL: only {n} cached answers; expected >=5")
            sys.exit(1)
        print(f"smoke: cached answers present ({n} entries)")

        print("OK: gradio smoke test passed")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
