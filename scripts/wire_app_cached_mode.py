"""Ensure app.py loads data/demo_cache.jsonl in cached mode.

app.py already has a CACHED_MODE flag and a _cached_answer() helper that
consumes data/cached_demo_answers.json. We (a) keep that working, and
(b) extend it to also read demo_cache.jsonl (the newly generated cache).
Idempotent.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

APP = Path("app.py")
DEMO_CACHE = Path("data/demo_cache.jsonl")

SENTINEL = "# --- demo_cache.jsonl loader (wire_app_cached_mode) ---"

NEW_BLOCK = f'''
{SENTINEL}
_demo_cache_v2 = None


def _load_demo_cache_v2() -> dict:
    """Load demo_cache.jsonl (one JSON entry per line) as question->entry dict."""
    global _demo_cache_v2
    if _demo_cache_v2 is None:
        _demo_cache_v2 = {{}}
        p = Path(__file__).parent / "data" / "demo_cache.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q = e.get("question")
                    if q:
                        _demo_cache_v2[q] = e
    return _demo_cache_v2
'''.strip()


def main():
    text = APP.read_text()
    if SENTINEL in text:
        print("app.py already wired")
        return

    # Insert helper after imports (after the _load_cache() helper)
    anchor = "def _load_cache() -> dict:"
    if anchor not in text:
        print(f"Anchor '{anchor}' not found in app.py; skipping.")
        return
    idx = text.find(anchor)
    # Insert NEW_BLOCK after first function ends — find first double-newline after anchor
    end = text.find("\n\n\n", idx)
    if end < 0:
        end = text.find("\n\n", idx + 20)
    text = text[:end] + "\n\n\n" + NEW_BLOCK + "\n" + text[end:]

    # Patch _cached_answer to also try demo_cache_v2
    old = '    cache = _load_cache()\n    entry = cache.get(question)'
    new = (
        '    cache = _load_cache()\n'
        '    entry = cache.get(question)\n'
        '    if not entry:\n'
        '        v2 = _load_demo_cache_v2()\n'
        '        v2_entry = v2.get(question)\n'
        '        if v2_entry:\n'
        '            return (\n'
        '                v2_entry.get("answer", ""),\n'
        '                "\\n".join(f"- ACN **{a}**" for a in v2_entry.get("sources", [])[:8]) or "*no sources*",\n'
        '                "**Cached demo** | latency {:.0f}ms retrieval + {:.0f}ms generation".format(\n'
        '                    v2_entry.get("retrieval_latency_ms", 0),\n'
        '                    v2_entry.get("generation_latency_ms", 0),\n'
        '                ),\n'
        '            )'
    )
    if old in text:
        text = text.replace(old, new)
    APP.write_text(text)
    print("app.py wired with demo_cache.jsonl fallback")


if __name__ == "__main__":
    main()
