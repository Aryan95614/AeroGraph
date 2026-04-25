"""Verify that the Space's declared Python runtime matches what we test against.

Reads the README YAML frontmatter that deploy_hf_space.py embeds into the
Space and asserts:
  - python_version is pinned (not left default)
  - If python_version >= 3.13, audioop-lts is in requirements
  - sdk_version is set
Exits non-zero on any mismatch.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

DEPLOY = Path("scripts/deploy_hf_space.py")


def extract(name: str, text: str) -> str:
    m = re.search(rf"{name} = '''(.*?)'''", text, re.S) or re.search(
        rf'{name} = """(.*?)"""', text, re.S
    )
    return m.group(1) if m else ""


def main():
    if not DEPLOY.exists():
        print(f"FAIL: {DEPLOY} missing")
        sys.exit(1)
    src = DEPLOY.read_text()

    # Pull the inline space_readme + requirements blocks.
    m_readme = re.search(r'space_readme = """(.*?)"""', src, re.S)
    m_reqs = re.search(r'requirements = """(.*?)"""', src, re.S)
    if not (m_readme and m_reqs):
        print("FAIL: can't locate space_readme or requirements in deploy_hf_space.py")
        sys.exit(1)

    readme = m_readme.group(1)
    reqs = m_reqs.group(1)

    # 1. python_version pinned
    m_py = re.search(r'python_version:\s*["\']?(\d+\.\d+)["\']?', readme)
    if not m_py:
        print("FAIL: python_version not pinned in space_readme frontmatter")
        sys.exit(1)
    py = m_py.group(1)
    major, minor = map(int, py.split("."))
    print(f"  python_version: {py}")

    # 2. If 3.13+, require audioop-lts
    if (major, minor) >= (3, 13):
        if "audioop-lts" not in reqs:
            print(f"FAIL: python_version {py} requires audioop-lts in requirements.txt")
            sys.exit(1)
        print("  audioop-lts present (required for 3.13+)")
    else:
        print(f"  audioop-lts not required at Python {py}")

    # 3. sdk_version set
    if "sdk_version" not in readme:
        print("FAIL: sdk_version not set in space_readme")
        sys.exit(1)
    sdk = re.search(r"sdk_version:\s*([\w.]+)", readme).group(1)
    print(f"  sdk_version: {sdk}")

    # 4. app_file must be app.py (or exist)
    m_app = re.search(r"app_file:\s*(\S+)", readme)
    if not m_app or not Path(m_app.group(1)).exists():
        print(f"FAIL: app_file does not exist: {m_app.group(1) if m_app else '(unset)'}")
        sys.exit(1)
    print(f"  app_file: {m_app.group(1)}")

    print("OK: space metadata parity verified")


if __name__ == "__main__":
    main()
