#!/usr/bin/env python3
"""Stop hook: block ending the turn if src/tests changed since pytest last ran.

Freshness signal is .pytest_cache's mtime (pytest touches it on every run) -
no separate marker file needed. Skippable via SKIP_TESTS_HOOK=1 (human-only).
"""
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WATCHED_DIRS = ("src", "tests")


def changed_files():
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *WATCHED_DIRS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return [line[3:] for line in result.stdout.splitlines() if line[3:].endswith(".py")]


def main():
    if os.environ.get("SKIP_TESTS_HOOK") == "1":
        return 0

    payload = json.loads(sys.stdin.read() or "{}")
    if payload.get("stop_hook_active"):
        # Already blocked once this turn - don't loop forever.
        return 0

    files = changed_files()
    if not files:
        return 0

    # nodeids is rewritten by pytest's collector on every run; the cache
    # *directory's* own mtime only changes when entries are added/removed,
    # not when an existing file inside it is overwritten.
    cache_marker = os.path.join(REPO_ROOT, ".pytest_cache", "v", "cache", "nodeids")
    cache_mtime = os.path.getmtime(cache_marker) if os.path.isfile(cache_marker) else 0

    stale = [f for f in files if os.path.getmtime(os.path.join(REPO_ROOT, f)) > cache_mtime]
    if stale:
        print(
            "Uncommitted changes under src/ or tests/ have no fresh pytest run:\n  "
            + "\n  ".join(stale)
            + "\nRun `PYTHONPATH=. pytest` before finishing.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
