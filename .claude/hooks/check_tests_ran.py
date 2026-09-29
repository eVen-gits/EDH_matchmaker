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
    files = []
    for line in result.stdout.splitlines():
        path = line[3:]
        if " -> " in path:
            # Rename/copy: "old/path.py -> new/path.py" - only the new path exists on disk.
            path = path.split(" -> ", 1)[1]
        if path.endswith(".py"):
            files.append(path)
    return files


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

    def is_stale(f):
        path = os.path.join(REPO_ROOT, f)
        if not os.path.isfile(path):
            # Deleted file - no mtime of its own. Fall back to the nearest
            # surviving ancestor directory, whose mtime is bumped when an
            # entry is removed from it.
            path = os.path.dirname(path)
            while path and not os.path.isdir(path):
                path = os.path.dirname(path)
            path = path or REPO_ROOT
        return os.path.getmtime(path) > cache_mtime

    stale = [f for f in files if is_stale(f)]
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
