"""
run_tests.py
============
Runs the test suite using  python -m pytest  so pytest does NOT need to be
on your system PATH.  Works on Windows, macOS, and Linux.

Usage (from the project root):
    python run_tests.py          # run all tests with verbose output
    python run_tests.py -q       # quiet (dots only)
    python run_tests.py -k nb    # run only tests whose name contains "nb"

This is equivalent to:
    python -m pytest tests/test_suite.py -v
"""

import subprocess
import sys
import os

# ── Make sure we run from the project root ─────────────────────────────────────
root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)

# ── Check that pytest is importable (installed in the current environment) ─────
try:
    import pytest                                          # noqa: F401
except ImportError:
    print("\n[ERROR] pytest is not installed in this Python environment.")
    print("        Run one of the following commands first:\n")
    print("          pip install pytest")
    print("          pip install -r requirements.txt\n")
    sys.exit(1)

# ── Build the command ──────────────────────────────────────────────────────────
# Pass any extra arguments the user typed after run_tests.py through to pytest.
extra_args = sys.argv[1:]          # e.g. ["-q"] or ["-k", "nb"]

cmd = [
    sys.executable,                # the exact Python being used right now
    "-m", "pytest",
    "tests/test_suite.py",
    "-v",                          # verbose by default
    "--tb=short",                  # compact traceback on failures
] + extra_args

print(f"\nRunning: {' '.join(cmd)}\n{'=' * 60}\n")

result = subprocess.run(cmd, cwd=root)
sys.exit(result.returncode)
