#!/usr/bin/env python3
"""scripts/run_guide_notebooks.py.

Test-run and clean the language guide notebooks under ``guides/*.ipynb``.

These notebooks are the single source for the published guide pages (the
docs converter turns each into an ``.mdx`` page). They are committed
*without* outputs so the diff stays small and the rendered page shows code, not
stale run artifacts. This script keeps them honest: it re-executes each notebook
top to bottom to prove the example still runs, then clears the outputs again.

A cell tagged ``docs:keep-output`` is the exception: its output is intentionally
shown on the page, so the output is kept (and sanitized) instead of cleared.

Typical usage::

    # Execute + clean every guide notebook (default)
    python scripts/run_guide_notebooks.py

    # Execute + clean one notebook
    python scripts/run_guide_notebooks.py --only building-constraints

    # Clean only, without executing (fast; for notebooks that need a GPU/cloud)
    python scripts/run_guide_notebooks.py --clear-only

    # Just list what would be processed
    python scripts/run_guide_notebooks.py --dry-run

Exits non-zero if any notebook fails to execute.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# Both the language guide notebooks and the example-program notebooks are test-run
# and cleaned the same way.
NOTEBOOK_DIRS = [REPO_ROOT / "guides", REPO_ROOT / "examples" / "notebooks"]

# Mime types that require a live kernel / runtime JS state to render, which a
# static notebook viewer cannot resolve. Always paired with a text/plain or
# text/html fallback — strip the live ones, keep the fallbacks. Only relevant
# for cells tagged ``docs:keep-output`` (every other cell's outputs are cleared).
_STRIP_MIMES = frozenset(
    {
        "application/vnd.jupyter.widget-view+json",  # ipywidgets (tqdm, sliders, etc.)
        "application/vnd.plotly.v1+json",  # plotly-native — text/html fallback renders fine
        "application/vnd.bokehjs_exec.v0+json",  # bokeh — HTML fallback renders fine
    }
)

# Text-bearing output mime types that may carry machine- or user-specific paths.
_TEXT_MIMES = frozenset({"text/plain", "text/html", "text/markdown"})


def _build_redaction_rules() -> list[tuple[re.Pattern[str], str]]:
    """Build ``(pattern, replacement)`` pairs that strip machine/user identifiers.

    Covers absolute paths descending into this repo (rewritten repo-relative),
    any user's home directory, and the running user's own home and username, so
    kept outputs don't leak local paths into the public repo.
    """
    repo = re.escape(REPO_ROOT.name)
    rules: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(?<![\w:/])(?:/[\w.+-]+)+/" + repo + r"/"), REPO_ROOT.name + "/"),
        (re.compile(r"/home/[^/\s\"']+"), "/home/user"),
        (re.compile(r"/Users/[^/\s\"']+"), "/Users/user"),
    ]
    home = os.path.expanduser("~")
    if home and home not in ("/", "/home", "/home/user"):
        rules.append((re.compile(re.escape(home)), "~"))
    user = getpass.getuser()
    if user and len(user) >= 3:
        rules.append((re.compile(r"\b" + re.escape(user) + r"\b"), "user"))
    return rules


_REDACTION_RULES = _build_redaction_rules()


def _redact_text(text: str) -> tuple[str, int]:
    """Apply every redaction rule to a string. Returns ``(new_text, count)``."""
    count = 0
    for pattern, replacement in _REDACTION_RULES:
        text, n = pattern.subn(replacement, text)
        count += n
    return text, count


def _redact_field(value: object) -> tuple[object, int]:
    """Redact a notebook text field that is a ``str`` or ``list[str]``."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        new_list: list[object] = []
        count = 0
        for item in value:
            if isinstance(item, str):
                red, n = _redact_text(item)
                new_list.append(red)
                count += n
            else:
                new_list.append(item)
        return new_list, count
    return value, 0


def discover_notebooks(only: str | None) -> list[Path]:
    """Return every guide / example-program notebook, filtered by ``only``."""
    notebooks = sorted(nb for d in NOTEBOOK_DIRS if d.is_dir() for nb in d.glob("*.ipynb"))
    if only:
        notebooks = [n for n in notebooks if only in n.stem]
    return notebooks


def execute_notebook(path: Path, timeout: int) -> tuple[bool, str]:
    """Run the notebook in place via ``jupyter nbconvert --execute --inplace``.

    Returns:
        tuple[bool, str]: ``(success, message)`` — ``"ok"`` on success, or the
        last line of stderr/stdout on failure.
    """
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except FileNotFoundError:
        return False, "jupyter not found (pip install nbconvert ipykernel)"
    if result.returncode != 0:
        last_line = (result.stderr or result.stdout or "nbconvert exit non-zero").strip().split("\n")[-1]
        return False, last_line
    return True, "ok"


def _sanitize_kept_outputs(cell: dict) -> tuple[int, int]:
    """Strip non-renderable mimes and redact identifiers in a kept cell's outputs."""
    stripped = 0
    redacted = 0
    for out in cell.get("outputs", []):
        if "text" in out:
            out["text"], n = _redact_field(out["text"])
            redacted += n
        data = out.get("data")
        if not isinstance(data, dict):
            continue
        for mime in list(data.keys()):
            if mime in _STRIP_MIMES:
                del data[mime]
                stripped += 1
            elif mime in _TEXT_MIMES:
                data[mime], n = _redact_field(data[mime])
                redacted += n
    return stripped, redacted


def clean_outputs(path: Path) -> tuple[int, int, int]:
    """Clean a guide notebook's outputs in place.

    Clears outputs for ordinary code cells; for cells tagged ``docs:keep-output``
    keeps the outputs but sanitizes them (strips live-only mimes, redacts paths).

    Returns:
        tuple[int, int, int]: ``(cleared, stripped, redacted)`` — cells cleared,
        mime entries stripped from kept cells, and identifier redactions applied.
    """
    nb = json.loads(path.read_text())
    cleared = 0
    stripped = 0
    redacted = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []))
        if "docs:keep-output" in tags:
            s, r = _sanitize_kept_outputs(cell)
            stripped += s
            redacted += r
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            cleared += 1
        cell["outputs"] = []
        cell["execution_count"] = None

    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]

    path.write_text(json.dumps(nb, indent=1) + "\n")
    return cleared, stripped, redacted


def _rel(path: Path) -> str:
    """Return ``path`` as a repo-relative string for user-facing messages."""
    return str(path.relative_to(REPO_ROOT))


def main() -> int:
    """Execute and/or clean the guide notebooks."""
    ap = argparse.ArgumentParser(
        description="Test-run language guide notebooks and clear their outputs for committing.",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="Substring filter on notebook stem (e.g. 'building-constraints')",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-notebook execution timeout in seconds (default: 1800). Cell timeout matches this.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List the discovered notebooks without executing or cleaning",
    )
    ap.add_argument(
        "--clear-only",
        action="store_true",
        help="Skip execution; only clear outputs. Fast; for notebooks that need a GPU/cloud to run.",
    )
    args = ap.parse_args()

    notebooks = discover_notebooks(args.only)
    if not notebooks:
        print("No guide notebooks matched.", flush=True)
        return 1

    print(f"Found {len(notebooks)} notebook(s):", flush=True)
    for n in notebooks:
        print(f"  {_rel(n)}", flush=True)
    print(flush=True)

    if args.dry_run:
        return 0

    mode = "clear-only" if args.clear_only else f"execute+clear (timeout {args.timeout}s/notebook)"
    print(f"Mode: {mode}", flush=True)
    print(flush=True)

    failures: list[tuple[Path, str]] = []
    total_cleared = 0
    total_stripped = 0
    total_redacted = 0

    progress = tqdm(notebooks, desc="Processing", unit="nb", file=sys.stderr)
    for nb_path in progress:
        rel = _rel(nb_path)
        progress.set_postfix_str(nb_path.stem)

        if args.clear_only:
            ok, msg = True, "ok"
        else:
            ok, msg = execute_notebook(nb_path, args.timeout)

        if ok:
            cleared, stripped, redacted = clean_outputs(nb_path)
            total_cleared += cleared
            total_stripped += stripped
            total_redacted += redacted
            print(f"  ok    {rel}  (cleared {cleared}, stripped {stripped}, redacted {redacted})", flush=True)
        else:
            failures.append((nb_path, msg))
            print(f"  FAIL  {rel}: {msg}", flush=True)

    progress.close()

    print(flush=True)
    print(
        f"Summary: {len(notebooks) - len(failures)}/{len(notebooks)} notebooks processed; "
        f"{total_cleared} cells cleared, {total_stripped} mimes stripped, {total_redacted} identifiers redacted.",
        flush=True,
    )

    if failures:
        print(flush=True)
        print(f"FAILURES ({len(failures)}):", flush=True)
        for nb_path, msg in failures:
            print(f"  {_rel(nb_path)}: {msg}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
