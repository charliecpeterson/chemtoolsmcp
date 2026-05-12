"""Session log + input-versioning helpers (program-agnostic).

The MCP tools ``init_session_log`` / ``append_session_log`` /
``next_versioned_path`` are tagged ``program="generic"`` and have always
been usable across programs — this module just relocates the impl into
``core/`` so future programs don't have to import from
``programs/nwchem/runner.py``.

Behavior is unchanged from the original NWChem-located versions.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any


_TYPE_EMOJI = {
    "step": "▶",
    "result": "✓",
    "error": "✗",
    "note": "◆",
    "summary": "★",
}


def init_session_log(
    log_path: str,
    session_title: str = "Chemtools Session",
    working_dir: str | None = None,
) -> dict[str, Any]:
    """Create (or overwrite) a Markdown session-log file with a header.

    The session log is a human-readable narrative the agent appends to as
    it works through a multi-step task. Calls to ``append_session_log``
    add timestamped entries; ``summary``-type entries close out the log.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"# {session_title}\n\n"
        f"**Started:** {ts}  \n"
        f"**Working directory:** {working_dir or 'unknown'}  \n\n"
        "---\n\n"
    )
    Path(log_path).write_text(header, encoding="utf-8")
    return {"log_path": log_path, "created": True, "timestamp": ts}


def append_session_log(
    log_path: str,
    entry_type: str,
    content: str,
) -> dict[str, Any]:
    """Append an entry to a session log.

    Parameters
    ----------
    entry_type
        One of ``step`` / ``result`` / ``error`` / ``note`` / ``summary``.
        Each gets a glyph prefix; unknown values get ``•``.
    content
        Markdown-formatted body. Stripped of leading/trailing whitespace.
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    marker = _TYPE_EMOJI.get(entry_type, "•")
    entry = f"## {marker} [{ts}] {entry_type.title()}\n\n{content.strip()}\n\n---\n\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
    return {"log_path": log_path, "appended": True, "timestamp": ts}


def next_versioned_path(path: str) -> str:
    """Return ``path`` with ``_v2`` / ``_v3`` / ... appended (before the
    extension) if the file exists; the original path otherwise.

    Useful for "never overwrite an input deck" workflows:

        next_versioned_path("fe.nw")          # 'fe.nw'      (doesn't exist yet)
        next_versioned_path("fe.nw")          # 'fe_v2.nw'   (after fe.nw exists)
        next_versioned_path("fe_v2.nw")       # 'fe_v3.nw'   (normalizes existing _vN)
    """
    p = Path(path)
    if not p.exists():
        return path
    stem = p.stem
    base_stem = re.sub(r"_v\d+$", "", stem)
    n = 2
    while True:
        candidate = p.parent / f"{base_stem}_v{n}{p.suffix}"
        if not candidate.exists():
            return str(candidate)
        n += 1
