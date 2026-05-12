"""DIRAC docs accessor — search / lookup / read bundled documentation.

180 Markdown files under ``chemtools/programs/dirac/data/docs/`` carry the
official DIRAC user-guide content (basis, COSCI, AOC, ECP, RECP, atomic
start, checkpoint, HDF5 schema, etc.). This module mirrors the
``programs/molcas/docs.py`` pattern: list, search, look up, read excerpts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_DOCS_DIR = Path(__file__).parent / "data" / "docs"


def _iter_doc_files() -> list[Path]:
    if not _DOCS_DIR.exists():
        return []
    return sorted(_DOCS_DIR.glob("*.md"))


def _rel(path: Path) -> str:
    return path.name


def list_docs() -> list[dict[str, Any]]:
    """Return every bundled DIRAC doc file with size in bytes."""
    return [
        {"name": _rel(p), "size_bytes": p.stat().st_size}
        for p in _iter_doc_files()
    ]


def search_docs(
    query: str,
    *,
    max_hits: int = 8,
    context_lines: int = 2,
) -> dict[str, Any]:
    """Substring + word-boundary search across the doc corpus.

    Returns up to ``max_hits`` matches, each with the file name, the
    matching line + ``context_lines`` of surrounding text, and a 1-based
    line number suitable for ``read_doc_excerpt``.
    """
    needle = query.strip().lower()
    if not needle:
        return {"query": query, "hits": [], "total_hits": 0}

    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    hits: list[dict[str, Any]] = []
    total = 0

    for p in _iter_doc_files():
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if pattern.search(line):
                total += 1
                if len(hits) >= max_hits:
                    continue
                lo = max(0, i - context_lines)
                hi = min(len(lines), i + context_lines + 1)
                snippet = "\n".join(lines[lo:hi])
                hits.append({
                    "doc": _rel(p),
                    "line": i + 1,
                    "match": line.strip(),
                    "snippet": snippet,
                })

    return {"query": query, "hits": hits, "total_hits": total}


def lookup_section(
    section: str,
    *,
    max_results: int = 1,
) -> dict[str, Any]:
    """Look up a DIRAC section / keyword name (e.g. "WAVE FUNCTION", "AOC",
    "REORDER", "MOLECULE"). Returns the top doc file(s) most likely to
    document that section based on title match + first-paragraph match.
    """
    needle = section.strip().lower()
    candidates: list[tuple[int, Path, str]] = []

    for p in _iter_doc_files():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        score = 0
        # Filename match (e.g. "wave_function.md")
        if needle.replace(" ", "_") in p.stem.lower():
            score += 100
        if needle in p.stem.lower():
            score += 50
        # H1 / H2 match
        head = text[:2000]
        for h in re.findall(r"^#{1,3}\s+(.+)$", head, re.MULTILINE):
            if needle in h.lower():
                score += 30
        # First-paragraph match
        if needle in head.lower():
            score += 5
        if score:
            candidates.append((score, p, text))

    candidates.sort(key=lambda t: -t[0])
    top = candidates[:max_results]
    return {
        "section": section,
        "matches": [
            {
                "doc": _rel(p),
                "score": score,
                "preview": text[:600],
            }
            for score, p, text in top
        ],
        "total_candidates": len(candidates),
    }


def read_doc_excerpt(
    name: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    max_lines: int = 200,
) -> dict[str, Any]:
    """Read an excerpt of a bundled doc by name. With no line range, returns
    the first ``max_lines``. With a range, slices that block.
    """
    matching = [p for p in _iter_doc_files() if p.name == name]
    if not matching:
        # Try a partial match
        matching = [p for p in _iter_doc_files() if name.lower() in p.name.lower()]
    if not matching:
        return {"doc": name, "found": False, "available_count": len(_iter_doc_files())}
    p = matching[0]
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    n = len(lines)

    if start_line is None:
        start = 0
        end = min(max_lines, n)
    else:
        start = max(0, start_line - 1)
        end = min(n, (end_line if end_line is not None else start + max_lines))
    return {
        "doc": p.name,
        "found": True,
        "n_lines_total": n,
        "start_line": start + 1,
        "end_line": end,
        "text": "\n".join(lines[start:end]),
    }


# Curated topic guides for high-value subjects an agent encounters often.
_TOPIC_GUIDES: dict[str, dict[str, Any]] = {
    "aoc": {
        "summary": (
            "Average-of-Configurations (AOC) is DIRAC's open-shell HF approach. "
            "There is NO ROHF in DIRAC — spin-orbit coupling makes the formalism "
            "incompatible with pure spin symmetry. Instead, AOC averages the "
            "Fock matrix over a set of Slater determinants generated by "
            "distributing N_open electrons among M_open spinors. Input directives "
            "live under **WAVE FUNCTIONS / *SCF: .CLOSED SHELL gives per-fsym "
            "closed counts, .OPEN SHELL declares the count of open shells and "
            "for each a line ``N_electrons/sym1,sym2`` listing per-fsym spinor "
            "counts. RESOLVE prints per-symmetry HOMO/LUMO at each iteration."
        ),
        "key_docs": ["aoc.md", "open_shell_scf.md"],
        "agent_pattern": (
            "1. parse_dirac_input — check for .OPEN SHELL block.\n"
            "2. read_orbital_summary(h5, fractional_only=True) — see which "
            "spinors carry the open electrons (occ ≈ N_open / M_spinors).\n"
            "3. If swap needed, draft .REORDER block under *SCF and restart."
        ),
    },
    "cosci": {
        "summary": (
            "COSCI = Complete Open-Shell CI, run after an AOC SCF. Diagonalizes "
            "the CI matrix in the open-shell space using AOC orbitals fixed, "
            "yielding individual ms-coupled states (singlet/triplet/etc. but for "
            "4c, J-coupled microstates)."
        ),
        "key_docs": ["cosci.md", "krcicalc.md"],
        "agent_pattern": (
            "Run AOC SCF first, then add **COSCI section with state spec; "
            "parse output for per-state energies + dominant CI weights."
        ),
    },
    "reorder": {
        "summary": (
            ".REORDER under *SCF reorders the starting orbitals before SCF runs. "
            "Format: comma-separated list of original indices in the new order, "
            "with ``a..b`` ranges. Example: ``2..7,1`` means orbital 1 moves to "
            "the end (after originals 2-7). Used to fix incorrect open-shell "
            "starting guesses by promoting valence orbitals into the open shell."
        ),
        "key_docs": ["reorder.md"],
        "agent_pattern": (
            "1. Inspect MOs via read_orbital_summary(h5).\n"
            "2. Identify wrong-character orbitals in the open-shell space.\n"
            "3. Draft .REORDER spec swapping the bad ones with correct-character "
            "candidates from the virtual/closed sets.\n"
            "4. Restart SCF with the modified input + same starting checkpoint."
        ),
    },
    "atomic_start": {
        "summary": (
            "Atomic-start workflow: run per-element atomic SCF first, save each "
            ".h5, then feed all of them into the molecular run via "
            "``--copy=\"A.h5 B.h5 ...\"`` so the per-atom orbitals seed the "
            "molecular SCF. Improves convergence for difficult cases (heavy "
            "elements, transition metals, open shell). Atomic .h5 files must be "
            "named to match the molecule's element labels in the .mol."
        ),
        "key_docs": ["atomic_start.md", "atomic_start2.md", "atomic_huckel.md"],
        "agent_pattern": (
            "1. For each unique element: draft per-atom .inp + .mol, run SCF, "
            "save the .h5.\n"
            "2. For the molecule: launch with ``--copy=\"Elem1.h5 Elem2.h5\"``.\n"
            "3. DIRAC reads each atomic .h5 and projects onto molecular orbitals "
            "as starting guess."
        ),
    },
    "checkpoint": {
        "summary": (
            "DIRAC ≥ 22 writes .h5 checkpoints alongside the text output. Older "
            "DFCOEF / DFPCMO / DFACMO Fortran-binary files are still used by "
            "some modules. ``--get=DFACMO`` requests the binary on disk; "
            "``--outcmo`` writes the converged molecular orbitals."
        ),
        "key_docs": ["checkpoint.md", "dfcoef.md", "dfcoef_and_dfpcmo.md"],
        "agent_pattern": (
            "Read MO data from .h5 via the binary reader (read_orbital_summary, "
            "read_mo_coefficients). DFCOEF/DFPCMO files are for inter-run "
            "restart only; their contents duplicate what's in the .h5."
        ),
    },
    "ecp": {
        "summary": (
            "ECPs / RECPs go in the .mol file. ECP segment counts go in the "
            "C-coord header (e.g. ``C   N_TYPES   N_SYMOPS  Y  Z  A``); each "
            "atomtype block then carries the ECP block after LARGE/SMALL "
            "basis. Use ``**HAMILTONIAN / .ECP`` to activate."
        ),
        "key_docs": ["ecp.md", "recp.md"],
        "agent_pattern": (
            "When drafting an input with heavy elements: check elements > Z=54, "
            "decide ECP vs all-electron DKH, draft .mol with explicit ECP block "
            "or library ECP name, add ``.ECP`` to **HAMILTONIAN."
        ),
    },
}


def get_topic_guide(topic: str) -> dict[str, Any]:
    """Look up a curated topic guide. Recognized topics: aoc, cosci, reorder,
    atomic_start, checkpoint, ecp.
    """
    key = topic.strip().lower().replace(" ", "_").replace("-", "_")
    guide = _TOPIC_GUIDES.get(key)
    if guide is None:
        return {
            "topic": topic,
            "available_topics": sorted(_TOPIC_GUIDES.keys()),
            "found": False,
        }
    return {"topic": key, "found": True, **guide}
