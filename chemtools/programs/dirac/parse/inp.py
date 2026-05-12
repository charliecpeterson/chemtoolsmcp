"""DIRAC `.inp` (job-control) file parser.

DIRAC inputs use a star-section / dot-keyword convention::

    **DIRAC              top-level section
    .WAVE FUNCTION       keyword inside **DIRAC
    .ANALYZE
    **HAMILTONIAN        next top-level section
    .DFT                 keyword inside **HAMILTONIAN
    *END OF INPUT

Some keywords take subsequent lines as arguments::

    .REORDER
    1,3..7,2

Some sections have sub-sections (a single ``*NAME`` line)::

    **WAVE FUNCTION
    .SCF
    *SCF
    .CLOSED SHELL
     44 44
    .OPEN SHELL
     1
     2/0,14

We tokenize into ``{section: {subsection: [{keyword: str, args: list[str]}]}}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_inp(path: str, contents: str | None = None) -> dict[str, Any]:
    """Parse a DIRAC `.inp` job-control file."""
    if contents is None:
        contents = Path(path).read_text(encoding="utf-8", errors="replace")

    sections: dict[str, dict[str, list[dict[str, Any]]]] = {}
    current_section: str | None = None
    current_subsection: str = "_default"
    pending_keyword: dict[str, Any] | None = None

    for raw in contents.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            # blank or comment
            continue
        if stripped.upper().startswith("*END OF INPUT"):
            break
        if stripped.startswith("**"):
            current_section = stripped[2:].strip()
            current_subsection = "_default"
            pending_keyword = None
            sections.setdefault(current_section, {}).setdefault(current_subsection, [])
            continue
        if stripped.startswith("*") and current_section is not None:
            current_subsection = stripped[1:].strip()
            pending_keyword = None
            sections[current_section].setdefault(current_subsection, [])
            continue
        if stripped.startswith(".") and current_section is not None:
            kw = stripped[1:].strip()
            pending_keyword = {"keyword": kw, "args": []}
            sections[current_section].setdefault(current_subsection, []).append(pending_keyword)
            continue
        if pending_keyword is not None:
            # continuation arg for the last keyword
            pending_keyword["args"].append(stripped)
            continue

    # Convenience roll-up: flat lists of which keywords appear under each top-level section
    keyword_index: dict[str, list[str]] = {}
    for sec, subs in sections.items():
        kw_list: list[str] = []
        for sub_entries in subs.values():
            for entry in sub_entries:
                kw_list.append(entry["keyword"])
        keyword_index[sec] = kw_list

    # DIRAC accepts "WAVE FUNCTION", "WAVE FUNCTIONS", and "WAVE F" — flatten
    # the section key for the boolean shortcuts so an agent doesn't have to
    # know which form the user typed.
    def _kws_for(*prefixes: str) -> list[str]:
        out: list[str] = []
        for sec, kws in keyword_index.items():
            sec_u = sec.upper()
            if any(sec_u.startswith(p) for p in prefixes):
                out.extend(kws)
        return out

    wf_kws = _kws_for("WAVE F")
    ham_kws = _kws_for("HAMILTONIAN")

    def _any_match(kws: list[str], needle: str) -> bool:
        return any(needle in (kw or "").upper() for kw in kws)

    return {
        "path": str(path),
        "sections": sections,
        "keyword_index": keyword_index,
        "has_reorder": _any_match(wf_kws, "REORDER"),
        "has_scf": _any_match(wf_kws, "SCF"),
        "has_dft": _any_match(ham_kws, "DFT"),
        "has_ecp": _any_match(ham_kws, "ECP"),
        "has_mp2": _any_match(wf_kws, "MP2"),
        "has_cosci": _any_match(wf_kws, "COSCI"),
        "has_open_shell": _any_match(wf_kws, "OPEN SHELL"),
        "has_closed_shell": _any_match(wf_kws, "CLOSED SHELL"),
        "is_response": "TDA" in keyword_index.get("RESPONSE", [])
                    or "RESPON" in (keyword_index.get("DIRAC", []) or []),
    }
