"""GRASP2018 docs accessor — search / lookup / read bundled documentation.

15 Markdown files under ``chemtools/data/grasp/docs/`` carry the official
GRASP2018 manual split into four parts (overview, CSF generation, sample
runs, convergence troubleshooting). This module mirrors the DIRAC / Molcas
docs pattern: list, search, lookup, read excerpts, topic guides.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_DOCS_DIR = Path(__file__).parents[2] / "data" / "grasp" / "docs"


def _iter_doc_files() -> list[Path]:
    if not _DOCS_DIR.exists():
        return []
    # Recurse into the part_i/part_ii/part_iii/part_iv subdirectories.
    return sorted(_DOCS_DIR.rglob("*.md"))


def _rel(path: Path) -> str:
    """Return the path relative to the docs root (e.g. ``part_i/01_GRASP2018.md``)."""
    try:
        return str(path.relative_to(_DOCS_DIR))
    except ValueError:
        return path.name


def list_docs() -> list[dict[str, Any]]:
    """Return every bundled GRASP doc with relative path + size in bytes."""
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
    """Substring search across the GRASP doc corpus.

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


def lookup_section(section: str, *, max_results: int = 5) -> dict[str, Any]:
    """Look up a GRASP exe / section / keyword (e.g. "rmcdhf", "rcsfgenerate",
    "Breit", "convergence"). Returns the top doc files most likely to
    document that section based on filename + H1/H2 match."""
    needle = section.strip().lower()
    candidates: list[tuple[int, Path, str]] = []

    for p in _iter_doc_files():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        score = 0
        stem = p.stem.lower()
        if needle.replace(" ", "_") in stem:
            score += 100
        if needle in stem:
            score += 50
        head = text[:2000]
        for h in re.findall(r"^#{1,3}\s+(.+)$", head, re.MULTILINE):
            if needle in h.lower():
                score += 30
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
                "preview": text[:300],
            }
            for score, p, text in top
        ],
    }


def read_doc_excerpt(
    name: str,
    *,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read a slice of a bundled doc. ``name`` is the relative path returned
    by ``list_docs`` (e.g. ``part_iii_sample_runs/01_Running_the_application_programs.md``)."""
    target = _DOCS_DIR / name
    if not target.exists() or not target.is_file():
        # Try looking it up by basename too (lenient match)
        candidates = [p for p in _iter_doc_files() if p.name == name]
        if not candidates:
            return {"error": "doc_not_found", "name": name,
                    "available": [_rel(p) for p in _iter_doc_files()]}
        target = candidates[0]

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    s = max(1, start_line)
    e = end_line if end_line is not None else total
    e = min(e, total)
    slice_text = "\n".join(lines[s - 1:e])
    return {
        "doc": _rel(target),
        "start_line": s,
        "end_line": e,
        "total_lines": total,
        "text": slice_text,
    }


# =============================================================================
# Curated topic guides — high-value cheatsheets for common workflows.
# =============================================================================

_TOPIC_GUIDES: dict[str, dict[str, Any]] = {
    "csf_generation": {
        "summary": (
            "rcsfgenerate builds the configuration state function (CSF) "
            "list that defines which determinants enter the MCDHF expansion.\n"
            "Configurations are entered in spectroscopic notation with "
            "(occ, marker) where marker is one of:\n"
            "  i — inactive (closed-shell, occupation fixed)\n"
            "  * — active (can excite)\n"
            "  c — closed (always doubly occupied)\n"
            "  N — minimum N electrons must remain\n\n"
            "Example for Cf 5f^10 7s^2:\n"
            "  7s(2,i)5f(10,*)\n\n"
            "Active orbital set defines the n,l ceiling for excitations: "
            "'7s,6p,5d,5f' means s up to 7, p up to 6, d up to 5, f up to 5.\n\n"
            "Excitations parameter:\n"
            "  0  — no excitations (Hartree-Fock-like)\n"
            "  N>0 — up to N excitations from the reference\n"
            "  N<0 — correlation orbitals always doubly occupied"
        ),
        "key_docs": ["part_ii_generating_lists_of_csfs/01_Lists_of_CSFs.md",
                     "part_ii_generating_lists_of_csfs/02_Running_the_CSFs_generation_programs.md"],
        "agent_pattern": (
            "1. Define a configuration list with the right markers (i for "
            "closed core, * for active shell).\n"
            "2. Set active_orbitals to bound the correlation space.\n"
            "3. Set twoj_min/twoj_max to span the J values you want.\n"
            "4. excitations=0 unless you specifically want CI-like correlation."
        ),
    },
    "convergence_debugging": {
        "summary": (
            "GRASP's rmcdhf has a 50-iteration hard cap by default. When "
            "convergence fails, the typical culprits are:\n\n"
            "1. Bad starting orbitals (Thomas-Fermi gave radial wave funcs "
            "that diverge for some subshells — TFWAVE error). Common for "
            "high-Z atoms. Fix: use the hf-bootstrap workflow (run hf code "
            "first, convert via rwfnmchfmcdf, then DHF picks up rwfn.inp).\n\n"
            "2. Block-level-selection mismatch — the per-block ASF serial "
            "numbers passed to rmcdhf don't match the number of CSFs in "
            "each block. Symptom: 'serial numbers must be in range [1,N]' "
            "followed by a Fortran 'End of file' crash. Fix: count blocks "
            "from rcsfgenerate output and pass exactly that many "
            "block_level_selections.\n\n"
            "3. Orbital diverging during SCF (rare, usually high-Z). Try "
            "increasing max_scf_cycles to 200+ or use the non-rel limit "
            "workflow first to get rough orbitals, then restart at c=137.\n\n"
            "4. mcp.30 / mcp.* not found — rangular wasn't run or its output "
            "got cleaned up between steps."
        ),
        "key_docs": ["part_iv_issues_of_convergence_trouble_/01_Methods_to_ensure_convergence.md"],
    },
    "nonrel_limit": {
        "summary": (
            "Setting the speed of light to a very large value (default "
            "2000 au, vs. physical 137.036 au) suppresses all relativistic "
            "effects in the Dirac-Coulomb Hamiltonian. The Hamiltonian "
            "smoothly reduces to the non-relativistic Schrödinger limit as "
            "c → ∞.\n\n"
            "Use cases:\n"
            "  - Compare DHF results to non-rel HF benchmarks\n"
            "  - Isolate relativistic contributions to a property\n"
            "  - Verify fine-structure splittings are physical (not numerical)\n\n"
            "Both rwfnestimate AND rmcdhf need the speed_of_light_au=2000 "
            "setting — they each prompt for the override. The workflow "
            "orchestrator plan_grasp_nonrel_limit_workflow handles both.\n\n"
            "Expected results: J=L+S and J=L-S levels degenerate (split "
            "in DHF). For Li 2P: DHF splitting ~0.56 cm-1, NR-limit ~0.01 cm-1."
        ),
        "key_docs": ["part_i_overview_of_grasp2018/04_Important_concepts_and_aspects_of_processing.md"],
    },
    "hf_bootstrap": {
        "summary": (
            "For high-Z atoms (Z≥80, especially Cf/Bk/Es/Fm) the default "
            "Thomas-Fermi starting orbital estimate diverges (TFWAVE error) "
            "for some inner subshells. The fix is a 2-step bootstrap:\n\n"
            "  1. Run hf (non-relativistic Hartree-Fock, ships with GRASP) "
            "to get converged non-rel orbitals (wfn.out).\n"
            "  2. rwfnmchfmcdf converts wfn.out → rwfn.out in GRASP format.\n"
            "  3. Copy rwfn.out → rwfn.inp; rwfnestimate auto-picks it up "
            "as the GRASP92 File source (instead of Thomas-Fermi).\n"
            "  4. rmcdhf converges in the usual way.\n\n"
            "The plan_grasp_hf_bootstrap_workflow orchestrator chains all "
            "this together. Required args: element_symbol (e.g. 'Cf'), "
            "hf_orbital_list (closed orbitals as space-separated string), "
            "hf_open_shell (open-shell occupation like '5f(10)')."
        ),
        "key_docs": ["part_iii_sample_runs/01_Running_the_application_programs.md"],
    },
    "level_interpretation": {
        "summary": (
            "rlevels prints the energy-level table with these columns:\n"
            "  No      — sequential index across all blocks\n"
            "  Pos     — position within its (J, parity) block\n"
            "  J       — total angular momentum (integer or half-integer)\n"
            "  Parity  — + or -\n"
            "  E (au)  — absolute energy in hartrees\n"
            "  E (cm⁻¹)— energy relative to the ground state, in wavenumbers\n"
            "  Splitting (cm⁻¹) — energy difference from the previous level\n"
            "  Config  — LSJ-coupling label (only filled if jj2lsj ran with "
            "mixing_coefficients=True; otherwise blank)\n\n"
            "For LSJ composition (the actual coupling structure of each "
            "level), see the .lsj.lbl file from jj2lsj. Each level lists "
            "its decomposition over LS-coupled basis functions with "
            "coefficient + weight_fraction.\n\n"
            "Use parse_grasp_levels for the table, summarize_grasp_terms "
            "to group by LSJ term, parse_grasp_lsjlbl for compositions."
        ),
        "key_docs": ["part_iii_sample_runs/03_Interpreting_the_output_files.md"],
    },
}


def list_topics() -> list[str]:
    """Topic names recognized by ``get_topic_guide``."""
    return sorted(_TOPIC_GUIDES.keys())


def get_topic_guide(topic: str) -> dict[str, Any]:
    """Return a curated GRASP cheatsheet for ``topic`` (e.g. ``'csf_generation'``,
    ``'convergence_debugging'``, ``'nonrel_limit'``, ``'hf_bootstrap'``,
    ``'level_interpretation'``)."""
    if topic not in _TOPIC_GUIDES:
        return {
            "error": "unknown_topic",
            "topic": topic,
            "available_topics": list_topics(),
        }
    payload = dict(_TOPIC_GUIDES[topic])
    payload["topic"] = topic
    return payload
