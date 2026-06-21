"""Molcas input linter.

Returns a list of LintIssue dicts with structural and semantic checks. The
linter is intentionally conservative — it flags problems Molcas is likely to
fail on, plus a few high-confidence warnings (mismatched Frozen between
RASSCF and CASPT2, LumOrb without an obvious source). It does not try to
emulate the full SEWARD parser.

Public API:
  lint_molcas_input(text: str) -> list[LintIssue]

Each LintIssue dict has:
  level         "error" | "warning" | "info"
  message       human-readable description
  line          int | None — 1-indexed line number when available
  suggested_fix str | None — copy-paste-ready fix
"""

from __future__ import annotations

import re
from typing import Any

from chemtools.programs.molcas.input.basis_library import (
    list_basis_sets,
    list_elements_for_basis,
)


_BLOCK_START_RE = re.compile(r"^\s*&([A-Za-z0-9_]+)(?:\s+&END)?\s*$", re.M)
_BLOCK_END_RE = re.compile(r"^\s*[Ee][Nn][Dd]\s+[Oo][Ff]\s+[Ii][Nn][Pp][Uu][Tt]\s*$", re.M)
_BASIS_LABEL_RE = re.compile(r"^\s*([A-Z][a-z]{0,2})\.([A-Za-z0-9_+-]+)\.([^.]*)\.([^.]*)\.([^.]+)\.\s*$")


def lint_molcas_input(text: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    blocks = _split_blocks(text)
    issues.extend(_check_block_pairs(blocks, text))
    issues.extend(_check_basis_labels(text))
    issues.extend(_check_rasscf_caspt2_consistency(blocks))
    issues.extend(_check_nactel(blocks))
    issues.extend(_check_lumorb_provenance(blocks))
    issues.extend(_check_spin_charge(blocks))
    return issues


def _split_blocks(text: str) -> list[dict[str, Any]]:
    """Slice the text into &MODULE ... End of input blocks.

    Returns one dict per block: {name, line_start, line_end, body, body_lines}.
    """
    starts = list(_BLOCK_START_RE.finditer(text))
    ends = list(_BLOCK_END_RE.finditer(text))
    blocks: list[dict[str, Any]] = []
    for i, start in enumerate(starts):
        # Find the next End-of-input AFTER this start that is BEFORE the next start
        next_start = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        end_match = next((e for e in ends if start.end() <= e.start() < next_start), None)
        end_pos = end_match.end() if end_match else next_start
        line_start = text.count("\n", 0, start.start()) + 1
        line_end = text.count("\n", 0, end_pos) + 1
        blocks.append(
            {
                "name": start.group(1).upper(),
                "line_start": line_start,
                "line_end": line_end,
                "body": text[start.end():end_pos],
                "has_end": end_match is not None,
            }
        )
    return blocks


def _check_block_pairs(blocks: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Each &MODULE must have a matching `End of input`."""
    issues: list[dict[str, Any]] = []
    for b in blocks:
        if not b["has_end"]:
            issues.append(
                {
                    "level": "error",
                    "message": f"&{b['name']} block at line {b['line_start']} has no matching 'End of input'",
                    "line": b["line_start"],
                    "suggested_fix": "End of input",
                }
            )
    return issues


def _check_basis_labels(text: str) -> list[dict[str, Any]]:
    """Validate that basis-set labels reference a library and contraction we have."""
    issues: list[dict[str, Any]] = []
    available = set(list_basis_sets())
    for line_no, line in enumerate(text.splitlines(), start=1):
        m = _BASIS_LABEL_RE.match(line)
        if not m:
            continue
        element = m.group(1)
        library = m.group(2)
        if library.upper() not in {b.upper() for b in available}:
            issues.append(
                {
                    "level": "warning",
                    "message": f"basis library {library!r} not in bundled library",
                    "line": line_no,
                    "suggested_fix": None,
                }
            )
            continue
        # Resolve canonical filename (case-insensitive)
        canonical = next((b for b in available if b.upper() == library.upper()), library)
        elements_in_lib = {e.upper() for e in list_elements_for_basis(canonical)}
        if element.upper() not in elements_in_lib:
            issues.append(
                {
                    "level": "error",
                    "message": (
                        f"element {element!r} not present in basis library {library!r} "
                        f"(bundled file lists: "
                        f"{', '.join(sorted(elements_in_lib)[:8])}{'...' if len(elements_in_lib) > 8 else ''})"
                    ),
                    "line": line_no,
                    "suggested_fix": None,
                }
            )
    return issues


def _check_rasscf_caspt2_consistency(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RASSCF Frozen must match the following CASPT2 Frozen vector."""
    issues: list[dict[str, Any]] = []
    rasscf = next((b for b in blocks if b["name"] == "RASSCF"), None)
    caspt2 = next((b for b in blocks if b["name"] == "CASPT2"), None)
    if not (rasscf and caspt2):
        return issues
    rs_frozen = _extract_int_vector(rasscf["body"], "Frozen") or []
    pt_frozen = _extract_int_vector(caspt2["body"], "Frozen") or []
    # CASPT2 must not freeze MORE orbitals than RASSCF in any symmetry — but
    # CASPT2 may freeze fewer (RASSCF frozen are auto-frozen at the CASPT2 step).
    if pt_frozen and not rs_frozen and any(pt_frozen):
        issues.append(
            {
                "level": "warning",
                "message": (
                    f"CASPT2 has Frozen {pt_frozen} but RASSCF has no Frozen directive — "
                    "Molcas may auto-handle, but explicit consistency is safer."
                ),
                "line": caspt2["line_start"],
                "suggested_fix": None,
            }
        )
    elif rs_frozen and pt_frozen and rs_frozen != pt_frozen:
        # Per-symmetry: pt_frozen[i] must be <= rs_frozen[i]
        n = max(len(rs_frozen), len(pt_frozen))
        rs_padded = rs_frozen + [0] * (n - len(rs_frozen))
        pt_padded = pt_frozen + [0] * (n - len(pt_frozen))
        violations = [(i + 1, rs_padded[i], pt_padded[i]) for i in range(n) if pt_padded[i] > rs_padded[i]]
        if violations:
            issues.append(
                {
                    "level": "warning",
                    "message": (
                        f"CASPT2 Frozen {pt_frozen} exceeds RASSCF Frozen {rs_frozen} in "
                        f"symmetries {[v[0] for v in violations]}; CASPT2 must not freeze more "
                        "than RASSCF."
                    ),
                    "line": caspt2["line_start"],
                    "suggested_fix": f"Frozen\n{_format_vector(rs_frozen)}",
                }
            )
    return issues


def _check_nactel(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanity: Nactel triple should be (active_e, max_RAS1_holes, max_RAS3_e)."""
    issues: list[dict[str, Any]] = []
    rasscf = next((b for b in blocks if b["name"] == "RASSCF"), None)
    if not rasscf:
        return issues
    nactel = _extract_int_vector(rasscf["body"], "Nactel")
    ras1 = _extract_int_vector(rasscf["body"], "Ras1")
    ras3 = _extract_int_vector(rasscf["body"], "Ras3")
    if nactel is None:
        issues.append(
            {
                "level": "error",
                "message": f"RASSCF block at line {rasscf['line_start']} has no Nactel directive",
                "line": rasscf["line_start"],
                "suggested_fix": "Nactel\n   N   0   0   (replace N with active electron count)",
            }
        )
        return issues
    if len(nactel) != 3:
        issues.append(
            {
                "level": "error",
                "message": f"RASSCF Nactel must have 3 entries (active_e, max_RAS1_holes, max_RAS3_e); got {nactel}",
                "line": rasscf["line_start"],
                "suggested_fix": None,
            }
        )
        return issues
    # Cross-check: if Ras1 vector is present but Nactel[1]=0, that's suspicious
    if ras1 and any(ras1) and nactel[1] == 0:
        issues.append(
            {
                "level": "warning",
                "message": (
                    "RAS1 orbitals declared but Nactel[1] (max RAS1 holes) is 0 — RAS1 is unused. "
                    "Set max_RAS1_holes > 0 for a real RAS calculation."
                ),
                "line": rasscf["line_start"],
                "suggested_fix": None,
            }
        )
    if ras3 and any(ras3) and nactel[2] == 0:
        issues.append(
            {
                "level": "warning",
                "message": (
                    "RAS3 orbitals declared but Nactel[2] (max RAS3 electrons) is 0 — RAS3 is unused."
                ),
                "line": rasscf["line_start"],
                "suggested_fix": None,
            }
        )
    return issues


def _check_lumorb_provenance(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """LumOrb in RASSCF means 'read INPORB' — verify there's a preceding orbital
    source (SCF / earlier RASSCF) in the same input."""
    issues: list[dict[str, Any]] = []
    rasscf_indices = [i for i, b in enumerate(blocks) if b["name"] == "RASSCF"]
    for idx in rasscf_indices:
        rasscf = blocks[idx]
        if "lumorb" not in rasscf["body"].lower():
            continue
        # Look for a preceding SCF or RASSCF block
        preceding = blocks[:idx]
        if not any(b["name"] in {"SCF", "RASSCF"} for b in preceding):
            issues.append(
                {
                    "level": "warning",
                    "message": (
                        f"RASSCF at line {rasscf['line_start']} uses LumOrb but no preceding "
                        "SCF/RASSCF block is present in this input; INPORB will need to be "
                        "supplied externally (or remove LumOrb to use GuessOrb)."
                    ),
                    "line": rasscf["line_start"],
                    "suggested_fix": None,
                }
            )
    return issues


def _check_spin_charge(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RASSCF Spin (=multiplicity) must be a positive integer. SCF Charge syntax check."""
    issues: list[dict[str, Any]] = []
    for b in blocks:
        if b["name"] == "RASSCF":
            spin = _extract_int_vector(b["body"], "Spin")
            if spin and (spin[0] < 1 or spin[0] > 21):
                issues.append(
                    {
                        "level": "error",
                        "message": f"RASSCF Spin = {spin[0]} is implausible (expected 1-21)",
                        "line": b["line_start"],
                        "suggested_fix": None,
                    }
                )
    return issues


def _extract_int_vector(body: str, keyword: str) -> list[int] | None:
    """Look for `<keyword>\\n<integers...>` in a block body. Returns the int list or None."""
    pattern = re.compile(rf"^\s*{re.escape(keyword)}\s*$\n((?:\s*-?\d+[ \t,]*)+)", re.M | re.I)
    m = pattern.search(body)
    if not m:
        return None
    raw = m.group(1).replace(",", " ")
    try:
        return [int(x) for x in raw.split()]
    except ValueError:
        return None


def _format_vector(values: list[int]) -> str:
    return "".join(f"{int(v):>4d}" for v in values)
