"""Parser for Molcas INPORB / RasOrb / ScfOrb / GssOrb / LprOrb / SpdOrb files.

These are human-readable orbital files with section markers. Format spec
(version 2.2, the standard since Molcas 7):

    #INPORB <version>      e.g. #INPORB 2.2
    #INFO
    * <comment>
    <flag> <nSym> <flag>
    <nBas[1]> <nBas[2]> ... <nBas[nSym]>
    <nOrb[1]> <nOrb[2]> ... <nOrb[nSym]>
    [* commentary lines starting with *BC: ...]
    [#EXTRAS                                       (optional)
     * <comment>
     <floats>
    ]
    #ORB
    * ORBITAL <sym>  <orb_index_in_sym>
     <nBas[sym] coefficients, 5 per line, ES22.14>
    [* ORBITAL <sym>  <orb_index_in_sym>           (repeat for each MO)
     ...]
    #OCC
    * OCCUPATION NUMBERS
    <occupations grouped by symmetry, 5 per line>
    [#OCHR                                         (human-readable, 4 decimals)
     ...
    ]
    [#ONE                                          (orbital energies)
     ...
    ]
    #INDEX
    * 1234567890
    0 <10-char typeindex per orbital>              (f / i / 1 / 2 / 3 / s / d)
    1 <10-char typeindex>
    ...                                            (one block per symmetry)

For UHF, the file additionally contains #UORB / #UOCC / #UONE for the beta
spin set, with the same per-symmetry nesting.

The "typeindex" character convention:
    f = frozen
    i = inactive
    1 = RAS1
    2 = RAS2 (the CAS-like full-CI subspace)
    3 = RAS3
    s = secondary
    d = deleted
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_VERSION_RE = re.compile(r"^#INPORB\s+(\S+)")

_ORBITAL_HEADER_RE = re.compile(r"^\*\s+ORBITAL\s+(\d+)\s+(\d+)\s*$")
_TYPEINDEX_LINE_RE = re.compile(r"^(\d+)\s+([fi123sd]+)\s*$")


_TYPEINDEX_NAMES = {
    "f": "frozen",
    "i": "inactive",
    "1": "ras1",
    "2": "ras2",
    "3": "ras3",
    "s": "secondary",
    "d": "deleted",
}


def parse_inporb(path: str | Path, *, parse_coefficients: bool = True) -> dict[str, Any]:
    text = Path(path).read_text()
    lines = text.splitlines()
    version = _parse_version(lines)
    sections = _split_sections(lines)
    info = _parse_info(sections.get("INFO", []))
    n_sym = info["n_symmetries"]
    n_bas = info["basis_functions_per_symmetry"]
    n_orb = info["orbitals_per_symmetry"]

    extras = _parse_extras(sections.get("EXTRAS", []))
    occupations = _parse_floats_grouped_by_symmetry(sections.get("OCC", []), n_orb)
    occupations_human = _parse_floats_grouped_by_symmetry(sections.get("OCHR", []), n_orb)
    energies = _parse_floats_grouped_by_symmetry(sections.get("ONE", []), n_orb)
    type_index = _parse_typeindex(sections.get("INDEX", []), n_orb)
    if parse_coefficients:
        orbitals = _parse_orbitals(sections.get("ORB", []), n_sym, n_orb, n_bas)
    else:
        orbitals = None
    # Beta-spin counterparts
    beta_occupations = _parse_floats_grouped_by_symmetry(sections.get("UOCC", []), n_orb) if "UOCC" in sections else None
    beta_energies = _parse_floats_grouped_by_symmetry(sections.get("UONE", []), n_orb) if "UONE" in sections else None
    if parse_coefficients and "UORB" in sections:
        beta_orbitals = _parse_orbitals(sections["UORB"], n_sym, n_orb, n_bas)
    else:
        beta_orbitals = None

    active_partition = _summarize_active_partition(type_index, occupations)

    return {
        "file": str(path),
        "format": "INPORB",
        "version": version,
        "info": info,
        "extras": extras,
        "alpha": {
            "occupations": occupations,
            "occupations_human": occupations_human,
            "orbital_energies": energies,
            "type_index": type_index,
            "orbital_coefficients": orbitals,
        },
        "beta": (
            {
                "occupations": beta_occupations,
                "orbital_energies": beta_energies,
                "orbital_coefficients": beta_orbitals,
            }
            if (beta_occupations or beta_energies or beta_orbitals)
            else None
        ),
        "is_uhf": "UORB" in sections or "UOCC" in sections,
        "active_space_partition": active_partition,
    }


def _parse_version(lines: list[str]) -> str | None:
    for line in lines[:5]:
        m = _VERSION_RE.match(line)
        if m:
            return m.group(1)
    return None


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Slice the file by `#NAME` markers. Returns body lines per section."""
    sections: dict[str, list[str]] = {}
    current_name: str | None = None
    current_body: list[str] = []
    for line in lines:
        if line.startswith("#") and not line.startswith("#INPORB"):
            if current_name is not None:
                sections[current_name] = current_body
            current_name = line[1:].strip().split()[0].upper()
            current_body = []
        else:
            if current_name is not None:
                current_body.append(line)
    if current_name is not None:
        sections[current_name] = current_body
    return sections


def _parse_info(body: list[str]) -> dict[str, Any]:
    """First non-comment line: 3 flag integers; second: nBas per sym; third: nOrb per sym."""
    numeric_lines = [ln for ln in body if ln.strip() and not ln.startswith("*")]
    flags = [int(x) for x in numeric_lines[0].split()] if len(numeric_lines) > 0 else []
    n_sym = flags[1] if len(flags) >= 2 else 0
    nbas = [int(x) for x in numeric_lines[1].split()] if len(numeric_lines) > 1 else []
    norb = [int(x) for x in numeric_lines[2].split()] if len(numeric_lines) > 2 else []
    title_lines = [ln.lstrip("*").strip() for ln in body if ln.startswith("*") and not ln.startswith("*BC:")]
    return {
        "title": title_lines[0] if title_lines else None,
        "info_flags": flags,
        "n_symmetries": n_sym,
        "basis_functions_per_symmetry": nbas,
        "orbitals_per_symmetry": norb,
        "total_basis_functions": sum(nbas),
        "total_orbitals": sum(norb),
    }


def _parse_extras(body: list[str]) -> dict[str, Any]:
    """The EXTRAS section is loosely structured key=value with comments. Just
    return the raw chunks so callers can interpret them."""
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in body:
        if line.startswith("*"):
            if current is not None:
                items.append(current)
            current = {"label": line.lstrip("*").strip(), "values": []}
        else:
            if current is not None:
                for token in line.split():
                    try:
                        current["values"].append(float(token))
                    except ValueError:
                        pass
    if current is not None:
        items.append(current)
    return {"items": items}


def _parse_floats_grouped_by_symmetry(body: list[str], n_orb: list[int]) -> list[list[float]] | None:
    """Read floats from `body` and split into chunks of n_orb[i] each.

    Skips comment lines (start with '*')."""
    if not body:
        return None
    floats: list[float] = []
    for line in body:
        if line.startswith("*"):
            continue
        for token in line.split():
            try:
                floats.append(float(token))
            except ValueError:
                pass
    chunks: list[list[float]] = []
    cursor = 0
    for n in n_orb:
        chunks.append(floats[cursor : cursor + n])
        cursor += n
    return chunks


def _parse_typeindex(body: list[str], n_orb: list[int]) -> list[list[str]] | None:
    """The INDEX section has one sub-block per symmetry. Each sub-block is a
    set of "<row_idx> <10 chars>" lines. Concatenate the chars to recover the
    per-orbital typeindex string for that symmetry, then trim to n_orb[sym].
    """
    if not body:
        return None
    blocks: list[list[str]] = []
    current_chars: list[str] = []
    for line in body:
        if line.startswith("*"):
            # New sub-block (or initial divider). The "* 1234567890" line
            # marks the start of a new symmetry block.
            if current_chars:
                blocks.append(current_chars)
                current_chars = []
            continue
        m = _TYPEINDEX_LINE_RE.match(line.strip())
        if m:
            current_chars.extend(m.group(2))
    if current_chars:
        blocks.append(current_chars)
    # Trim each block to the declared orbital count
    trimmed: list[list[str]] = []
    for sym_idx, block in enumerate(blocks):
        n = n_orb[sym_idx] if sym_idx < len(n_orb) else len(block)
        trimmed.append(block[:n])
    return trimmed


def _parse_orbitals(
    body: list[str],
    n_sym: int,
    n_orb: list[int],
    n_bas: list[int],
) -> list[list[list[float]]]:
    """Returns coefficients[sym_idx][orb_idx_in_sym] = [coeff_per_basis_function].

    Sym indices are 0-indexed; orbital indices within a symmetry are 0-indexed.
    """
    out: list[list[list[float]]] = [[] for _ in range(n_sym)]
    current_sym: int | None = None
    current_orb_idx: int | None = None
    current_coeffs: list[float] = []
    expected_len = 0

    def _flush():
        nonlocal current_coeffs, current_sym, current_orb_idx
        if current_sym is not None and current_orb_idx is not None and current_coeffs:
            sym0 = current_sym - 1
            if 0 <= sym0 < n_sym:
                # Orbital indices within a symmetry are 1-indexed in the file
                while len(out[sym0]) < current_orb_idx:
                    out[sym0].append([])
                out[sym0][current_orb_idx - 1] = current_coeffs.copy()
        current_coeffs = []

    for line in body:
        m = _ORBITAL_HEADER_RE.match(line)
        if m:
            _flush()
            current_sym = int(m.group(1))
            current_orb_idx = int(m.group(2))
            sym0 = current_sym - 1
            expected_len = n_bas[sym0] if 0 <= sym0 < len(n_bas) else 0
            continue
        for token in line.split():
            try:
                current_coeffs.append(float(token))
            except ValueError:
                pass
            if expected_len and len(current_coeffs) >= expected_len:
                # Stop accumulating once we have a full orbital — but still scan
                # the rest of the line for stray tokens (defensive).
                pass
    _flush()
    return out


def _summarize_active_partition(
    type_index: list[list[str]] | None,
    occupations: list[list[float]] | None,
) -> dict[str, Any] | None:
    """Cross-tabulate per-symmetry typeindex with occupations.

    Returns counts of each orbital class plus the active orbitals' occupation
    numbers — exactly what the agent needs to reason about RASSCF active-space
    quality from the orbital file alone.
    """
    if not type_index:
        return None
    summary: dict[str, Any] = {
        "by_symmetry": [],
        "totals": {name: 0 for name in _TYPEINDEX_NAMES.values()},
        "active_orbital_occupations": [],
    }
    for sym_idx, sym_block in enumerate(type_index):
        sym_counts = {name: 0 for name in _TYPEINDEX_NAMES.values()}
        sym_actives: list[float] = []
        sym_occs = occupations[sym_idx] if (occupations and sym_idx < len(occupations)) else []
        for orb_idx, code in enumerate(sym_block):
            name = _TYPEINDEX_NAMES.get(code)
            if name is None:
                continue
            sym_counts[name] += 1
            summary["totals"][name] += 1
            if code in {"1", "2", "3"} and orb_idx < len(sym_occs):
                sym_actives.append(round(sym_occs[orb_idx], 6))
                summary["active_orbital_occupations"].append(
                    {"symmetry": sym_idx + 1, "orbital_in_sym": orb_idx + 1, "occupation": round(sym_occs[orb_idx], 6), "ras_class": code}
                )
        summary["by_symmetry"].append(
            {
                "symmetry": sym_idx + 1,
                "counts": sym_counts,
                "active_occupations": sym_actives,
            }
        )
    # CAS / RAS signature derived purely from the index
    n_active = (
        summary["totals"]["ras1"]
        + summary["totals"]["ras2"]
        + summary["totals"]["ras3"]
    )
    n_active_e = sum(occ for occ_record in summary["active_orbital_occupations"] for occ in [occ_record["occupation"]])
    if n_active:
        is_ras = bool(summary["totals"]["ras1"] or summary["totals"]["ras3"])
        label = "RAS" if is_ras else "CAS"
        summary["signature"] = f"{label}({round(n_active_e):d},{n_active})"
    return summary


# --- Writer + orbital-swap helper --------------------------------------------

def write_inporb(parsed: dict[str, Any], output_path: str | Path) -> None:
    """Serialize a parsed INPORB structure (as returned by parse_inporb) back
    to disk in Molcas's native version-2.2 format.

    Preserves: title, info flags, nBas/nOrb, extras, occupations (raw + human),
    orbital energies, typeindex (frozen/inactive/RAS/secondary), and full
    per-symmetry MO coefficient blocks.
    """
    info = parsed.get("info") or {}
    alpha = parsed.get("alpha") or {}
    version = parsed.get("version") or "2.2"
    title = info.get("title")
    flags = info.get("info_flags") or [0, len(info.get("basis_functions_per_symmetry") or [1]), 0]
    n_bas = info.get("basis_functions_per_symmetry") or []
    n_orb = info.get("orbitals_per_symmetry") or []
    extras = (parsed.get("extras") or {}).get("items") or []
    occupations = alpha.get("occupations")
    occupations_human = alpha.get("occupations_human")
    energies = alpha.get("orbital_energies")
    type_index = alpha.get("type_index")
    orbital_coeffs = alpha.get("orbital_coefficients")

    lines: list[str] = [f"#INPORB {version}", "#INFO"]
    lines.append(f"* {title}" if title else "* (no title)")
    lines.append(_fmt_int_vector(flags, width=8))
    lines.append(_fmt_int_vector(n_bas, width=8))
    lines.append(_fmt_int_vector(n_orb, width=8))

    if extras:
        lines.append("#EXTRAS")
        for item in extras:
            lines.append(f"* {item.get('label', '')}")
            for v in item.get("values", []):
                lines.append(f" {_fmt_float_es22(v)}")

    if orbital_coeffs is not None:
        lines.append("#ORB")
        for sym_idx, sym_block in enumerate(orbital_coeffs, start=1):
            for orb_idx, coefs in enumerate(sym_block, start=1):
                if not coefs:
                    continue
                lines.append(f"* ORBITAL {sym_idx:>4d}{orb_idx:>5d}")
                lines.extend(_fmt_floats_5_per_line(coefs))

    if occupations is not None:
        lines.append("#OCC")
        lines.append("* OCCUPATION NUMBERS")
        for sym_occs in occupations:
            lines.extend(_fmt_floats_5_per_line(sym_occs))

    if occupations_human is not None:
        lines.append("#OCHR")
        lines.append("* OCCUPATION NUMBERS (HUMAN-READABLE)")
        for sym_occs in occupations_human:
            lines.extend(_fmt_floats_10_per_line_short(sym_occs))

    if energies is not None:
        lines.append("#ONE")
        lines.append("* ONE ELECTRON ENERGIES")
        for sym_energies in energies:
            lines.extend(_fmt_floats_10_per_line(sym_energies))

    if type_index is not None:
        lines.append("#INDEX")
        for sym_typeindex in type_index:
            lines.append("* 1234567890")
            for row_start in range(0, len(sym_typeindex), 10):
                chunk = "".join(sym_typeindex[row_start:row_start + 10])
                row_label = str((row_start // 10) % 10)
                lines.append(f"{row_label} {chunk}")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def swap_orbitals_in_inporb(
    input_path: str | Path,
    output_path: str | Path,
    swaps: list[tuple[int, int]],
    *,
    symmetry: int = 1,
) -> dict[str, Any]:
    """Swap pairs of orbitals within a symmetry block of an INPORB file.

    Parameters
    ----------
    input_path
        Existing INPORB / RasOrb / ScfOrb / etc. file.
    output_path
        Where to write the modified file.
    swaps
        List of (orb_i, orb_j) 1-indexed orbital pairs within the chosen
        symmetry. Each pair swaps coefficients + occupation + energy + typeindex.
    symmetry
        1-indexed symmetry irrep (1 for C1, 1 or 2 for Cs/C2/Ci, etc.). Default 1.

    Returns a dict with the swap summary (validated indices, before/after
    typeindex characters per pair).
    """
    parsed = parse_inporb(input_path, parse_coefficients=True)
    sym_idx = symmetry - 1
    n_bas = parsed["info"]["basis_functions_per_symmetry"]
    if sym_idx < 0 or sym_idx >= len(n_bas):
        raise ValueError(f"symmetry={symmetry} out of range; have {len(n_bas)} irreps")
    n_orb_this_sym = parsed["info"]["orbitals_per_symmetry"][sym_idx]

    summary: list[dict[str, Any]] = []
    coeffs = parsed["alpha"]["orbital_coefficients"]
    occs = parsed["alpha"]["occupations"]
    occs_hr = parsed["alpha"].get("occupations_human")
    energies = parsed["alpha"].get("orbital_energies")
    type_index = parsed["alpha"].get("type_index")

    for (i, j) in swaps:
        if not (1 <= i <= n_orb_this_sym and 1 <= j <= n_orb_this_sym):
            raise ValueError(
                f"swap pair ({i},{j}) out of range; symmetry {symmetry} has {n_orb_this_sym} orbitals"
            )
        if i == j:
            continue
        before_ti = (type_index[sym_idx][i - 1], type_index[sym_idx][j - 1]) if type_index else (None, None)
        before_occ = (occs[sym_idx][i - 1], occs[sym_idx][j - 1]) if occs else (None, None)

        if coeffs is not None:
            coeffs[sym_idx][i - 1], coeffs[sym_idx][j - 1] = coeffs[sym_idx][j - 1], coeffs[sym_idx][i - 1]
        if occs is not None:
            occs[sym_idx][i - 1], occs[sym_idx][j - 1] = occs[sym_idx][j - 1], occs[sym_idx][i - 1]
        if occs_hr is not None:
            occs_hr[sym_idx][i - 1], occs_hr[sym_idx][j - 1] = occs_hr[sym_idx][j - 1], occs_hr[sym_idx][i - 1]
        if energies is not None:
            energies[sym_idx][i - 1], energies[sym_idx][j - 1] = energies[sym_idx][j - 1], energies[sym_idx][i - 1]
        if type_index is not None:
            type_index[sym_idx][i - 1], type_index[sym_idx][j - 1] = type_index[sym_idx][j - 1], type_index[sym_idx][i - 1]

        summary.append(
            {
                "orbital_a": i,
                "orbital_b": j,
                "symmetry": symmetry,
                "typeindex_before": before_ti,
                "typeindex_after": (type_index[sym_idx][i - 1], type_index[sym_idx][j - 1]) if type_index else None,
                "occupation_before": before_occ,
                "occupation_after": (occs[sym_idx][i - 1], occs[sym_idx][j - 1]) if occs else None,
            }
        )

    write_inporb(parsed, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "symmetry": symmetry,
        "swaps": summary,
        "n_swaps": len(summary),
    }


# --- Formatting helpers -------------------------------------------------------

def _fmt_int_vector(values, width: int = 8) -> str:
    return "".join(f"{int(v):>{width}d}" for v in values)


def _fmt_float_es22(value: float) -> str:
    # Molcas writes coefficients/occupations in 1.14E+xx style; many writers use
    # `{: .14E}` which gives 22 chars. Match that for round-trip stability.
    return f" {value: .14E}"


def _fmt_floats_5_per_line(values) -> list[str]:
    out: list[str] = []
    for i in range(0, len(values), 5):
        chunk = values[i:i + 5]
        out.append("".join(_fmt_float_es22(v) for v in chunk))
    return out


def _fmt_floats_10_per_line(values) -> list[str]:
    """For #ONE — orbital energies use a shorter `1.4E+xx` format, 10 per line."""
    out: list[str] = []
    for i in range(0, len(values), 10):
        chunk = values[i:i + 10]
        out.append("".join(f" {v: .4E}" for v in chunk))
    return out


def _fmt_floats_10_per_line_short(values) -> list[str]:
    """For #OCHR — short human-readable occupations 10 per line, e.g. 2.0000."""
    out: list[str] = []
    for i in range(0, len(values), 10):
        chunk = values[i:i + 10]
        out.append("".join(f"  {v:6.4f}" for v in chunk))
    return out


def detect_format(path: str | Path) -> str | None:
    """Return 'INPORB' if the first line begins with '#INPORB', else None."""
    try:
        with open(path, "r") as f:
            first = f.readline()
    except OSError:
        return None
    if first.startswith("#INPORB"):
        return "INPORB"
    return None
