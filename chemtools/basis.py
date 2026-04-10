from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import normalize_path, read_text


PERIODIC_SYMBOLS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}

BLOCK_START_RE = re.compile(r'^\s*basis\s+"([^"]+)"\s+([A-Za-z]+)', re.IGNORECASE)
ECP_START_RE = re.compile(r'^\s*ecp\s+"([^"]+)"', re.IGNORECASE)
GEOMETRY_START_RE = re.compile(r"^\s*geometry\b", re.IGNORECASE)
END_RE = re.compile(r"^\s*end\s*$", re.IGNORECASE)
NELEC_RE = re.compile(r"^\s*([A-Za-z]{1,3})\s+nelec\s+(\d+)\s*$", re.IGNORECASE)


def normalize_element_symbol(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", symbol.strip())
    if not cleaned:
        raise ValueError("empty element symbol")
    normalized = cleaned[0].upper() + cleaned[1:].lower()
    if normalized not in PERIODIC_SYMBOLS:
        raise ValueError(f"unsupported element symbol: {symbol}")
    return normalized


def _scan_basis_library(library_path: str | Path) -> dict[str, Path]:
    root = Path(library_path)
    if not root.is_dir():
        raise ValueError(f"basis library path does not exist or is not a directory: {library_path}")
    entries: dict[str, Path] = {}
    for path in root.iterdir():
        if path.is_file():
            entries[path.name.casefold()] = path
    return entries


def _find_basis_file(basis_name: str, library_path: str | Path) -> Path:
    entries = _scan_basis_library(library_path)
    key = basis_name.casefold()
    if key in entries:
        return entries[key]

    normalized = key.replace('"', "")
    for candidate_key, candidate_path in entries.items():
        if candidate_key.replace('"', "") == normalized:
            return candidate_path

    # Try common aliases: * -> s, ** -> ss (e.g. 6-31g* -> 6-31gs)
    alias = normalized.replace("**", "ss").replace("*", "s")
    if alias != normalized:
        if alias in entries:
            return entries[alias]
        for candidate_key, candidate_path in entries.items():
            if candidate_key.replace('"', "") == alias:
                return candidate_path

    # Suggest close matches to help the caller
    import difflib
    close = difflib.get_close_matches(normalized, [k.replace('"', "") for k in entries], n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"basis set '{basis_name}' was not found in {library_path}.{hint}")


def _parse_basis_blocks(contents: str) -> tuple[dict[str, str], str | None]:
    lines = contents.splitlines()
    blocks: dict[str, str] = {}
    first_mode: str | None = None
    index = 0

    while index < len(lines):
        match = BLOCK_START_RE.match(lines[index])
        if not match:
            index += 1
            continue

        label = match.group(1)
        mode = match.group(2).lower()
        if first_mode is None:
            first_mode = mode
        block_lines = [lines[index]]
        index += 1
        while index < len(lines):
            block_lines.append(lines[index])
            if END_RE.match(lines[index]):
                index += 1
                break
            index += 1

        element = label.split("_", 1)[0].strip()
        try:
            normalized = normalize_element_symbol(element)
        except ValueError:
            continue
        blocks[normalized] = "\n".join(block_lines)

    return blocks, first_mode


def _parse_ecp_blocks(contents: str) -> dict[str, dict[str, Any]]:
    lines = contents.splitlines()
    blocks: dict[str, dict[str, Any]] = {}
    index = 0

    while index < len(lines):
        match = ECP_START_RE.match(lines[index])
        if not match:
            index += 1
            continue

        label = match.group(1)
        block_lines = [lines[index]]
        block_nelec: int | None = None
        index += 1
        while index < len(lines):
            current = lines[index]
            block_lines.append(current)
            if nelec_match := NELEC_RE.match(current):
                try:
                    block_nelec = int(nelec_match.group(2))
                except ValueError:
                    block_nelec = None
            if END_RE.match(current):
                index += 1
                break
            index += 1

        element = label.split("_", 1)[0].strip()
        try:
            normalized = normalize_element_symbol(element)
        except ValueError:
            continue
        blocks[normalized] = {
            "text": "\n".join(block_lines),
            "label": label,
            "nelec": block_nelec,
        }

    return blocks


def _normalize_assignment_map(assignments: dict[str, str] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for symbol, family in (assignments or {}).items():
        if not str(family).strip():
            continue
        normalized[normalize_element_symbol(symbol)] = str(family).strip()
    return normalized


def list_basis_sets(library_path: str | Path) -> dict[str, Any]:
    entries = _scan_basis_library(library_path)
    return {
        "library_path": normalize_path(library_path),
        "basis_sets": sorted(path.name for path in entries.values()),
        "count": len(entries),
    }


def resolve_basis_set(basis_name: str, elements: list[str], library_path: str | Path) -> dict[str, Any]:
    requested = [normalize_element_symbol(element) for element in elements]
    basis_path = _find_basis_file(basis_name, library_path)
    contents = read_text(basis_path)
    blocks, mode = _parse_basis_blocks(contents)

    available = sorted(blocks)
    covered = [element for element in requested if element in blocks]
    missing = [element for element in requested if element not in blocks]

    return {
        "basis_name": basis_path.name,
        "requested_basis_name": basis_name,
        "file": normalize_path(basis_path),
        "requested_elements": requested,
        "available_elements": available,
        "covered_elements": covered,
        "missing_elements": missing,
        "all_elements_covered": not missing,
        "mode": mode,
    }


def extract_basis_blocks(basis_name: str, elements: list[str], library_path: str | Path) -> dict[str, Any]:
    resolved = resolve_basis_set(basis_name, elements, library_path)
    contents = read_text(resolved["file"])
    blocks, _ = _parse_basis_blocks(contents)
    extracted = {element: blocks[element] for element in resolved["covered_elements"]}
    return {
        **resolved,
        "blocks": extracted,
    }


def _extract_block_body(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) <= 2:
        return []
    return lines[1:-1]


def resolve_ecp_set(ecp_name: str, elements: list[str], library_path: str | Path) -> dict[str, Any]:
    requested = [normalize_element_symbol(element) for element in elements]
    ecp_path = _find_basis_file(ecp_name, library_path)
    contents = read_text(ecp_path)
    blocks = _parse_ecp_blocks(contents)

    available = sorted(blocks)
    covered = [element for element in requested if element in blocks]
    missing = [element for element in requested if element not in blocks]

    return {
        "ecp_name": ecp_path.name,
        "requested_ecp_name": ecp_name,
        "file": normalize_path(ecp_path),
        "requested_elements": requested,
        "available_elements": available,
        "covered_elements": covered,
        "missing_elements": missing,
        "all_elements_covered": not missing,
        "nelec_by_element": {element: blocks[element]["nelec"] for element in covered},
    }


def extract_ecp_blocks(ecp_name: str, elements: list[str], library_path: str | Path) -> dict[str, Any]:
    resolved = resolve_ecp_set(ecp_name, elements, library_path)
    contents = read_text(resolved["file"])
    blocks = _parse_ecp_blocks(contents)
    extracted = {element: blocks[element]["text"] for element in resolved["covered_elements"]}
    return {
        **resolved,
        "blocks": extracted,
    }


def render_nwchem_basis_block(
    basis_name: str,
    elements: list[str],
    library_path: str | Path,
    block_name: str = "ao basis",
    mode: str | None = None,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    resolved = resolve_basis_set(basis_name, elements, library_path)
    resolved_mode = (mode or resolved["mode"] or "spherical").lower()
    lines = [f'basis "{block_name}" {resolved_mode}']
    if inline_blocks:
        extracted = extract_basis_blocks(basis_name, elements, library_path)
        for element in resolved["covered_elements"]:
            lines.extend(_extract_block_body(extracted["blocks"][element]))
    else:
        for element in resolved["covered_elements"]:
            lines.append(f"  {element:<2} library {resolved['basis_name']}")
    lines.append("end")
    return {
        **resolved,
        "block_name": block_name,
        "selected_mode": resolved_mode,
        "inline_blocks": inline_blocks,
        "text": "\n".join(lines),
    }


def render_nwchem_ecp_block(
    ecp_name: str,
    elements: list[str],
    library_path: str | Path,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    resolved = resolve_ecp_set(ecp_name, elements, library_path)
    lines = ["ecp"]
    if inline_blocks:
        extracted = extract_ecp_blocks(ecp_name, elements, library_path)
        for element in resolved["covered_elements"]:
            lines.extend(_extract_block_body(extracted["blocks"][element]))
    else:
        for element in resolved["covered_elements"]:
            lines.append(f"  {element:<2} library {resolved['ecp_name']}")
    lines.append("end")
    return {
        **resolved,
        "inline_blocks": inline_blocks,
        "text": "\n".join(lines),
    }


def resolve_mixed_basis_assignments(
    assignments: dict[str, str],
    elements: list[str],
    library_path: str | Path,
    default_basis: str | None = None,
) -> dict[str, Any]:
    requested = [normalize_element_symbol(element) for element in elements]
    normalized_assignments = _normalize_assignment_map(assignments)
    assignment_map: dict[str, str] = {}
    missing_assignments: list[str] = []

    for element in requested:
        assigned = normalized_assignments.get(element) or default_basis
        if not assigned:
            missing_assignments.append(element)
            continue
        assignment_map[element] = assigned

    grouped: dict[str, list[str]] = {}
    for element in requested:
        family = assignment_map.get(element)
        if family is None:
            continue
        grouped.setdefault(family, []).append(element)

    resolved_groups: list[dict[str, Any]] = []
    mode_candidates: set[str] = set()
    missing_coverage: list[str] = []
    for family, family_elements in grouped.items():
        resolved = resolve_basis_set(family, family_elements, library_path)
        resolved_groups.append(resolved)
        if resolved["mode"]:
            mode_candidates.add(str(resolved["mode"]).lower())
        missing_coverage.extend(resolved["missing_elements"])

    return {
        "requested_elements": requested,
        "assignments": assignment_map,
        "missing_assignments": missing_assignments,
        "missing_coverage": missing_coverage,
        "all_elements_covered": not missing_assignments and not missing_coverage,
        "resolved_groups": resolved_groups,
        "mode_candidates": sorted(mode_candidates),
    }


def render_mixed_nwchem_basis_block(
    assignments: dict[str, str],
    elements: list[str],
    library_path: str | Path,
    block_name: str = "ao basis",
    default_basis: str | None = None,
    mode: str | None = None,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    resolved = resolve_mixed_basis_assignments(
        assignments=assignments,
        elements=elements,
        library_path=library_path,
        default_basis=default_basis,
    )
    if resolved["missing_assignments"]:
        raise ValueError(f"basis assignments are missing for elements: {', '.join(resolved['missing_assignments'])}")
    if resolved["missing_coverage"]:
        raise ValueError(f"basis library coverage is missing for elements: {', '.join(resolved['missing_coverage'])}")

    mode_candidates = resolved["mode_candidates"]
    if mode:
        selected_mode = mode.lower()
    elif not mode_candidates:
        selected_mode = "spherical"
    elif len(mode_candidates) == 1:
        selected_mode = mode_candidates[0]
    else:
        raise ValueError(
            "mixed basis assignments use families with incompatible basis modes; pass an explicit mode override"
        )

    lines = [f'basis "{block_name}" {selected_mode}']
    if inline_blocks:
        block_lookup: dict[str, list[str]] = {}
        for group in resolved["resolved_groups"]:
            extracted = extract_basis_blocks(
                group["basis_name"],
                group["requested_elements"],
                library_path,
            )
            for element, block_text in extracted["blocks"].items():
                block_lookup[element] = _extract_block_body(block_text)
        for element in resolved["requested_elements"]:
            lines.extend(block_lookup[element])
    else:
        for element in resolved["requested_elements"]:
            lines.append(f"  {element:<2} library {resolved['assignments'][element]}")
    lines.append("end")
    return {
        **resolved,
        "block_name": block_name,
        "selected_mode": selected_mode,
        "inline_blocks": inline_blocks,
        "text": "\n".join(lines),
    }


def resolve_mixed_ecp_assignments(
    assignments: dict[str, str],
    elements: list[str],
    library_path: str | Path,
    default_ecp: str | None = None,
) -> dict[str, Any]:
    requested = [normalize_element_symbol(element) for element in elements]
    normalized_assignments = _normalize_assignment_map(assignments)
    assignment_map: dict[str, str] = {}

    for element in requested:
        assigned = normalized_assignments.get(element) or default_ecp
        if assigned:
            assignment_map[element] = assigned

    grouped: dict[str, list[str]] = {}
    for element in requested:
        family = assignment_map.get(element)
        if family is None:
            continue
        grouped.setdefault(family, []).append(element)

    resolved_groups: list[dict[str, Any]] = []
    missing_coverage: list[str] = []
    for family, family_elements in grouped.items():
        resolved = resolve_ecp_set(family, family_elements, library_path)
        resolved_groups.append(resolved)
        missing_coverage.extend(resolved["missing_elements"])

    return {
        "requested_elements": requested,
        "assignments": assignment_map,
        "elements_with_ecp": [element for element in requested if element in assignment_map],
        "missing_coverage": missing_coverage,
        "all_elements_covered": not missing_coverage,
        "resolved_groups": resolved_groups,
    }


def render_mixed_nwchem_ecp_block(
    assignments: dict[str, str],
    elements: list[str],
    library_path: str | Path,
    default_ecp: str | None = None,
    inline_blocks: bool = True,
) -> dict[str, Any] | None:
    resolved = resolve_mixed_ecp_assignments(
        assignments=assignments,
        elements=elements,
        library_path=library_path,
        default_ecp=default_ecp,
    )
    if not resolved["elements_with_ecp"]:
        return None
    if resolved["missing_coverage"]:
        raise ValueError(f"ecp library coverage is missing for elements: {', '.join(resolved['missing_coverage'])}")

    lines = ["ecp"]
    if inline_blocks:
        block_lookup: dict[str, list[str]] = {}
        for group in resolved["resolved_groups"]:
            extracted = extract_ecp_blocks(
                group["ecp_name"],
                group["requested_elements"],
                library_path,
            )
            for element, block_text in extracted["blocks"].items():
                block_lookup[element] = _extract_block_body(block_text)
        for element in resolved["elements_with_ecp"]:
            lines.extend(block_lookup[element])
    else:
        for element in resolved["elements_with_ecp"]:
            lines.append(f"  {element:<2} library {resolved['assignments'][element]}")
    lines.append("end")
    return {
        **resolved,
        "inline_blocks": inline_blocks,
        "text": "\n".join(lines),
    }


def extract_nwchem_geometry_elements(path: str | Path) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    in_geometry = False
    elements: list[str] = []
    seen: set[str] = set()
    geometry_blocks = 0

    for line in lines:
        if not in_geometry:
            if GEOMETRY_START_RE.match(line):
                in_geometry = True
                geometry_blocks += 1
            continue

        if END_RE.match(line):
            in_geometry = False
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        head = stripped.split()[0]
        try:
            symbol = normalize_element_symbol(head)
        except ValueError:
            continue
        if symbol not in seen:
            seen.add(symbol)
            elements.append(symbol)

    return {
        "file": normalize_path(path),
        "elements": elements,
        "element_count": len(elements),
        "geometry_block_count": geometry_blocks,
    }


def render_nwchem_basis_block_from_geometry(
    basis_name: str,
    input_path: str | Path,
    library_path: str | Path,
    block_name: str = "ao basis",
    mode: str | None = None,
    inline_blocks: bool = True,
) -> dict[str, Any]:
    geometry = extract_nwchem_geometry_elements(input_path)
    rendered = render_nwchem_basis_block(
        basis_name=basis_name,
        elements=geometry["elements"],
        library_path=library_path,
        block_name=block_name,
        mode=mode,
        inline_blocks=inline_blocks,
    )
    return {
        **rendered,
        "geometry_file": geometry["file"],
    }
