from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .basis import extract_nwchem_geometry_elements, normalize_element_symbol
from .common import normalize_path, read_text


CHARGE_RE = re.compile(r"^\s*charge\s+([+-]?\d+)\s*$", re.IGNORECASE)
MULT_RE = re.compile(r"^\s*(?:mult|multiplicity)\s+(\d+)\s*$", re.IGNORECASE)
TASK_RE = re.compile(r"^\s*task\s+([A-Za-z0-9_\-]+)(?:\s+([A-Za-z0-9_\-]+))?", re.IGNORECASE)
SET_GEOMETRY_RE = re.compile(r"^\s*set\s+geometry\s+([A-Za-z0-9_\-]+)", re.IGNORECASE)
GEOMETRY_RE = re.compile(r"^\s*geometry\b", re.IGNORECASE)
MODULE_START_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\b", re.IGNORECASE)
BASIS_RE = re.compile(r"^\s*basis\b", re.IGNORECASE)
ECP_RE = re.compile(r"^\s*ecp\b", re.IGNORECASE)
LIBRARY_LINE_RE = re.compile(r"^\s*([A-Za-z\*][A-Za-z0-9\*]*)\s+library\s+([^\s]+)", re.IGNORECASE)
VECTORS_RE = re.compile(r"^\s*vectors\b", re.IGNORECASE)
START_RE = re.compile(r"^\s*start\b", re.IGNORECASE)
END_RE = re.compile(r"^\s*end\s*$", re.IGNORECASE)
DFT_HDR_RE = re.compile(r"^\s*dft\s*$", re.IGNORECASE)
VECTORS_OUTPUT_RE = re.compile(r"^\s*vectors\s+output\s+(\S+)", re.IGNORECASE)
FRAGMENT_VECTORS_RE = re.compile(r"^\s*vectors\s+input\s+fragment\s+(.+)$", re.IGNORECASE)
DFT_MULT_RE = re.compile(r"^\s*mult\s+(\d+)", re.IGNORECASE)
CARTESIAN_ATOM_RE = re.compile(r"^\s*([A-Za-z]{1,3})\s+[-+]?\d")

TRANSITION_METALS = {
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
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}


def parse_start_blocks(path: str) -> list[dict[str, Any]]:
    """Parse a (possibly multi-start) NWChem input into per-start-block records.

    Each record contains: start_name, elements, charge, multiplicity,
    vectors_output, fragment_inputs, tasks.

    Returns [] for files with no ``start`` directive (single-start files).
    """
    try:
        contents = read_text(path)
    except OSError:
        return []

    lines = contents.splitlines()
    has_start = any(START_RE.match(line) for line in lines)
    if not has_start:
        return []

    blocks: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    in_geometry = False
    in_dft = False

    for line in lines:
        if START_RE.match(line):
            if cur is not None:
                blocks.append(cur)
            tokens = line.split()
            start_name = tokens[1] if len(tokens) > 1 else None
            cur = {
                "start_name": start_name,
                "elements": [],
                "charge": 0,
                "multiplicity": None,
                "vectors_output": None,
                "fragment_inputs": [],
                "tasks": [],
            }
            in_geometry = False
            in_dft = False
            continue

        if cur is None:
            continue

        if GEOMETRY_RE.match(line):
            in_geometry = True
            in_dft = False
            continue
        if DFT_HDR_RE.match(line):
            in_dft = True
            in_geometry = False
            continue
        if END_RE.match(line):
            in_geometry = False
            in_dft = False
            continue

        if in_geometry:
            m = CARTESIAN_ATOM_RE.match(line)
            if m:
                cur["elements"].append(m.group(1).capitalize())
            continue

        if m := CHARGE_RE.match(line):
            cur["charge"] = int(m.group(1))
            continue

        if m := TASK_RE.match(line):
            cur["tasks"].append({"module": m.group(1), "operation": m.group(2)})
            continue

        if in_dft:
            if m := DFT_MULT_RE.match(line):
                cur["multiplicity"] = int(m.group(1))
            if m := VECTORS_OUTPUT_RE.match(line):
                cur["vectors_output"] = m.group(1)
            if m := FRAGMENT_VECTORS_RE.match(line):
                cur["fragment_inputs"] = m.group(1).split()

    if cur is not None:
        blocks.append(cur)

    return blocks


def inspect_nwchem_input(path: str) -> dict[str, Any]:
    contents = read_text(path)
    geometry = extract_nwchem_geometry_elements(path)

    charges: list[int] = []
    multiplicities: list[int] = []
    tasks: list[dict[str, str | None]] = []
    geometry_refs: list[str] = []

    for line in contents.splitlines():
        if match := CHARGE_RE.match(line):
            charges.append(int(match.group(1)))
        if match := MULT_RE.match(line):
            multiplicities.append(int(match.group(1)))
        if match := TASK_RE.match(line):
            tasks.append({"module": match.group(1), "operation": match.group(2)})
        if match := SET_GEOMETRY_RE.match(line):
            geometry_refs.append(match.group(1))

    elements = geometry["elements"]
    transition_metals = [element for element in elements if element in TRANSITION_METALS]

    return {
        "file": normalize_path(path),
        "elements": elements,
        "transition_metals": transition_metals,
        "charge": charges[-1] if charges else None,
        "charges_seen": charges,
        "multiplicity": multiplicities[-1] if multiplicities else None,
        "multiplicities_seen": multiplicities,
        "tasks": tasks,
        "geometry_names_selected": geometry_refs,
        "geometry_block_count": geometry["geometry_block_count"],
        "start_present": any(START_RE.match(line) for line in contents.splitlines()),
        "start_blocks": parse_start_blocks(path),
    }


def load_geometry_source(path: str, block_index: int = 0) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() == ".xyz":
        geometry = _extract_xyz_geometry_block(path)
        geometry["source_kind"] = "xyz"
        return geometry

    geometry = extract_nwchem_geometry_block(path, block_index=block_index)
    geometry["source_kind"] = "nwchem_input"
    return geometry


def extract_nwchem_geometry_block(path: str, block_index: int = 0) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    in_geometry = False
    geometry_count = -1

    start_line = None
    end_line = None
    header_line = None
    directives: list[str] = []
    atoms: list[dict[str, Any]] = []

    for idx, line in enumerate(lines):
        if not in_geometry:
            if GEOMETRY_RE.match(line):
                geometry_count += 1
                if geometry_count == block_index:
                    in_geometry = True
                    start_line = idx
                    header_line = line
                    directives = []
                    atoms = []
            continue

        if re.match(r"^\s*end\s*$", line, re.IGNORECASE):
            end_line = idx
            break

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue

        parts = stripped.split()
        if _looks_like_cartesian_atom(parts):
            atoms.append(
                {
                    "label": parts[0],
                    "element": normalize_element_symbol(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                }
            )
        else:
            directives.append(stripped)

    if start_line is None or end_line is None or header_line is None:
        raise ValueError(f"could not find geometry block {block_index} in {path}")
    if not atoms:
        raise ValueError(f"geometry block {block_index} in {path} does not look like cartesian coordinates")

    return {
        "file": normalize_path(path),
        "block_index": block_index,
        "start_line": start_line,
        "end_line": end_line,
        "header_line": header_line,
        "directives": directives,
        "atoms": atoms,
        "atom_count": len(atoms),
    }


def render_nwchem_geometry_block(
    header_line: str,
    atoms: list[dict[str, Any]],
    directives: list[str] | None = None,
) -> str:
    lines = [header_line.rstrip()]
    for directive in directives or []:
        lines.append(f"  {directive.strip()}")
    for atom in atoms:
        lines.append(
            f"  {atom['label']:<2} {atom['x']: .8f} {atom['y']: .8f} {atom['z']: .8f}"
        )
    lines.append("end")
    return "\n".join(lines)


def replace_nwchem_geometry_block(
    path: str,
    geometry_block_text: str,
    block_index: int = 0,
) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    geometry = extract_nwchem_geometry_block(path, block_index=block_index)
    new_lines = (
        lines[: geometry["start_line"]]
        + geometry_block_text.splitlines()
        + lines[geometry["end_line"] + 1 :]
    )
    return {
        "file": normalize_path(path),
        "block_index": block_index,
        "text": "\n".join(new_lines) + ("\n" if contents.endswith("\n") else ""),
    }


def replace_nwchem_tasks(
    path: str,
    task_lines: list[str],
) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    task_indices = [index for index, line in enumerate(lines) if TASK_RE.match(line)]

    replacement = [task.rstrip() for task in task_lines]
    if task_indices:
        first = task_indices[0]
        remaining = [line for index, line in enumerate(lines) if index not in set(task_indices)]
        insertion_index = first
        new_lines = remaining[:insertion_index] + replacement + remaining[insertion_index:]
    else:
        new_lines = lines + ([""] if lines and lines[-1].strip() else []) + replacement

    return {
        "file": normalize_path(path),
        "task_count_replaced": len(task_indices),
        "task_lines": replacement,
        "text": "\n".join(new_lines) + ("\n" if contents.endswith("\n") else ""),
    }


def extract_nwchem_module_block(path: str, module: str = "dft", block_index: int = -1) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    blocks = _find_module_blocks(lines, module)
    if not blocks:
        raise ValueError(f"could not find {module} block in {path}")

    selected = blocks[block_index]
    return {
        "file": normalize_path(path),
        "module": module.lower(),
        "block_index": block_index,
        "start_line": selected["start_line"],
        "end_line": selected["end_line"],
        "header_line": selected["header_line"],
        "body_lines": selected["body_lines"],
    }


def replace_nwchem_module_block(
    path: str,
    module_block_text: str,
    module: str = "dft",
    block_index: int = -1,
) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    module_block = extract_nwchem_module_block(path, module=module, block_index=block_index)
    new_lines = (
        lines[: module_block["start_line"]]
        + module_block_text.splitlines()
        + lines[module_block["end_line"] + 1 :]
    )
    return {
        "file": normalize_path(path),
        "module": module.lower(),
        "block_index": block_index,
        "text": "\n".join(new_lines) + ("\n" if contents.endswith("\n") else ""),
    }


def render_nwchem_module_block(
    header_line: str,
    body_lines: list[str],
) -> str:
    lines = [header_line.rstrip()]
    lines.extend(line.rstrip() for line in body_lines)
    lines.append("end")
    return "\n".join(lines)


def _parse_basis_block_body(
    header_line: str, body_lines: list[str]
) -> dict[str, Any]:
    """Shared parser for a single basis block body — used by both single and multi-block inspectors."""
    library_assignments: dict[str, str] = {}
    explicit_elements: list[str] = []
    explicit_seen: set[str] = set()
    default_library: str | None = None
    mode: str | None = None

    for token in header_line.split()[1:]:
        lowered = token.strip().strip('"').lower()
        if lowered in {"spherical", "cartesian"}:
            mode = lowered
            break

    for raw_line in body_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if match := LIBRARY_LINE_RE.match(stripped):
            symbol = match.group(1)
            family = match.group(2)
            if symbol == "*":
                default_library = family
            else:
                library_assignments[normalize_element_symbol(symbol)] = family
            continue
        parts = stripped.split()
        if not parts:
            continue
        try:
            symbol = normalize_element_symbol(parts[0])
        except ValueError:
            continue
        if symbol not in explicit_seen:
            explicit_seen.add(symbol)
            explicit_elements.append(symbol)

    return {
        "mode": mode,
        "library_assignments": library_assignments,
        "default_library": default_library,
        "explicit_elements": explicit_elements,
        "has_library_lines": bool(library_assignments or default_library),
        "has_manual_content": bool(explicit_elements),
    }


def inspect_nwchem_basis_block(path: str) -> dict[str, Any]:
    block = _extract_named_block(path, BASIS_RE, "basis")
    return {**block, **_parse_basis_block_body(block["header_line"], block["body_lines"])}


def inspect_all_nwchem_basis_blocks(path: str) -> list[dict[str, Any]]:
    """Return inspection dicts for every basis block in the input file."""
    contents = read_text(path)
    lines = contents.splitlines()
    raw_blocks = _find_module_blocks(lines, "basis")
    results: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_blocks):
        parsed = _parse_basis_block_body(raw["header_line"], raw["body_lines"])
        results.append({
            "file": normalize_path(path),
            "keyword": "basis",
            "block_index": idx,
            "start_line": raw["start_line"],
            "end_line": raw["end_line"],
            "header_line": raw["header_line"],
            "body_lines": raw["body_lines"],
            **parsed,
        })
    return results


_NELEC_RE = re.compile(r"^\s*([A-Za-z]{1,3})\s+nelec\s+(\d+)\s*$", re.IGNORECASE)


def inspect_nwchem_ecp_block(path: str) -> dict[str, Any]:
    block = _extract_named_block(path, ECP_RE, "ecp")
    body_lines = block["body_lines"]
    library_assignments: dict[str, str] = {}
    explicit_elements: list[str] = []
    explicit_seen: set[str] = set()
    default_library: str | None = None
    nelec_by_element: dict[str, int] = {}

    for raw_line in body_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if match := LIBRARY_LINE_RE.match(stripped):
            symbol = match.group(1)
            family = match.group(2)
            if symbol == "*":
                default_library = family
            else:
                library_assignments[normalize_element_symbol(symbol)] = family
            continue
        # Parse inline "Elem nelec N" lines from explicit ECP definitions
        if m := _NELEC_RE.match(stripped):
            try:
                sym = normalize_element_symbol(m.group(1))
                nelec_by_element[sym] = int(m.group(2))
            except ValueError:
                pass
            continue
        parts = stripped.split()
        if not parts:
            continue
        try:
            symbol = normalize_element_symbol(parts[0])
        except ValueError:
            continue
        if symbol not in explicit_seen:
            explicit_seen.add(symbol)
            explicit_elements.append(symbol)

    return {
        **block,
        "library_assignments": library_assignments,
        "default_library": default_library,
        "explicit_elements": explicit_elements,
        "nelec_by_element": nelec_by_element,
        "has_library_lines": bool(library_assignments or default_library),
        "has_manual_content": bool(explicit_elements),
    }


def inspect_nwchem_module_vectors(path: str, module: str = "dft", block_index: int = -1) -> dict[str, Any]:
    module_block = extract_nwchem_module_block(path, module=module, block_index=block_index)
    vectors_lines = [line.strip() for line in module_block["body_lines"] if VECTORS_RE.match(line)]
    has_vectors = bool(vectors_lines)
    has_input = any(" input " in f" {line.lower()} " for line in vectors_lines)
    has_output = any(" output " in f" {line.lower()} " for line in vectors_lines)
    return {
        **module_block,
        "vectors_lines": vectors_lines,
        "has_vectors": has_vectors,
        "has_vectors_input": has_input,
        "has_vectors_output": has_output,
    }


def _extract_xyz_geometry_block(path: str) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} does not look like an XYZ file")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"{path} does not look like an XYZ file") from exc

    atoms: list[dict[str, Any]] = []
    for line in lines[2 : 2 + atom_count]:
        parts = line.split()
        if not _looks_like_cartesian_atom(parts):
            raise ValueError(f"{path} contains an invalid XYZ coordinate line: {line}")
        atoms.append(
            {
                "label": parts[0],
                "element": normalize_element_symbol(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
            }
        )

    if len(atoms) != atom_count:
        raise ValueError(f"{path} is missing XYZ atom lines")

    return {
        "file": normalize_path(path),
        "block_index": 0,
        "start_line": 0,
        "end_line": atom_count + 1,
        "header_line": "geometry units angstroms",
        "directives": [],
        "atoms": atoms,
        "atom_count": len(atoms),
        "comment": lines[1] if len(lines) > 1 else "",
    }


def _looks_like_cartesian_atom(parts: list[str]) -> bool:
    if len(parts) < 4:
        return False
    try:
        normalize_element_symbol(parts[0])
        float(parts[1])
        float(parts[2])
        float(parts[3])
    except (ValueError, TypeError):
        return False
    return True


def _find_module_blocks(lines: list[str], module: str) -> list[dict[str, Any]]:
    target = module.lower()
    blocks: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        match = MODULE_START_RE.match(stripped)
        if match and match.group(1).lower() == target:
            start_line = idx
            header_line = lines[idx]
            body_lines: list[str] = []
            idx += 1
            while idx < len(lines):
                if re.match(r"^\s*end\s*$", lines[idx], re.IGNORECASE):
                    blocks.append(
                        {
                            "start_line": start_line,
                            "end_line": idx,
                            "header_line": header_line,
                            "body_lines": body_lines,
                        }
                    )
                    break
                body_lines.append(lines[idx])
                idx += 1
        idx += 1
    return blocks


def _extract_named_block(path: str, pattern: re.Pattern[str], label: str) -> dict[str, Any]:
    contents = read_text(path)
    lines = contents.splitlines()
    start_line: int | None = None
    header_line: str | None = None
    body_lines: list[str] = []

    for index, line in enumerate(lines):
        if pattern.match(line):
            start_line = index
            header_line = line
            for inner_index in range(index + 1, len(lines)):
                if re.match(r"^\s*end\s*$", lines[inner_index], re.IGNORECASE):
                    return {
                        "file": normalize_path(path),
                        "keyword": label,
                        "start_line": start_line,
                        "end_line": inner_index,
                        "header_line": header_line,
                        "body_lines": body_lines,
                    }
                body_lines.append(lines[inner_index])
            break

    raise ValueError(f"could not find {label} block in {path}")
