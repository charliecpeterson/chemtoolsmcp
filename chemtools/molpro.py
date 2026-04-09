from __future__ import annotations

from typing import Any

from .common import (
    BASIS_RE,
    CHARGE_RE,
    LABEL_RE,
    METHOD_RE,
    PROGRAM_RE,
    make_metadata,
    parse_scientific_float,
    split_tokens,
)

KNOWN_METHODS = {
    "RHF",
    "UHF",
    "KS",
    "UKS",
    "DFT",
    "MCSCF",
    "CASSCF",
    "RASCI",
    "CASPT2",
    "MRCC",
    "CI",
    "MRCI",
    "MP2",
}

HARTREE_TO_EV = 27.211386245988


def classify_program(program: str) -> str:
    uppercase = program.strip().upper()
    if uppercase.startswith("SEWARD"):
        return "integrals"
    if "KOHN-SHAM" in uppercase or uppercase.startswith("DFT"):
        return "dft"
    if uppercase.startswith("MULTI") or uppercase.startswith("MCSCF"):
        return "multi"
    if uppercase.startswith("OPT") or uppercase.startswith("GEOOPT"):
        return "optimization"
    if "FREQUENCIES" in uppercase:
        return "frequency"
    if uppercase.startswith("HESSIAN"):
        return "hessian"
    if "SCF" in uppercase or "HARTREE-F" in uppercase:
        return "scf"
    if uppercase.startswith("CASPT2") or uppercase.startswith("RS2"):
        return "caspt2"
    if uppercase.startswith("CI") or uppercase.startswith("MRCI"):
        return "ci"
    if uppercase.startswith("MRCC"):
        return "mrcc"
    if uppercase.startswith("MP2"):
        return "mp2"
    if uppercase.startswith("POP"):
        return "population"
    if uppercase.startswith("DMA") or uppercase.startswith("PROPERTIES"):
        return "properties"
    return "other"


def parse_tasks(path: str, contents: str) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    current_task: dict[str, Any] | None = None

    current_basis = None
    current_charge = None
    current_method = None
    method_hints_all: list[str] = []
    current_label = None
    current_point_group = None
    last_line = 0

    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        last_line = line_number
        line = raw_line.rstrip()

        basis_match = BASIS_RE.match(line)
        if basis_match:
            current_basis = basis_match.group(1).strip()

        charge_match = CHARGE_RE.match(line)
        if charge_match:
            current_charge = charge_match.group(1).strip()

        label_match = LABEL_RE.match(line)
        if label_match:
            current_label = label_match.group(1).strip()

        if "Point group" in line:
            current_point_group = line.rsplit(" ", 1)[-1].strip() or current_point_group

        method_match = METHOD_RE.match(line)
        if method_match:
            candidate = method_match.group(1).strip().upper()
            if candidate in KNOWN_METHODS:
                if current_method is None:
                    current_method = candidate
                if candidate not in method_hints_all:
                    method_hints_all.append(candidate)

        program_match = PROGRAM_RE.match(line)
        if program_match:
            program = program_match.group(1).strip()
            kind = classify_program(program)

            if current_task is not None:
                current_task["line_end"] = line_number - 1
                redundant_hessian = current_task["kind"] == "hessian" and kind == "frequency"
                if not redundant_hessian:
                    tasks.append(current_task)

            if kind == "integrals":
                current_task = None
                continue

            current_task = {
                "kind": kind,
                "program": program,
                "point_group": current_point_group,
                "basis": current_basis,
                "charge": current_charge,
                "method_hint": current_method,
                "method_hints_all": list(method_hints_all),
                "label": current_label,
                "line_start": line_number,
                "line_end": line_number,
                "total_energy_hartree": None,
            }

    if current_task is not None:
        current_task["line_end"] = max(last_line, current_task["line_start"])
        tasks.append(current_task)

    generic_tasks = []
    for task in tasks:
        generic_kind = {
            "optimization": "optimization",
            "frequency": "frequency",
            "scf": "scf",
            "dft": "scf",
            "mp2": "single_point",
            "caspt2": "single_point",
            "ci": "single_point",
            "mrcc": "single_point",
        }.get(task["kind"], "other")
        label = task["kind"].replace("_", " ").title()
        if task["method_hint"]:
            label = f"{label} · {task['method_hint']}"
        generic_tasks.append(
            {
                "program": "molpro",
                "kind": generic_kind,
                "label": label,
                "energy_hartree": task["total_energy_hartree"],
                "line_start": task["line_start"],
                "line_end": task["line_end"],
                "extra": {
                    "program": task["program"],
                    "basis": task["basis"],
                    "charge": task["charge"],
                    "point_group": task["point_group"],
                    "method_hint": task["method_hint"],
                    "label": task["label"],
                },
            }
        )

    return {
        "metadata": make_metadata(path, contents, "molpro"),
        "generic_tasks": generic_tasks,
        "program_summary": {
            "kind": "molpro",
            "task_count": len(tasks),
            "raw": {"tasks": tasks},
        },
    }


def parse_mos(path: str, contents: str, top_n: int = 5, include_coefficients: bool = False) -> dict[str, Any]:
    lines = contents.splitlines()
    alpha_occ = _find_occupancy(lines, "Final alpha occupancy")
    beta_occ = _find_occupancy(lines, "Final beta occupancy")

    alpha_mos = _parse_spin_block(lines, "Orbital energies for positive spin", alpha_occ, "alpha")
    alpha_count = len(alpha_mos)
    beta_mos = _parse_spin_block(lines, "Orbital energies for negative spin", beta_occ, "beta")
    beta_count = len(beta_mos)

    if not alpha_mos and not beta_mos:
        alpha_mos, beta_mos = _parse_electron_orbital_sections(lines)

    orbitals = alpha_mos + beta_mos
    for index, orbital in enumerate(orbitals, start=1):
        orbital["vector_number"] = index

    occupied = [orbital for orbital in orbitals if orbital["occupancy"] > 0.1]
    virtual = [orbital for orbital in orbitals if orbital["occupancy"] <= 0.1]
    homo = max(occupied, key=lambda item: item["energy_hartree"]) if occupied else None
    lumo = min(virtual, key=lambda item: item["energy_hartree"]) if virtual else None
    gap = None
    gap_ev = None
    if homo is not None and lumo is not None:
        gap = lumo["energy_hartree"] - homo["energy_hartree"]
        gap_ev = gap * HARTREE_TO_EV

    compact_orbitals = []
    for orbital in orbitals:
        ranked = sorted(orbital["coefficients"], key=lambda item: abs(item["coefficient"]), reverse=True)
        compact_orbitals.append(
            {
                "vector_number": orbital["vector_number"],
                "occupancy": orbital["occupancy"],
                "energy_hartree": orbital["energy_hartree"],
                "symmetry": orbital["symmetry"],
                "mo_center_angstrom": None,
                "r_squared": None,
                "top_contributors": ranked[:top_n],
                "coefficients": orbital["coefficients"] if include_coefficients else None,
            }
        )

    return {
        "metadata": make_metadata(path, contents, "molpro"),
        "orbital_count": len(compact_orbitals),
        "occupied_count": len(occupied),
        "virtual_count": len(virtual),
        "homo": _frontier(homo),
        "lumo": _frontier(lumo),
        "homo_lumo_gap_hartree": gap,
        "homo_lumo_gap_ev": gap_ev,
        "populations": {
            "mulliken": "MULLIKEN POPULATIONS" in contents.upper(),
            "lowdin": False,
            "natural": False,
        },
        "orbitals": compact_orbitals,
        "extras": {
            "alpha_occupancy": alpha_occ,
            "beta_occupancy": beta_occ,
            "alpha_orbital_count": alpha_count,
            "beta_orbital_count": beta_count,
        },
    }


def _find_occupancy(lines: list[str], marker: str) -> int | None:
    for line in lines:
        if marker in line:
            try:
                return int(line.split(":", 1)[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def _parse_spin_block(
    lines: list[str], header: str, occupancy: int | None, spin_label: str
) -> list[dict[str, Any]]:
    try:
        start_idx = next(index for index, line in enumerate(lines) if header in line)
    except StopIteration:
        return []

    idx = start_idx + 1
    pending_labels: list[str] | None = None
    entries: list[tuple[float, str | None]] = []

    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed:
            idx += 1
            continue
        if trimmed.startswith("Orbital energies") and idx != start_idx:
            break
        if (
            trimmed.startswith("HOMO")
            or trimmed.startswith("LUMO")
            or trimmed.startswith("Orbitals saved")
            or trimmed.startswith("PROGRAM")
        ):
            break
        if trimmed.startswith("-") or trimmed.startswith("+"):
            values = [parse_scientific_float(token) for token in trimmed.split()]
            numeric_values = [value for value in values if value is not None]
            if pending_labels:
                for index2, value in enumerate(numeric_values):
                    label = pending_labels[index2] if index2 < len(pending_labels) else None
                    entries.append((value, label))
                pending_labels = None
            else:
                for value in numeric_values:
                    entries.append((value, None))
        else:
            tokens = [token for token in trimmed.split() if token]
            if tokens:
                pending_labels = tokens
        idx += 1

    if not entries:
        return []

    inferred_occupancy = min(occupancy, len(entries)) if occupancy is not None else sum(
        1 for energy, _ in entries if energy < 0.0
    )
    orbitals = []
    for index2, (energy, label) in enumerate(entries):
        orbitals.append(
            {
                "vector_number": index2 + 1,
                "occupancy": 1.0 if index2 < inferred_occupancy else 0.0,
                "energy_hartree": energy,
                "symmetry": f"{spin_label} {label}" if label else None,
                "coefficients": [],
            }
        )
    return orbitals


def _parse_electron_orbital_sections(lines: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alpha = _parse_electron_orbital_table(lines, "ELECTRON ORBITALS FOR POSITIVE SPIN", "alpha")
    beta = _parse_electron_orbital_table(lines, "ELECTRON ORBITALS FOR NEGATIVE SPIN", "beta")
    if not alpha and not beta:
        alpha = _parse_electron_orbital_table(lines, "ELECTRON ORBITALS", None)
    return alpha, beta


def _parse_electron_orbital_table(
    lines: list[str], header: str, spin_label: str | None
) -> list[dict[str, Any]]:
    header_upper = header.upper()
    start_idx = None
    for index, line in enumerate(lines):
        if line.strip().upper().startswith(header_upper):
            start_idx = index
            break
    if start_idx is None:
        return []

    idx = start_idx + 1
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if trimmed.startswith("Orbital") and "Occupation" in trimmed:
            idx += 1
            break
        idx += 1

    orbitals = []
    current_orbital = None
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed:
            idx += 1
            continue
        upper = trimmed.upper()
        if (
            upper.startswith("====")
            or upper.startswith("PROGRAM")
            or upper.startswith("HOMO")
            or upper.startswith("LUMO")
            or upper.startswith("ORBITALS SAVED")
            or upper.startswith("DATASETS")
            or upper.startswith("NATURAL")
            or upper.startswith("MOLPRO")
        ):
            break
        if trimmed.startswith("Orbital"):
            idx += 1
            continue

        tokens = split_tokens(trimmed)
        if not tokens:
            idx += 1
            continue

        if _looks_like_orbital_header(tokens):
            if current_orbital is not None:
                orbitals.append(current_orbital)
            current_orbital = _parse_orbital_header(tokens, spin_label)
            if current_orbital is None:
                break
        elif current_orbital is not None:
            if not _append_coefficients(tokens, 0, current_orbital):
                break
        else:
            break
        idx += 1

    if current_orbital is not None:
        orbitals.append(current_orbital)
    return orbitals


def _parse_orbital_header(tokens: list[str], spin_label: str | None) -> dict[str, Any] | None:
    if len(tokens) < 3 or not _looks_like_label(tokens[0]):
        return None
    occupancy = parse_scientific_float(tokens[1])
    energy = parse_scientific_float(tokens[2])
    if occupancy is None or energy is None:
        return None
    orbital = {
        "vector_number": 0,
        "occupancy": occupancy,
        "energy_hartree": energy,
        "symmetry": f"{spin_label} {tokens[0]}" if spin_label else tokens[0],
        "coefficients": [],
    }
    _append_coefficients(tokens, 3, orbital)
    return orbital


def _append_coefficients(tokens: list[str], start: int, orbital: dict[str, Any]) -> bool:
    cursor = start
    appended = False
    while cursor + 3 < len(tokens):
        coefficient = parse_scientific_float(tokens[cursor + 3])
        if coefficient is None:
            break
        orbital["coefficients"].append(
            {
                "basis_function_number": len(orbital["coefficients"]) + 1,
                "coefficient": coefficient,
                "label": f"{tokens[cursor]} {tokens[cursor + 1]} {tokens[cursor + 2]}",
            }
        )
        appended = True
        cursor += 4
    return appended


def _looks_like_orbital_header(tokens: list[str]) -> bool:
    return len(tokens) >= 3 and _looks_like_label(tokens[0])


def _looks_like_label(token: str) -> bool:
    if "." not in token:
        return False
    left, right = token.split(".", 1)
    return bool(left) and left.isdigit() and bool(right) and right.isalnum()


def _frontier(orbital: dict[str, Any] | None) -> dict[str, Any] | None:
    if orbital is None:
        return None
    return {
        "vector_number": orbital["vector_number"],
        "occupancy": orbital["occupancy"],
        "energy_hartree": orbital["energy_hartree"],
        "symmetry": orbital["symmetry"],
    }

