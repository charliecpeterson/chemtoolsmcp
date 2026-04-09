from __future__ import annotations

from collections import defaultdict, deque
import re
from typing import Any

from .common import make_metadata, parse_float_after_delimiter, parse_scientific_float, split_tokens

METHOD_PATTERNS: list[tuple[int, str, tuple[str, ...]]] = [
    (5, "CCSD(T)", ("ccsd(t)",)),
    (4, "CCSD", ("ccsd total energy", " ccsd ")),
    (3, "MP2", ("total mp2 energy", " mp2 ")),
    (3, "MCSCF", ("total mcscf energy", " mcscf ")),
    (2, "DFT", ("total dft energy", " dft ", "b3lyp", "pbe0", "pbe ")),
    (1, "SCF", ("total scf energy", " scf ", " rhf", " uhf", " rohf")),
    (0, "TCE", ("tce",)),
]

BOHR_TO_ANGSTROM = 0.529177210903
FREQUENCY_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([-\d.DEde+]+)\s+\|\|\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s*$"
)
THERMO_AU_RE = re.compile(r"\(\s*([-\d.DEde+]+)\s+au\s*\)", re.IGNORECASE)
OPT_PROGRESS_RE = re.compile(
    r"^\@\s+(\d+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.DEde+]+)\s+([-\d.]+)\s*$"
)
POPULATION_HEADER_RE = re.compile(
    r"^\s*(Total Density|Spin Density)\s*-\s*(Mulliken|Lowdin|L[öo]wdin)\s+Population Analysis\s*$",
    re.IGNORECASE,
)
MCSCF_ENERGY_RE = re.compile(r">>>\|\s*MCSCF energy:\s*([-\d.DEde+]+)")
MCSCF_TOTAL_ENERGY_RE = re.compile(r"Total MCSCF energy\s*=\s*([-\d.DEde+]+)", re.IGNORECASE)
MCSCF_LEVELSHIFT_RE = re.compile(r"Increase level shift to\s+([-\d.DEde+]+)", re.IGNORECASE)
MCSCF_RESIDUE_RE = re.compile(
    r"Precondition failed to converge:Residue:\s*current=\s*([+-]?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)\s*required=\s*([+-]?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)",
    re.IGNORECASE,
)
MCSCF_NEGATIVE_CURVATURE_RE = re.compile(r"Negative curvature:\s*hessian=\s*([-\d.DEde+]+)", re.IGNORECASE)
MCSCF_SETTING_RE = re.compile(
    r"^\s*(active|actelec|multiplicity|state|hessian|maxiter|thresh|tol2e|level|symmetry)\s+(.+?)\s*$",
    re.IGNORECASE,
)
MCSCF_SUMMARY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z\s]+?):\s+(.+?)\s*$")
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
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
}
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.85,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Mo": 1.54,
    "W": 1.62,
    "Re": 1.51,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "Pt": 1.36,
    "Au": 1.36,
    "U": 1.96,
}


def _frontier_orbital_window(
    orbitals: list[dict],
    homo: dict | None,
    lumo: dict | None,
    somos: list[dict],
    window: int = 5,
) -> list[dict]:
    """Return only the frontier orbital window: SOMOs + window orbitals around HOMO/LUMO.

    This keeps the payload small for large systems while preserving the orbitals
    an agent actually needs to inspect (frontier region, SOMOs, and the nearby
    occupied/virtual orbitals relevant for freeze-count decisions).
    """
    indices: set[int] = set()

    # Always include SOMOs
    for orb in somos:
        indices.add(orb["vector_number"])

    # Window around HOMO
    if homo is not None:
        homo_vec = homo["vector_number"]
        homo_spin = homo["spin"]
        spin_orbs = [o for o in orbitals if o["spin"] == homo_spin]
        spin_orbs_sorted = sorted(spin_orbs, key=lambda o: o["vector_number"])
        try:
            pos = next(i for i, o in enumerate(spin_orbs_sorted) if o["vector_number"] == homo_vec)
            for o in spin_orbs_sorted[max(0, pos - window): pos + window + 1]:
                indices.add(o["vector_number"])
        except StopIteration:
            pass

    # Window around LUMO
    if lumo is not None:
        lumo_vec = lumo["vector_number"]
        lumo_spin = lumo["spin"]
        spin_orbs = [o for o in orbitals if o["spin"] == lumo_spin]
        spin_orbs_sorted = sorted(spin_orbs, key=lambda o: o["vector_number"])
        try:
            pos = next(i for i, o in enumerate(spin_orbs_sorted) if o["vector_number"] == lumo_vec)
            for o in spin_orbs_sorted[max(0, pos - window): pos + window + 1]:
                indices.add(o["vector_number"])
        except StopIteration:
            pass

    return [o for o in orbitals if o["vector_number"] in indices]


def parse_mos(path: str, contents: str, top_n: int = 5, include_coefficients: bool = False, include_all_orbitals: bool = False) -> dict[str, Any]:
    lines = contents.splitlines()
    latest_sections: dict[str, list[dict[str, Any]]] = {}
    idx = 0
    headers = (
        "Final Molecular Orbital Analysis",
        "Final Alpha Molecular Orbital Analysis",
        "Final Beta Molecular Orbital Analysis",
    )

    while idx < len(lines):
        if any(header in lines[idx] for header in headers):
            spin = None
            if "Final Alpha Molecular Orbital Analysis" in lines[idx]:
                spin = "alpha"
            elif "Final Beta Molecular Orbital Analysis" in lines[idx]:
                spin = "beta"
            section_orbitals: list[dict[str, Any]] = []
            idx += 2
            while idx < len(lines):
                trimmed = lines[idx].strip()
                if not trimmed:
                    idx += 1
                    continue
                if (
                    trimmed.startswith("center of mass")
                    or trimmed.startswith("Task  times")
                    or trimmed.startswith("Parallel integral")
                ):
                    break
                if any(header in trimmed for header in headers):
                    break
                if trimmed.startswith("Vector"):
                    orbital, idx = _parse_single_orbital(lines, idx, spin=spin)
                    if orbital is not None:
                        section_orbitals.append(orbital)
                    continue
                idx += 1
            latest_sections[spin or "restricted"] = section_orbitals
            continue
        idx += 1

    orbitals: list[dict[str, Any]]
    if "alpha" in latest_sections or "beta" in latest_sections:
        orbitals = latest_sections.get("alpha", []) + latest_sections.get("beta", [])
    else:
        orbitals = latest_sections.get("restricted", [])

    spin_occupancies: dict[str, dict[int, float]] = defaultdict(dict)
    for orbital in orbitals:
        if orbital["spin"] in {"alpha", "beta"}:
            spin_occupancies[orbital["spin"]][orbital["vector_number"]] = orbital["occupancy"]

    compact_orbitals = []
    for orbital in orbitals:
        coeffs = orbital["coefficients"]
        ranked = sorted(coeffs, key=lambda item: abs(item["coefficient"]), reverse=True)
        character = _summarize_orbital_character(coeffs, top_n=top_n)
        occupation_label = _classify_orbital_occupation(
            orbital["occupancy"],
            spin=orbital["spin"],
            vector_number=orbital["vector_number"],
            spin_occupancies=spin_occupancies,
        )
        compact_orbitals.append(
            {
                "vector_number": orbital["vector_number"],
                "occupancy": orbital["occupancy"],
                "occupation_label": occupation_label,
                "energy_hartree": orbital["energy_hartree"],
                "symmetry": orbital["symmetry"],
                "spin": orbital["spin"],
                "mo_center_angstrom": orbital["mo_center_angstrom"],
                "r_squared": orbital["r_squared"],
                "visible_weight": character["visible_weight"],
                "dominant_character": character["dominant_character"],
                "top_atom_contributions": character["top_atom_contributions"],
                "ao_shell_contributions": character["ao_shell_contributions"],
                "top_contributors": ranked[:top_n],
                "coefficients": coeffs if include_coefficients else None,
            }
        )

    occupied = [orbital for orbital in compact_orbitals if orbital["occupancy"] > 0.1]
    virtual = [orbital for orbital in compact_orbitals if orbital["occupancy"] <= 0.1]
    homo = max(occupied, key=lambda item: item["energy_hartree"]) if occupied else None
    lumo = min(virtual, key=lambda item: item["energy_hartree"]) if virtual else None
    gap = None
    if homo is not None and lumo is not None:
        gap = lumo["energy_hartree"] - homo["energy_hartree"]

    population_sections = _parse_population_sections(contents)
    populations = _build_population_availability(population_sections)
    spin_channels = _summarize_spin_channels(compact_orbitals, top_n=top_n)
    somos = sorted(
        [orbital for orbital in compact_orbitals if orbital["occupation_label"] == "singly_occupied"],
        key=lambda item: item["energy_hartree"],
        reverse=True,
    )

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "orbital_count": len(compact_orbitals),
        "occupied_count": len(occupied),
        "virtual_count": len(virtual),
        "somo_count": len(somos),
        "somos": [_frontier_orbital(orbital) for orbital in somos[: max(top_n, 3)]],
        "spin_channels": spin_channels,
        "homo": _frontier_orbital(homo),
        "lumo": _frontier_orbital(lumo),
        "homo_lumo_gap_hartree": gap,
        "populations": populations,
        "orbitals": compact_orbitals if include_all_orbitals else _frontier_orbital_window(
            compact_orbitals, homo=homo, lumo=lumo, somos=somos, window=top_n
        ),
    }


def parse_population_analysis(path: str, contents: str) -> dict[str, Any]:
    sections = _parse_population_sections(contents)
    methods = _build_population_methods(sections)
    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "section_count": len(sections),
        "available_methods": [name for name, payload in methods.items() if payload["available"]],
        "methods": methods,
    }


def parse_mcscf_output(path: str, contents: str) -> dict[str, Any]:
    lines = contents.splitlines()
    block_settings = _extract_named_block_settings(lines, "mcscf")
    summary = _parse_mcscf_module_summary(lines)

    energies = [
        value
        for value in (parse_scientific_float(match.group(1)) for match in MCSCF_ENERGY_RE.finditer(contents))
        if value is not None
    ]
    total_energy_match = MCSCF_TOTAL_ENERGY_RE.search(contents)
    total_energy = parse_scientific_float(total_energy_match.group(1)) if total_energy_match else None

    residue_warnings: list[dict[str, float | None]] = []
    for match in MCSCF_RESIDUE_RE.finditer(contents):
        residue_warnings.append(
            {
                "current": parse_scientific_float(match.group(1)),
                "required": parse_scientific_float(match.group(2)),
            }
        )

    level_shift_adjustments = [
        value
        for value in (parse_scientific_float(match.group(1)) for match in MCSCF_LEVELSHIFT_RE.finditer(contents))
        if value is not None
    ]
    negative_curvatures = [
        value
        for value in (parse_scientific_float(match.group(1)) for match in MCSCF_NEGATIVE_CURVATURE_RE.finditer(contents))
        if value is not None
    ]

    ci_vector = _parse_mcscf_ci_vector(lines)
    natural_occupations = _parse_mcscf_natural_occupations(lines)
    active_space_mulliken = _parse_mcscf_mulliken_density(lines, "active space")
    total_density_mulliken = _parse_mcscf_mulliken_density(lines, "total")

    lower_contents = contents.lower()
    parse_error = "failed to parse multiplicity in state" in lower_contents or "there is an error in the input file" in lower_contents
    task_times = "task  times" in lower_contents
    converged_ci = "converged ci vector" in lower_contents

    if parse_error:
        status = "failed"
        failure_mode = "input_parse_error"
    elif total_energy is not None and task_times:
        status = "converged"
        failure_mode = None
    elif energies:
        status = "incomplete"
        failure_mode = "mcscf_not_finished"
    else:
        status = "unknown"
        failure_mode = None

    initial_levelshift = _coerce_float(block_settings.get("level")) or _coerce_float(summary.get("initial_levelshift"))
    final_levelshift = level_shift_adjustments[-1] if level_shift_adjustments else initial_levelshift

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "status": status,
        "failure_mode": failure_mode,
        "settings": {
            "active_orbitals": _coerce_int(block_settings.get("active")) or _coerce_int(summary.get("active_shells")),
            "active_electrons": _coerce_int(block_settings.get("actelec")) or _coerce_int(summary.get("active_electrons")),
            "multiplicity": _coerce_int(block_settings.get("multiplicity")) or _coerce_int(summary.get("multiplicity")),
            "state": block_settings.get("state"),
            "symmetry": block_settings.get("symmetry") or summary.get("symmetry"),
            "hessian": block_settings.get("hessian"),
            "maxiter": _coerce_int(block_settings.get("maxiter")) or _coerce_int(summary.get("maximum_iterations")),
            "thresh": _coerce_float(block_settings.get("thresh")) or _coerce_float(summary.get("converge_threshold")),
            "tol2e": _coerce_float(block_settings.get("tol2e")),
            "initial_levelshift": initial_levelshift,
            "vectors_input": block_settings.get("vectors_input"),
            "vectors_output": block_settings.get("vectors_output"),
            "inactive_shells": _coerce_int(summary.get("inactive_shells")),
            "active_shells": _coerce_int(summary.get("active_shells")),
            "determinants": _coerce_int(summary.get("determinants")),
        },
        "iteration_count": len(energies),
        "first_iteration_energy_hartree": energies[0] if energies else None,
        "final_iteration_energy_hartree": energies[-1] if energies else None,
        "final_energy_hartree": total_energy if total_energy is not None else (energies[-1] if energies else None),
        "energy_change_hartree": (energies[-1] - energies[0]) if len(energies) >= 2 else None,
        "iteration_energies_hartree": energies,
        "precondition_warning_count": len(residue_warnings),
        "residue_warnings": residue_warnings,
        "level_shift_adjustment_count": len(level_shift_adjustments),
        "level_shift_adjustments": level_shift_adjustments,
        "final_levelshift": final_levelshift,
        "negative_curvature_count": len(negative_curvatures),
        "negative_curvatures": negative_curvatures,
        "converged_ci_vector": converged_ci,
        "ci_vector": ci_vector,
        "natural_occupations": natural_occupations,
        "total_density_mulliken": total_density_mulliken,
        "active_space_mulliken": active_space_mulliken,
    }


def _parse_single_orbital(lines: list[str], header_index: int, spin: str | None = None) -> tuple[dict[str, Any] | None, int]:
    header = lines[header_index].strip()
    parts = split_tokens(header)
    if len(parts) < 4:
        return None, header_index + 1
    try:
        vector_number = int(parts[1])
    except ValueError:
        return None, header_index + 1

    occ_token = next((token for token in parts if token.startswith("Occ=")), None)
    if occ_token is None:
        return None, header_index + 1
    occupancy = parse_scientific_float(occ_token[4:])
    if occupancy is None:
        return None, header_index + 1

    energy_index = next((i for i, token in enumerate(parts) if token.startswith("E=")), None)
    if energy_index is None:
        return None, header_index + 1
    energy_token = parts[energy_index][2:] or (parts[energy_index + 1] if energy_index + 1 < len(parts) else "")
    energy = parse_scientific_float(energy_token)
    if energy is None:
        return None, header_index + 1

    symmetry = None
    for token in parts:
        if token.startswith("Symmetry="):
            symmetry = token[9:]
            break

    idx = header_index + 1
    mo_center = None
    r_squared = None
    if idx < len(lines) and lines[idx].strip().startswith("MO Center="):
        center_line = lines[idx].split("MO Center=", 1)[1]
        center_parts = [part.strip() for part in center_line.split(",")]
        if len(center_parts) >= 3:
            x = parse_scientific_float(center_parts[0])
            y = parse_scientific_float(center_parts[1])
            z_part = center_parts[2]
            if "r^2=" in z_part:
                z_text, r2_text = z_part.split("r^2=", 1)
                z = parse_scientific_float(z_text.strip())
                r_squared = parse_scientific_float(r2_text.strip())
            else:
                z = parse_scientific_float(z_part)
            if x is not None and y is not None and z is not None:
                mo_center = [x, y, z]
        idx += 3

    coefficients: list[dict[str, Any]] = []
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed:
            idx += 1
            break
        if trimmed.startswith("Vector"):
            break
        line = lines[idx]
        midpoint = len(line) // 2
        for chunk in (line[:midpoint], line[midpoint:]):
            entry = _parse_coefficient_entry(chunk)
            if entry is not None:
                coefficients.append(entry)
        idx += 1

    return (
        {
            "vector_number": vector_number,
            "occupancy": occupancy,
            "energy_hartree": energy,
            "symmetry": symmetry,
            "spin": spin,
            "mo_center_angstrom": mo_center,
            "r_squared": r_squared,
            "coefficients": coefficients,
        },
        idx,
    )


def _parse_coefficient_entry(text: str) -> dict[str, Any] | None:
    parts = split_tokens(text)
    if len(parts) < 4:
        return None
    try:
        basis_function_number = int(parts[0])
        coefficient = float(parts[1])
    except ValueError:
        return None
    label = " ".join(parts[2:])
    atom_index, element, ao_label, ao_shell = _parse_orbital_label(label)
    return {
        "basis_function_number": basis_function_number,
        "coefficient": coefficient,
        "label": label,
        "atom_index": atom_index,
        "element": element,
        "ao_label": ao_label,
        "ao_shell": ao_shell,
    }


def _frontier_orbital(orbital: dict[str, Any] | None) -> dict[str, Any] | None:
    if orbital is None:
        return None
    return {
        "vector_number": orbital["vector_number"],
        "occupancy": orbital["occupancy"],
        "occupation_label": orbital.get("occupation_label"),
        "energy_hartree": orbital["energy_hartree"],
        "symmetry": orbital["symmetry"],
        "spin": orbital.get("spin"),
        "dominant_character": orbital.get("dominant_character"),
        "top_atom_contributions": orbital.get("top_atom_contributions"),
        "ao_shell_contributions": orbital.get("ao_shell_contributions"),
    }


def _parse_orbital_label(label: str) -> tuple[int | None, str | None, str | None, str | None]:
    parts = split_tokens(label)
    if len(parts) < 2:
        return None, None, None, None
    atom_index = None
    try:
        atom_index = int(parts[0])
    except ValueError:
        pass
    element = parts[1] if len(parts) >= 2 else None
    ao_label = " ".join(parts[2:]) if len(parts) >= 3 else None
    ao_shell = None
    if ao_label:
        match = re.search(r"[A-Za-z]", ao_label)
        if match is not None:
            ao_shell = match.group(0).lower()
    return atom_index, element, ao_label, ao_shell


def _classify_orbital_occupation(
    occupancy: float,
    spin: str | None = None,
    vector_number: int | None = None,
    spin_occupancies: dict[str, dict[int, float]] | None = None,
) -> str:
    if occupancy <= 0.1:
        return "virtual"
    if spin in {"alpha", "beta"} and vector_number is not None and spin_occupancies is not None:
        other_spin = "beta" if spin == "alpha" else "alpha"
        other_occupancy = spin_occupancies.get(other_spin, {}).get(vector_number, 0.0)
        if other_occupancy <= 0.1:
            return "singly_occupied"
        return "occupied"
    if 0.25 < occupancy < 1.75:
        return "singly_occupied"
    return "occupied"


def _summarize_orbital_character(coefficients: list[dict[str, Any]], top_n: int = 5) -> dict[str, Any]:
    atom_weights: dict[tuple[int, str], float] = defaultdict(float)
    atom_functions: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    shell_weights: dict[str, float] = defaultdict(float)
    visible_weight = 0.0

    for coeff in coefficients:
        weight = coeff["coefficient"] ** 2
        visible_weight += weight
        atom_index = coeff.get("atom_index")
        element = coeff.get("element")
        if atom_index is not None and element:
            atom_key = (atom_index, element)
            atom_weights[atom_key] += weight
            atom_functions[atom_key].append(
                {
                    "label": coeff.get("ao_label") or coeff["label"],
                    "coefficient": coeff["coefficient"],
                    "weight": weight,
                }
            )
        shell = coeff.get("ao_shell")
        if shell:
            shell_weights[shell] += weight

    top_atom_contributions = []
    for (atom_index, element), weight in sorted(atom_weights.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        dominant_functions = sorted(atom_functions[(atom_index, element)], key=lambda item: item["weight"], reverse=True)
        top_atom_contributions.append(
            {
                "atom_index": atom_index,
                "element": element,
                "weight": weight,
                "fraction_of_visible": (weight / visible_weight) if visible_weight else 0.0,
                "dominant_functions": dominant_functions[: min(top_n, 3)],
            }
        )

    ao_shell_contributions = [
        {
            "ao_shell": shell,
            "weight": weight,
            "fraction_of_visible": (weight / visible_weight) if visible_weight else 0.0,
        }
        for shell, weight in sorted(shell_weights.items(), key=lambda item: item[1], reverse=True)
    ]

    dominant_atom_labels = [
        f"{item['atom_index']} {item['element']} ({item['fraction_of_visible'] * 100:.0f}%)"
        for item in top_atom_contributions[:3]
    ]
    dominant_shell_labels = [
        f"{item['ao_shell']}-like ({item['fraction_of_visible'] * 100:.0f}%)"
        for item in ao_shell_contributions[:2]
    ]
    dominant_character_parts = dominant_atom_labels[:2] + dominant_shell_labels[:1]

    return {
        "visible_weight": visible_weight,
        "top_atom_contributions": top_atom_contributions,
        "ao_shell_contributions": ao_shell_contributions,
        "dominant_character": ", ".join(dominant_character_parts) if dominant_character_parts else None,
    }


def _summarize_spin_channels(orbitals: list[dict[str, Any]], top_n: int = 5) -> dict[str, Any]:
    channels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for orbital in orbitals:
        channels[orbital.get("spin") or "restricted"].append(orbital)

    summary: dict[str, Any] = {}
    for channel, channel_orbitals in channels.items():
        occupied = [orbital for orbital in channel_orbitals if orbital["occupancy"] > 0.1]
        virtual = [orbital for orbital in channel_orbitals if orbital["occupancy"] <= 0.1]
        somos = [
            orbital
            for orbital in channel_orbitals
            if orbital["occupation_label"] == "singly_occupied"
        ]
        homo = max(occupied, key=lambda item: item["energy_hartree"]) if occupied else None
        lumo = min(virtual, key=lambda item: item["energy_hartree"]) if virtual else None
        gap = None
        if homo is not None and lumo is not None:
            gap = lumo["energy_hartree"] - homo["energy_hartree"]
        summary[channel] = {
            "orbital_count": len(channel_orbitals),
            "occupied_count": len(occupied),
            "virtual_count": len(virtual),
            "somo_count": len(somos),
            "somos": [_frontier_orbital(orbital) for orbital in sorted(somos, key=lambda item: item["energy_hartree"], reverse=True)[:top_n]],
            "homo": _frontier_orbital(homo),
            "lumo": _frontier_orbital(lumo),
            "homo_lumo_gap_hartree": gap,
        }
    return summary


def _parse_population_sections(contents: str) -> list[dict[str, Any]]:
    lines = contents.splitlines()
    sections: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        match = POPULATION_HEADER_RE.match(lines[idx])
        if match is None:
            idx += 1
            continue

        density_kind = "total" if match.group(1).lower().startswith("total") else "spin"
        method_label = match.group(2).lower()
        method = "lowdin" if method_label.startswith("low") or method_label.startswith("löw") else "mulliken"

        idx += 1
        while idx < len(lines) and not lines[idx].strip().startswith("Atom"):
            idx += 1
        if idx >= len(lines):
            break
        idx += 2

        atoms: list[dict[str, Any]] = []
        while idx < len(lines):
            trimmed = lines[idx].strip()
            if not trimmed:
                break
            if not trimmed[0].isdigit():
                break
            atom = _parse_population_atom(trimmed, density_kind=density_kind)
            if atom is not None:
                atoms.append(atom)
            idx += 1

        if atoms:
            sections.append(_finalize_population_section(method, density_kind, atoms, section_index=len(sections) + 1))
        continue
    return sections


def _parse_population_atom(line: str, density_kind: str) -> dict[str, Any] | None:
    parts = split_tokens(line)
    if len(parts) < 4:
        return None
    try:
        atom_index = int(parts[0])
    except ValueError:
        return None
    element = parts[1]
    nuclear_charge = parse_scientific_float(parts[2])
    population = parse_scientific_float(parts[3])
    if nuclear_charge is None or population is None:
        return None
    shell_populations = [value for token in parts[4:] if (value := parse_scientific_float(token)) is not None]
    net_atomic_charge = (nuclear_charge - population) if density_kind == "total" else None
    return {
        "atom_index": atom_index,
        "element": element,
        "nuclear_charge": nuclear_charge,
        "population": population,
        "net_atomic_charge": net_atomic_charge,
        "shell_populations": shell_populations,
    }


def _finalize_population_section(
    method: str,
    density_kind: str,
    atoms: list[dict[str, Any]],
    section_index: int,
) -> dict[str, Any]:
    population_sum = sum(atom["population"] for atom in atoms)
    net_charge_sum = None
    largest_positive_sites = None
    largest_negative_sites = None
    largest_spin_sites = None

    if density_kind == "total":
        net_charge_sum = sum((atom["net_atomic_charge"] or 0.0) for atom in atoms)
        largest_positive_sites = sorted(atoms, key=lambda item: item["net_atomic_charge"] or 0.0, reverse=True)[:5]
        largest_negative_sites = sorted(atoms, key=lambda item: item["net_atomic_charge"] or 0.0)[:5]
    else:
        largest_spin_sites = sorted(atoms, key=lambda item: abs(item["population"]), reverse=True)[:5]

    return {
        "section_index": section_index,
        "method": method,
        "density_kind": density_kind,
        "atom_count": len(atoms),
        "population_sum": population_sum,
        "net_charge_sum": net_charge_sum,
        "largest_positive_sites": largest_positive_sites,
        "largest_negative_sites": largest_negative_sites,
        "largest_spin_sites": largest_spin_sites,
        "atoms": atoms,
    }


def _build_population_methods(sections: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in ("mulliken", "lowdin"):
        total_sections = [section for section in sections if section["method"] == method and section["density_kind"] == "total"]
        spin_sections = [section for section in sections if section["method"] == method and section["density_kind"] == "spin"]
        methods[method] = {
            "available": bool(total_sections or spin_sections),
            "section_counts": {
                "total": len(total_sections),
                "spin": len(spin_sections),
            },
            "latest_total": total_sections[-1] if total_sections else None,
            "latest_spin": spin_sections[-1] if spin_sections else None,
        }
    methods["natural"] = {
        "available": False,
        "section_counts": {"total": 0, "spin": 0},
        "latest_total": None,
        "latest_spin": None,
    }
    return methods


def _build_population_availability(sections: list[dict[str, Any]]) -> dict[str, Any]:
    methods = _build_population_methods(sections)
    availability: dict[str, Any] = {}
    for method, payload in methods.items():
        latest_total = payload["latest_total"]
        latest_spin = payload["latest_spin"]
        availability[method] = {
            "available": payload["available"],
            "has_total_density": latest_total is not None,
            "has_spin_density": latest_spin is not None,
            "section_counts": payload["section_counts"],
            "net_charge_sum": latest_total["net_charge_sum"] if latest_total is not None else None,
            "spin_population_sum": latest_spin["population_sum"] if latest_spin is not None else None,
        }
    return availability


def _extract_named_block_settings(lines: list[str], block_name: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    in_block = False
    for raw_line in lines:
        stripped = raw_line.strip()
        lower = stripped.lower()
        if not in_block:
            if lower == block_name.lower():
                in_block = True
            continue
        if lower == "end":
            break
        setting_match = MCSCF_SETTING_RE.match(raw_line)
        if setting_match:
            settings[setting_match.group(1).lower()] = setting_match.group(2).strip()
            continue
        if lower.startswith("vectors "):
            settings.update(_parse_mcscf_vectors_line(stripped))
    return settings


def _parse_mcscf_vectors_line(line: str) -> dict[str, str]:
    parts = split_tokens(line)
    payload: dict[str, str] = {}
    for idx, token in enumerate(parts[:-1]):
        lower = token.lower()
        if lower == "input":
            payload["vectors_input"] = parts[idx + 1]
        elif lower == "output":
            payload["vectors_output"] = parts[idx + 1]
    return payload


def _parse_mcscf_module_summary(lines: list[str]) -> dict[str, str]:
    summary: dict[str, str] = {}
    joined = "\n".join(lines)
    determinants_match = re.search(r"No\.\s+of determinants:\s+(\d+)", joined, re.IGNORECASE)
    if determinants_match:
        summary["determinants"] = determinants_match.group(1)

    for index, raw_line in enumerate(lines):
        if "NWChem Direct MCSCF Module" not in raw_line:
            continue
        for follow in lines[index + 1 :]:
            stripped = follow.strip()
            if not stripped:
                continue
            if stripped.startswith("Loading old vectors from job with title"):
                break
            match = MCSCF_SUMMARY_VALUE_RE.match(follow)
            if not match:
                continue
            key = match.group(1).strip().lower().replace(" ", "_")
            summary[key] = match.group(2).strip()
        break
    return summary


def _parse_mcscf_ci_vector(lines: list[str]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    in_section = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if not in_section:
            if stripped == "Converged CI vector":
                in_section = True
            continue
        if not stripped:
            if entries:
                break
            continue
        if stripped.startswith("Index"):
            continue
        match = re.match(r"^\s*(\d+)\s+([-\d.DEde+]+)\s+(.+?)\s*$", raw_line)
        if not match:
            if entries:
                break
            continue
        entries.append(
            {
                "index": int(match.group(1)),
                "coefficient": parse_scientific_float(match.group(2)),
                "occupation": match.group(3).strip(),
            }
        )
    entries.sort(key=lambda item: abs(item["coefficient"] or 0.0), reverse=True)
    return {
        "configuration_count": len(entries),
        "leading_configurations": entries[:20],
    }


def _parse_mcscf_natural_occupations(lines: list[str]) -> list[dict[str, Any]]:
    occupations: list[dict[str, Any]] = []
    in_section = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if not in_section:
            if stripped == "Natural orbital occupation numbers":
                in_section = True
            continue
        if not stripped:
            if occupations:
                break
            continue
        if re.match(r"^\d+\s*$", stripped):
            continue
        match = re.match(r"^\s*(\d+)\s+([-\d.DEde+]+)\s*$", raw_line)
        if not match:
            if occupations:
                break
            continue
        occupations.append(
            {
                "orbital_index": int(match.group(1)),
                "occupation": parse_scientific_float(match.group(2)),
            }
        )
    return occupations


def _parse_mcscf_mulliken_density(lines: list[str], label: str) -> dict[str, Any] | None:
    target_header = f"Mulliken analysis of the {label} density".lower()
    atoms: list[dict[str, Any]] = []
    in_section = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if not in_section:
            if stripped.lower() == target_header:
                in_section = True
            continue
        if not stripped:
            if atoms:
                break
            continue
        if stripped.startswith("Atom") or stripped.startswith("-----------") or stripped.startswith("-"):
            continue
        match = re.match(r"^\s*(\d+)\s+([A-Za-z]{1,3})\s+\d+\s+([-\d.DEde+]+)\s+", raw_line)
        if not match:
            if atoms:
                break
            continue
        atoms.append(
            {
                "atom_index": int(match.group(1)),
                "element": match.group(2),
                "charge": parse_scientific_float(match.group(3)),
            }
        )
    if not atoms:
        return None
    return {
        "atom_count": len(atoms),
        "atoms": atoms,
    }


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    parsed = parse_scientific_float(str(value))
    if parsed is None:
        return None
    return int(round(parsed))


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return parse_scientific_float(str(value))

