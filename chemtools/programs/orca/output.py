"""Parse bounded completion and scientific evidence from ORCA text output.

The initial contract covers markers observed in the ORCA 6.1.1 reference
cases. It preserves warnings as evidence without treating them as failures.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from chemtools.programs.orca.excited_states import (
    ACTIVE_ORBITAL_FAILURE,
    parse_excited_state_evidence,
)
from chemtools.programs.orca.esd import parse_esd_evidence


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_VERSION_RE = re.compile(
    r"^\s*Program Version\s+([^\s]+)\s+-\s+RELEASE\s+-\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_INPUT_FILE_RE = re.compile(r"^NAME\s*=\s*(.+?\S)\s*$")
_SIMPLE_INPUT_RE = re.compile(r"^\|\s*\d+>\s*!\s*(.*\S)\s*$")
_BASIS_RE = re.compile(r"^Your calculation utilizes the basis:\s*(\S+)\s*$")
_AUXILIARY_BASIS_RE = re.compile(
    r"^Your calculation utilizes the auxiliary basis:\s*(\S+)\s*$"
)
_NUMBER_OF_BASIS_FUNCTIONS_RE = re.compile(
    r"^Number of basis functions\s+\.\.\.\s+(\d+)\s*$",
    re.IGNORECASE,
)
_NUMBER_OF_ELECTRONS_RE = re.compile(
    r"^Number of Electrons\s+NEL\s+\.\.\.\.\s+(\d+)\s*$",
    re.IGNORECASE,
)
_RELATIVISTIC_METHOD_RE = re.compile(
    r"^Relativistic Method\s+\.\.\.\s+(.+?\S)\s*$",
    re.IGNORECASE,
)
_SCF_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", re.IGNORECASE)
_SCF_FAILURE_RE = re.compile(
    r"SCF NOT CONVERGED AFTER\s+(\d+)\s+CYCLES",
    re.IGNORECASE,
)
_INITIAL_GUESS_RE = re.compile(r"^INITIAL GUESS:\s*(.+?\S)\s*$", re.IGNORECASE)
_ERROR_TERMINATION_RE = re.compile(
    r"^ORCA finished by error termination in\s+(.+?\S)\s*$",
    re.IGNORECASE,
)
_RIJCOSX_ON_RE = re.compile(r"^RIJ-COSX\b.*\.{4}\s+on$", re.IGNORECASE)
_COUPLED_CLUSTER_CONVERGED = "The Coupled-Cluster iterations have converged"
_T1_DIAGNOSTIC_RE = re.compile(
    rf"^T1 diagnostic\s+\.\.\.\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_TRIPLES_CORRECTION_RE = re.compile(
    rf"^Triples Correction \(T\)\s+\.\.\.\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_FINAL_CORRELATION_ENERGY_RE = re.compile(
    rf"^Final correlation energy\s+\.\.\.\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_CCSD_ENERGY_RE = re.compile(
    rf"^E\(CCSD\)\s+\.\.\.\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_CCSD_T_ENERGY_RE = re.compile(
    rf"^E\(CCSD\(T\)\)\s+\.\.\.\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_ENERGY_RE = re.compile(
    rf"^FINAL SINGLE POINT ENERGY\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_MM_ENERGY_RE = re.compile(
    rf"^FINAL SINGLE POINT ENERGY \(MM\)\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_QMMM_ENERGY_RE = re.compile(
    rf"^FINAL SINGLE POINT ENERGY \(QM/MM\)\s+({_FLOAT})\s*$",
    re.IGNORECASE,
)
_MULTISCALE_VALUE_RE = re.compile(
    r"^(Multiscale model|Coupling Scheme|Embedding Scheme)\s+\.\.\.\s+(.+?\S)\s*$",
    re.IGNORECASE,
)
_MULTISCALE_COUNT_RE = re.compile(
    r"^(Point charges in QM calc\. from MM atoms|Size of QMMM System|"
    r"Size of MM Subsystem|Size of QM Subsystem(?: \(excl HF/ECP\))?|"
    r"Number of link atoms|Number of ECP layers|Number of ECP layer atoms)"
    r"\s*\.\.\.\s+(\d+)\s*$",
    re.IGNORECASE,
)
_CHARGE_CONVERGENCE_RE = re.compile(
    rf"^Maximum charge difference\s+({_FLOAT})\s+"
    rf"\(Threshold:\s+({_FLOAT})\)\s*$",
    re.IGNORECASE,
)
_OPT_CYCLES_RE = re.compile(r"\(AFTER\s+(\d+)\s+CYCLES\)", re.IGNORECASE)
_FREQUENCY_RE = re.compile(
    rf"^\s*(\d+):\s+({_FLOAT})\s+cm\*\*-1\s*$",
    re.IGNORECASE,
)
_SPIN_EXPECTATION_RE = re.compile(
    rf"^Expectation value of <S\*\*2>\s*:\s*({_FLOAT})\s*$",
    re.IGNORECASE,
)
_SPIN_IDEAL_RE = re.compile(
    rf"^Ideal value S\*\(S\+1\) for S=({_FLOAT})\s*:\s*({_FLOAT})\s*$",
    re.IGNORECASE,
)
_SPIN_POPULATION_SUM_RE = re.compile(
    rf"^Sum of atomic spin populations:\s*({_FLOAT})\s*$",
    re.IGNORECASE,
)
_WAVEFUNCTION_TYPE_RE = re.compile(
    r"^Hartree-Fock type\s+HFTyp\s+\.\.\.\.\s+(\S+)\s*$",
    re.IGNORECASE,
)
_TOTAL_CHARGE_RE = re.compile(
    r"^Total Charge\s+Charge\s+\.\.\.\.\s+([+-]?\d+)\s*$",
    re.IGNORECASE,
)
_MULTIPLICITY_RE = re.compile(
    r"^Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)\s*$",
    re.IGNORECASE,
)
_THERMOCHEMISTRY_RE = re.compile(
    rf"^THERMOCHEMISTRY AT\s+({_FLOAT})K\s*$",
    re.IGNORECASE,
)
_PRESSURE_RE = re.compile(rf"^Pressure\s+\.{{3}}\s+({_FLOAT})\s+atm\s*$")
_RUNTIME_RE = re.compile(
    r"^TOTAL RUN TIME:\s*(\d+) days (\d+) hours (\d+) minutes "
    r"(\d+) seconds (\d+) msec\s*$",
    re.IGNORECASE,
)
_WARNING_RE = re.compile(r"^\s*WARNING:\s*(.*\S)\s*$", re.IGNORECASE)
_GEOMETRY_HEADER = "CARTESIAN COORDINATES (ANGSTROEM)"
_NORMAL_TERMINATION = "****ORCA TERMINATED NORMALLY****"


def looks_like_orca(text: str) -> bool:
    head = text[:8192]
    return _VERSION_RE.search(head) is not None and bool(
        re.search(r"^\s*\*\s+O\s+R\s+C\s+A\s+\*\s*$", head, re.MULTILINE)
    )


def parse_version(text: str) -> str | None:
    match = _VERSION_RE.search(text[:8192])
    return match.group(1) if match else None


def parse_orca_output(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return parse_orca_output_text(
        source.read_text(encoding="utf-8", errors="replace")
    )


def parse_orca_output_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    version = parse_version(text)
    if version is None:
        raise ValueError("ORCA output does not contain an ORCA release banner.")

    parsed: dict[str, Any] = {
        "format": "orca-output/1",
        "program_version": version,
        "line_count": len(lines),
        "input_file": None,
        "simple_keywords": [],
        "basis": None,
        "basis_sets": [],
        "auxiliary_basis_sets": [],
        "number_of_basis_functions": None,
        "number_of_electrons": None,
        "relativistic_method": None,
        "ri_approximation": None,
        "initial_guess": None,
        "scf_cycles": [],
        "scf_failures": [],
        "coupled_cluster": {
            "converged_line": None,
            "t1_diagnostic": None,
            "t1_diagnostic_line": None,
            "triples_correction_hartree": None,
            "triples_correction_line": None,
            "correlation_energy_hartree": None,
            "correlation_energy_line": None,
            "ccsd_energy_hartree": None,
            "ccsd_energy_line": None,
            "ccsd_t_energy_hartree": None,
            "ccsd_t_energy_line": None,
        },
        "casscf": None,
        "multireference_pt2": None,
        "mrci": None,
        "tddft": None,
        "eom_ccsd": None,
        "esd": None,
        "multiscale": None,
        "energies": [],
        "optimization": {
            "started_line": None,
            "converged_line": None,
            "cycles": None,
        },
        "frequency": {
            "started_line": None,
            "all_frequencies_cm1": [],
            "frequencies_cm1": [],
            "imaginary_frequencies_cm1": [],
        },
        "thermochemistry": None,
        "wavefunction_type": None,
        "charge": None,
        "multiplicity": None,
        "spin": {},
        "warnings": [],
        "normal_termination": False,
        "normal_termination_line": None,
        "error_termination": None,
        "runtime_seconds": None,
        "geometry": _last_geometry(lines),
    }

    in_frequency_table = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if match := _INPUT_FILE_RE.match(stripped):
            parsed["input_file"] = match.group(1)
        if match := _SIMPLE_INPUT_RE.match(line):
            parsed["simple_keywords"].extend(match.group(1).split())
        if match := _BASIS_RE.match(stripped):
            basis = {"name": match.group(1), "line": number}
            if basis["name"] not in {
                item["name"] for item in parsed["basis_sets"]
            }:
                parsed["basis_sets"].append(basis)
            if parsed["basis"] is None:
                parsed["basis"] = basis["name"]
        if match := _AUXILIARY_BASIS_RE.match(stripped):
            auxiliary_basis = {"name": match.group(1), "line": number}
            if auxiliary_basis["name"] not in {
                item["name"] for item in parsed["auxiliary_basis_sets"]
            }:
                parsed["auxiliary_basis_sets"].append(auxiliary_basis)
        if (
            parsed["number_of_basis_functions"] is None
            and (match := _NUMBER_OF_BASIS_FUNCTIONS_RE.match(stripped))
        ):
            parsed["number_of_basis_functions"] = int(match.group(1))
        if match := _NUMBER_OF_ELECTRONS_RE.match(stripped):
            parsed["number_of_electrons"] = int(match.group(1))
        if match := _RELATIVISTIC_METHOD_RE.match(stripped):
            parsed["relativistic_method"] = match.group(1)
        if _RIJCOSX_ON_RE.match(stripped):
            parsed["ri_approximation"] = {
                "name": "RIJCOSX",
                "line": number,
            }
        if match := _SCF_RE.search(line):
            parsed["scf_cycles"].append({
                "cycles": int(match.group(1)),
                "line": number,
            })
        if match := _SCF_FAILURE_RE.search(line):
            parsed["scf_failures"].append({
                "cycles": int(match.group(1)),
                "line": number,
            })
        if match := _INITIAL_GUESS_RE.match(stripped):
            parsed["initial_guess"] = match.group(1)
        if _COUPLED_CLUSTER_CONVERGED in line:
            parsed["coupled_cluster"]["converged_line"] = number
        if match := _T1_DIAGNOSTIC_RE.match(stripped):
            parsed["coupled_cluster"]["t1_diagnostic"] = _float(
                match.group(1)
            )
            parsed["coupled_cluster"]["t1_diagnostic_line"] = number
        if match := _TRIPLES_CORRECTION_RE.match(stripped):
            parsed["coupled_cluster"]["triples_correction_hartree"] = (
                _float(match.group(1))
            )
            parsed["coupled_cluster"]["triples_correction_line"] = number
        if match := _FINAL_CORRELATION_ENERGY_RE.match(stripped):
            parsed["coupled_cluster"]["correlation_energy_hartree"] = (
                _float(match.group(1))
            )
            parsed["coupled_cluster"]["correlation_energy_line"] = number
        if match := _CCSD_ENERGY_RE.match(stripped):
            parsed["coupled_cluster"]["ccsd_energy_hartree"] = _float(
                match.group(1)
            )
            parsed["coupled_cluster"]["ccsd_energy_line"] = number
        if match := _CCSD_T_ENERGY_RE.match(stripped):
            parsed["coupled_cluster"]["ccsd_t_energy_hartree"] = _float(
                match.group(1)
            )
            parsed["coupled_cluster"]["ccsd_t_energy_line"] = number
        if match := _MULTISCALE_VALUE_RE.match(stripped):
            multiscale = _ensure_multiscale(parsed)
            key = {
                "multiscale model": "model",
                "coupling scheme": "coupling_scheme",
                "embedding scheme": "embedding_scheme",
            }[match.group(1).casefold()]
            multiscale[key] = match.group(2)
        if match := _MULTISCALE_COUNT_RE.match(stripped):
            multiscale = _ensure_multiscale(parsed)
            key = {
                "point charges in qm calc. from mm atoms": "point_charge_count",
                "size of qmmm system": "system_size_atoms",
                "size of mm subsystem": "mm_atoms",
                "size of qm subsystem": "qm_atoms",
                "size of qm subsystem (excl hf/ecp)": "qm_atoms",
                "number of link atoms": "link_atoms",
                "number of ecp layers": "ecp_layers",
                "number of ecp layer atoms": "ecp_atoms",
            }[match.group(1).casefold()]
            multiscale[key] = int(match.group(2))
        if match := _CHARGE_CONVERGENCE_RE.match(stripped):
            multiscale = _ensure_multiscale(parsed)
            multiscale["charge_convergence"].append({
                "maximum_difference": _float(match.group(1)),
                "threshold": _float(match.group(2)),
                "line": number,
            })
        if match := _MM_ENERGY_RE.match(stripped):
            multiscale = _ensure_multiscale(parsed)
            multiscale["mm_energy_hartree"] = _float(match.group(1))
            multiscale["mm_energy_line"] = number
        if match := _QMMM_ENERGY_RE.match(stripped):
            multiscale = _ensure_multiscale(parsed)
            multiscale["qmmm_energy_hartree"] = _float(match.group(1))
            multiscale["qmmm_energy_line"] = number
        if match := _ENERGY_RE.match(stripped):
            parsed["energies"].append({
                "value_hartree": _float(match.group(1)),
                "line": number,
            })
        if stripped == "* Geometry Optimization Run *":
            parsed["optimization"]["started_line"] = number
        if "THE OPTIMIZATION HAS CONVERGED" in line:
            parsed["optimization"]["converged_line"] = number
        if match := _OPT_CYCLES_RE.search(line):
            parsed["optimization"]["cycles"] = int(match.group(1))
        if stripped == "VIBRATIONAL FREQUENCIES":
            parsed["frequency"]["started_line"] = number
            in_frequency_table = True
            continue
        if in_frequency_table and stripped == "NORMAL MODES":
            in_frequency_table = False
        if in_frequency_table and (match := _FREQUENCY_RE.match(line)):
            parsed["frequency"]["all_frequencies_cm1"].append(
                _float(match.group(2))
            )
        if match := _THERMOCHEMISTRY_RE.match(stripped):
            parsed["thermochemistry"] = {
                "temperature_kelvin": _float(match.group(1)),
                "pressure_atm": None,
                "line": number,
            }
        elif parsed["thermochemistry"] is not None and (
            match := _PRESSURE_RE.match(stripped)
        ):
            parsed["thermochemistry"]["pressure_atm"] = _float(
                match.group(1)
            )
        if match := _SPIN_EXPECTATION_RE.match(stripped):
            parsed["spin"]["expectation_s2"] = _float(match.group(1))
            parsed["spin"]["expectation_line"] = number
        if match := _SPIN_IDEAL_RE.match(stripped):
            parsed["spin"]["spin_s"] = _float(match.group(1))
            parsed["spin"]["ideal_s2"] = _float(match.group(2))
        if match := _SPIN_POPULATION_SUM_RE.match(stripped):
            parsed["spin"]["mulliken_spin_population_sum"] = _float(
                match.group(1)
            )
            parsed["spin"]["mulliken_spin_population_sum_line"] = number
        if match := _WAVEFUNCTION_TYPE_RE.match(stripped):
            parsed["wavefunction_type"] = match.group(1)
        if match := _TOTAL_CHARGE_RE.match(stripped):
            parsed["charge"] = int(match.group(1))
        if match := _MULTIPLICITY_RE.match(stripped):
            parsed["multiplicity"] = int(match.group(1))
        if match := _WARNING_RE.match(line):
            parsed["warnings"].append({
                "message": match.group(1),
                "line": number,
            })
        if ACTIVE_ORBITAL_FAILURE in line:
            parsed["warnings"].append({
                "message": ACTIVE_ORBITAL_FAILURE,
                "line": number,
            })
        if stripped == _NORMAL_TERMINATION:
            parsed["normal_termination"] = True
            parsed["normal_termination_line"] = number
        if match := _ERROR_TERMINATION_RE.match(stripped):
            parsed["error_termination"] = {
                "module": match.group(1),
                "line": number,
            }
        if match := _RUNTIME_RE.match(stripped):
            days, hours, minutes, seconds, milliseconds = map(
                int, match.groups()
            )
            parsed["runtime_seconds"] = (
                days * 86400
                + hours * 3600
                + minutes * 60
                + seconds
                + milliseconds / 1000
            )

    all_frequencies = parsed["frequency"]["all_frequencies_cm1"]
    parsed["frequency"]["frequencies_cm1"] = [
        value for value in all_frequencies if value != 0.0
    ]
    parsed["frequency"]["imaginary_frequencies_cm1"] = [
        value for value in all_frequencies if value < 0.0
    ]
    parsed.update(parse_excited_state_evidence(lines))
    parsed["esd"] = parse_esd_evidence(lines)
    return parsed


def _ensure_multiscale(parsed: dict[str, Any]) -> dict[str, Any]:
    if parsed["multiscale"] is None:
        parsed["multiscale"] = {
            "model": None,
            "coupling_scheme": None,
            "embedding_scheme": None,
            "point_charge_count": None,
            "system_size_atoms": None,
            "mm_atoms": None,
            "qm_atoms": None,
            "link_atoms": None,
            "ecp_layers": None,
            "ecp_atoms": None,
            "charge_convergence": [],
            "mm_energy_hartree": None,
            "mm_energy_line": None,
            "qmmm_energy_hartree": None,
            "qmmm_energy_line": None,
        }
    return parsed["multiscale"]


def _last_geometry(lines: list[str]) -> dict[str, Any] | None:
    geometries = []
    for index, line in enumerate(lines):
        if line.strip() != _GEOMETRY_HEADER:
            continue
        atoms = []
        for number, candidate in enumerate(lines[index + 2 :], start=index + 3):
            stripped = candidate.strip()
            if not stripped:
                break
            fields = stripped.split()
            if len(fields) < 4:
                break
            try:
                x, y, z = (_float(value) for value in fields[1:4])
            except ValueError:
                break
            atoms.append({
                "element": fields[0],
                "x": x,
                "y": y,
                "z": z,
                "line": number,
            })
        if atoms:
            geometries.append({
                "units": "angstrom",
                "atoms": atoms,
                "header_line": index + 1,
            })
    return geometries[-1] if geometries else None


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = [
    "looks_like_orca",
    "parse_orca_output",
    "parse_orca_output_text",
    "parse_version",
]
