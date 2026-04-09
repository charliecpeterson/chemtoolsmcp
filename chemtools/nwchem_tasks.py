from __future__ import annotations


from collections import defaultdict, deque
import math
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


def detect_method_token(line_lc: str) -> tuple[int, str] | None:
    for priority, label, needles in METHOD_PATTERNS:
        if any(needle in line_lc for needle in needles):
            return priority, label
    if line_lc.endswith("scf"):
        return 1, "SCF"
    return None


def detect_energy_token(line: str) -> tuple[int, float] | None:
    lc = line.lower()
    if "ccsd(t) total energy" in lc:
        value = parse_float_after_delimiter(line, "=")
        return (5, value) if value is not None else None
    if "ccsd total energy" in lc:
        value = parse_float_after_delimiter(line, "=")
        return (4, value) if value is not None else None
    if "mbpt(2) total energy" in lc:
        value = parse_float_after_delimiter(line, "=")
        return (3, value) if value is not None else None
    if "total mp2 energy" in lc:
        value = parse_float_after_delimiter(line, "=")
        return (3, value) if value is not None else None
    if "total dft energy" in lc:
        value = parse_float_after_delimiter(line, "=")
        return (2, value) if value is not None else None
    if "total mcscf energy" in lc:
        value = parse_float_after_delimiter(line, "=")
        return (3, value) if value is not None else None
    if "total scf energy" in lc:
        value = parse_float_after_delimiter(line, "=")
        return (1, value) if value is not None else None
    return None


def detect_basis_token(line: str) -> str | None:
    stripped = line.strip()
    lc = stripped.lower()

    if lc.startswith("basis"):
        if '"' in stripped:
            first = stripped.find('"')
            second = stripped.find('"', first + 1)
            if second > first:
                return stripped[first + 1 : second]
        parts = stripped.split()
        return parts[-1].strip('"') if parts else None

    if " library " in f" {lc} ":
        parts = stripped.split()
        for idx, token in enumerate(parts[:-1]):
            if token.lower() == "library":
                return parts[idx + 1].strip('"')

    if lc.startswith("setting basis") and "=" in stripped:
        return stripped.split("=", 1)[1].strip().strip('"')

    return None


def _task_label(kind: str, method: str | None) -> str:
    base = {
        "optimization": "Optimization",
        "frequency": "Frequency",
        "raman": "Raman",
        "single_point": "Single Point",
        "mcscf": "MCSCF",
        "property": "Property",
        "unknown": "Task",
    }.get(kind, "Task")
    return f"{base} · {method}" if method else base


def parse_tasks(path: str, contents: str) -> dict[str, Any]:
    lines = contents.splitlines(keepends=True)
    summary_tasks: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    global_method: tuple[int, str] | None = None
    global_basis: str | None = None

    task: dict[str, Any] = {
        "kind": None,
        "start_byte": 0,
        "opt_frames": 0,
        "opt_energy": None,
        "opt_energy_profile": [],
        "freq_modes": [],
        "pending_freq_indices": deque(),
        "sp_energy": None,
        "method": None,
        "basis": None,
        "in_section": False,
        "has_errors": False,
    }

    def reset_task(start_byte: int) -> None:
        task["kind"] = None
        task["start_byte"] = start_byte
        task["opt_frames"] = 0
        task["opt_energy"] = None
        task["opt_energy_profile"] = []
        task["freq_modes"] = []
        task["pending_freq_indices"] = deque()
        task["sp_energy"] = None
        task["method"] = None
        task["basis"] = None
        task["in_section"] = False
        task["has_errors"] = False

    def emit_task(end_byte: int, saw_task_times: bool) -> None:
        kind = task["kind"]
        if kind is None:
            return
        method_hint = task["method"] or global_method
        basis_hint = task["basis"] or global_basis
        outcome = "failed" if task["has_errors"] else ("success" if saw_task_times else "incomplete")
        if kind == "optimization":
            total_energy = task["opt_energy"]
            frame_count = max(task["opt_frames"], 1)
            mode_count = None
            energy_profile = list(task["opt_energy_profile"])
            freq_modes = []
        else:
            total_energy = task["sp_energy"][1] if task["sp_energy"] else None
            frame_count = None
            mode_count = len(task["freq_modes"]) or None
            energy_profile = []
            freq_modes = list(task["freq_modes"])
        summary_tasks.append(
            {
                "kind": kind,
                "label": _task_label(kind, method_hint[1] if method_hint else None),
                "method": method_hint[1] if method_hint else None,
                "basis": basis_hint,
                "total_energy_hartree": total_energy,
                "frame_count": frame_count,
                "mode_count": mode_count,
                "energy_profile": energy_profile,
                "frequency_modes": freq_modes,
                "boundary": {
                    "start_byte": task["start_byte"],
                    "end_byte": end_byte,
                },
                "outcome": outcome,
            }
        )

    current_byte = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        ltrim = line.strip()
        lc = ltrim.lower()

        method_token = detect_method_token(lc)
        if method_token is not None:
            if global_method is None or method_token[0] > global_method[0]:
                global_method = method_token
            if task["in_section"] and (task["method"] is None or method_token[0] > task["method"][0]):
                task["method"] = method_token

        basis_token = detect_basis_token(line)
        if basis_token is not None:
            global_basis = basis_token
            if task["in_section"]:
                task["basis"] = basis_token

        if task["in_section"] and (
            "error:" in lc or "aborting" in lc or "segmentation fault" in lc or "nwc_abort" in lc
        ):
            task["has_errors"] = True
            diagnostics.append({"kind": "error", "message": ltrim, "line": line_number})

        energy_token = detect_energy_token(ltrim)
        if task["in_section"] and energy_token is not None:
            priority, value = energy_token
            if task["kind"] == "optimization" and priority <= 2:
                if not task["opt_energy_profile"] or task["opt_energy_profile"][-1] != value:
                    task["opt_energy_profile"].append(value)
                task["opt_energy"] = value
            if task["sp_energy"] is None or priority >= task["sp_energy"][0]:
                task["sp_energy"] = energy_token

        if "NWChem Input Module" in line:
            if task["kind"] is not None:
                emit_task(current_byte, False)
            reset_task(current_byte)
            task["in_section"] = True
            current_byte += len(raw_line)
            continue

        if "task" in lc and "times" in lc and ("cpu:" in lc or "wall:" in lc):
            emit_task(current_byte, True)
            reset_task(current_byte + len(raw_line))
            current_byte += len(raw_line)
            continue

        if task["kind"] is None:
            if "nwchem geometry optimization" in lc:
                task["kind"] = "optimization"
            elif "raman analysis" in lc:
                task["kind"] = "raman"
            elif "normal mode eigenvectors" in lc or "nwchem nuclear hessian and frequency" in lc:
                task["kind"] = "frequency"
            elif "nwchem property module" in lc:
                task["kind"] = "property"
            elif "nwchem direct mcscf module" in lc:
                task["kind"] = "mcscf"
            elif "extensible many-electron theory" in lc or "tensor contraction engine" in lc:
                task["kind"] = "tce"
            elif "total scf energy" in lc or "total dft energy" in lc:
                task["kind"] = "single_point"

        if task["kind"] == "optimization":
            if "output coordinates in angstroms" in lc or "output coordinates in a.u." in lc:
                task["opt_frames"] += 1
        elif task["kind"] in {"frequency", "raman"}:
            trimmed = line.lstrip()
            if (
                trimmed.startswith("Frequency")
                and not trimmed.startswith("P.Frequency")
                and "=" not in trimmed
            ):
                values = [float(token) for token in trimmed.split()[1:] if parse_scientific_float(token) is not None]
                for freq in values:
                    task["freq_modes"].append({"frequency_cm1": freq, "ir_intensity": None})
                    task["pending_freq_indices"].append(len(task["freq_modes"]) - 1)
            elif trimmed.startswith("IR Inten") or trimmed.lower().startswith("ir intensity"):
                tokens = trimmed.split()
                values: list[float] = []
                for token in tokens:
                    parsed = parse_scientific_float(token)
                    if parsed is not None:
                        values.append(parsed)
                for ir_intensity in values:
                    if not task["pending_freq_indices"]:
                        break
                    idx = task["pending_freq_indices"].popleft()
                    task["freq_modes"][idx]["ir_intensity"] = ir_intensity

        current_byte += len(raw_line)

    emit_task(current_byte, False)

    if not summary_tasks:
        summary_tasks.append(
            {
                "kind": "unknown",
                "label": "Task",
                "method": None,
                "basis": None,
                "total_energy_hartree": None,
                "frame_count": None,
                "mode_count": None,
                "energy_profile": [],
                "frequency_modes": [],
                "boundary": {"start_byte": 0, "end_byte": current_byte},
                "outcome": "unknown",
            }
        )

    generic_tasks = []
    for task_summary in summary_tasks:
        generic_kind = {
            "optimization": "optimization",
            "frequency": "frequency",
            "raman": "frequency",
            "single_point": "single_point",
            "mcscf": "single_point",
            "tce": "single_point",
        }.get(task_summary["kind"], "other")
        generic_tasks.append(
            {
                "program": "nwchem",
                "kind": generic_kind,
                "label": task_summary["label"],
                "energy_hartree": task_summary["total_energy_hartree"],
                "line_start": None,
                "line_end": None,
                "extra": {
                    "basis": task_summary["basis"],
                    "frame_count": task_summary["frame_count"],
                    "mode_count": task_summary["mode_count"],
                    "opt_energy_trajectory": task_summary["energy_profile"],
                    "outcome": task_summary["outcome"],
                },
            }
        )

    outcome = "unknown"
    if any(task_summary["outcome"] == "failed" for task_summary in summary_tasks):
        outcome = "failed"
    elif all(task_summary["outcome"] == "success" for task_summary in summary_tasks):
        outcome = "success"
    elif any(task_summary["outcome"] == "incomplete" for task_summary in summary_tasks):
        outcome = "incomplete"

    return {
        "metadata": make_metadata(path, contents, "nwchem"),
        "generic_tasks": generic_tasks,
        "program_summary": {
            "kind": "nwchem",
            "outcome": outcome,
            "task_count": len(summary_tasks),
            "diagnostics": diagnostics,
            "raw": {
                "tasks": summary_tasks,
                "outcome": outcome,
                "diagnostics": diagnostics,
            },
        },
    }

