from __future__ import annotations


from collections import defaultdict, deque
import math
import re
from typing import Any

from chemtools.core.common import make_metadata, parse_float_after_delimiter, parse_scientific_float, split_tokens


METHOD_PATTERNS: list[tuple[int, str, tuple[str, ...]]] = [
    (5, "CCSD(T)", ("ccsd(t)",)),
    (4, "CCSD", ("ccsd total energy", " ccsd ")),
    (3, "MP2", ("total mp2 energy", " mp2 ")),
    (3, "MCSCF", ("total mcscf energy", " mcscf ")),
    (2, "DFT", ("total dft energy", " dft ", "b3lyp", "pbe0", "pbe ")),
    (1, "SCF", ("total scf energy", " scf ", " rhf", " uhf", " rohf")),
    (0, "TCE", ("tce",)),
]
ENERGY_PREFIXES: list[tuple[int, tuple[str, ...]]] = [
    (5, ("ccsd(t) total energy", "total ccsd(t) energy")),
    (4, ("ccsd total energy", "total ccsd energy")),
    (3, ("mbpt(2) total energy", "total mp2 energy")),
    (3, ("total mcscf energy",)),
    (2, ("total dft energy",)),
    (1, ("total scf energy",)),
]
TRAILING_FLOAT_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)\s*$"
)
BASIS_LIBRARY_ASSIGNMENT_RE = re.compile(
    r'^\s*(?:\*|\S+)\s+library\s+(?:"([^"]+)"|(\S+))',
    re.IGNORECASE,
)

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
# Re-exported from core for backward compatibility — see chemtools.core.common
from chemtools.core.common import COVALENT_RADII  # noqa: F401


def detect_method_token(line_lc: str) -> tuple[int, str] | None:
    for priority, label, needles in METHOD_PATTERNS:
        if any(needle in line_lc for needle in needles):
            return priority, label
    if line_lc.endswith("scf"):
        return 1, "SCF"
    return None


def detect_energy_token(line: str) -> tuple[int, float] | None:
    lc = line.strip().lower()
    for priority, prefixes in ENERGY_PREFIXES:
        if not lc.startswith(prefixes):
            continue
        match = TRAILING_FLOAT_RE.search(line)
        if match is None:
            return None
        value = parse_scientific_float(match.group(1))
        return (priority, value) if value is not None else None
    return None


def detect_basis_token(line: str) -> str | None:
    match = BASIS_LIBRARY_ASSIGNMENT_RE.match(line)
    if match is None:
        return None
    return match.group(1) or match.group(2)


def _task_label(kind: str, method: str | None) -> str:
    base = {
        "gradient": "Gradient",
        "optimization": "Optimization",
        "frequency": "Frequency",
        "raman": "Raman",
        "single_point": "Single Point",
        "mcscf": "MCSCF",
        "property": "Property",
        "tddft": "TDDFT (excited states)",
        "dplot": "Density Plot",
        "unknown": "Task",
    }.get(kind, "Task")
    return f"{base} · {method}" if method else base


def parse_tasks(path: str, contents: str) -> dict[str, Any]:
    lines = contents.splitlines(keepends=True)
    summary_tasks: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    global_method: tuple[int, str] | None = None
    global_basis_families: set[str] = set()

    task: dict[str, Any] = {
        "kind": None,
        "start_byte": 0,
        "start_line": 1,
        "opt_frames": 0,
        "opt_energy": None,
        "opt_energy_priority": None,
        "opt_energy_profile": [],
        "freq_modes": [],
        "pending_freq_indices": deque(),
        "sp_energy": None,
        "method": None,
        "module_method": None,
        "tddft_energy": None,
        "basis_families": set(),
        "in_section": False,
        "has_errors": False,
    }

    def reset_task(start_byte: int, start_line: int) -> None:
        task["kind"] = None
        task["start_byte"] = start_byte
        task["start_line"] = start_line
        task["opt_frames"] = 0
        task["opt_energy"] = None
        task["opt_energy_priority"] = None
        task["opt_energy_profile"] = []
        task["freq_modes"] = []
        task["pending_freq_indices"] = deque()
        task["sp_energy"] = None
        task["method"] = None
        task["module_method"] = None
        task["tddft_energy"] = None
        task["basis_families"] = set()
        task["in_section"] = False
        task["has_errors"] = False

    def emit_task(
        end_byte: int,
        end_line: int,
        saw_task_times: bool,
    ) -> None:
        kind = task["kind"]
        if kind is None:
            return
        method_hint = task["method"] or global_method
        method = task["module_method"] or (
            method_hint[1] if method_hint else None
        )
        basis_families = (
            task["basis_families"] or global_basis_families
        )
        basis_hint = (
            next(iter(basis_families))
            if len(basis_families) == 1
            else None
        )
        outcome = "failed" if task["has_errors"] else ("success" if saw_task_times else "incomplete")
        if kind == "optimization":
            total_energy = task["opt_energy"]
            frame_count = max(task["opt_frames"], 1)
            mode_count = None
            energy_profile = list(task["opt_energy_profile"])
            freq_modes = []
        else:
            if (
                task["module_method"] == "TDDFT"
                and task["tddft_energy"] is not None
            ):
                total_energy = task["tddft_energy"]
            else:
                total_energy = (
                    task["sp_energy"][1] if task["sp_energy"] else None
                )
            frame_count = None
            mode_count = len(task["freq_modes"]) or None
            energy_profile = []
            freq_modes = list(task["freq_modes"])
        summary_tasks.append(
            {
                "kind": kind,
                "label": _task_label(kind, method),
                "method": method,
                "basis": basis_hint,
                "total_energy_hartree": total_energy,
                "frame_count": frame_count,
                "mode_count": mode_count,
                "energy_profile": energy_profile,
                "frequency_modes": freq_modes,
                "boundary": {
                    "start_byte": task["start_byte"],
                    "end_byte": end_byte,
                    "line_start": task["start_line"],
                    "line_end": end_line,
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
            global_basis_families.add(basis_token)
            if task["in_section"]:
                task["basis_families"].add(basis_token)

        if task["in_section"] and (
            "error:" in lc
            or "aborting" in lc
            or "segmentation fault" in lc
            or "nwc_abort" in lc
            or "hnd_property: energy failure" in lc
            or "there is an error in the input file" in lc
        ):
            task["has_errors"] = True
            if task["kind"] is None:
                task["kind"] = "unknown"
            diagnostics.append({"kind": "error", "message": ltrim, "line": line_number})

        energy_token = detect_energy_token(ltrim)
        if energy_token is not None:
            task["in_section"] = True
            priority, value = energy_token
            if task["kind"] == "optimization":
                current_priority = task["opt_energy_priority"]
                if current_priority is None or priority > current_priority:
                    task["opt_energy_priority"] = priority
                    task["opt_energy_profile"] = [value]
                    task["opt_energy"] = value
                elif priority == current_priority:
                    if (
                        not task["opt_energy_profile"]
                        or task["opt_energy_profile"][-1] != value
                    ):
                        task["opt_energy_profile"].append(value)
                    task["opt_energy"] = value
            if task["sp_energy"] is None or priority >= task["sp_energy"][0]:
                task["sp_energy"] = energy_token

        if task["kind"] == "tddft" and "excited state energy" in lc:
            value = parse_float_after_delimiter(ltrim, "=")
            if value is not None:
                task["tddft_energy"] = value

        if "NWChem Input Module" in line:
            if task["kind"] is not None:
                emit_task(current_byte, line_number - 1, False)
            reset_task(current_byte, line_number)
            task["in_section"] = True
            current_byte += len(raw_line)
            continue

        if "task" in lc and "times" in lc and ("cpu:" in lc or "wall:" in lc):
            emit_task(current_byte, line_number, True)
            reset_task(current_byte + len(raw_line), line_number + 1)
            current_byte += len(raw_line)
            continue

        if "nwchem tddft gradient module" in lc:
            task["kind"] = "gradient"
            task["module_method"] = "TDDFT"
        elif "nwchem tddft module" in lc:
            task["kind"] = "tddft"
            task["module_method"] = "TDDFT"
        elif "specified for the density plot" in lc:
            task["kind"] = "dplot"
            task["module_method"] = "DPLOT"
        elif task["kind"] is None:
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
            elif "nwchem dft module" in lc or "nwchem scf module" in lc:
                task["kind"] = "single_point"
            elif energy_token is not None:
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

    emit_task(current_byte, len(lines), False)

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
                "boundary": {
                    "start_byte": 0,
                    "end_byte": current_byte,
                    "line_start": 1 if lines else 0,
                    "line_end": len(lines),
                },
                "outcome": "unknown",
            }
        )

    generic_tasks = []
    for task_summary in summary_tasks:
        generic_kind = {
            "optimization": "optimization",
            "gradient": "gradient",
            "frequency": "frequency",
            "raman": "frequency",
            "single_point": "single_point",
            "mcscf": "single_point",
            "tce": "single_point",
            "tddft": "single_point",
            "property": "property",
            "dplot": "property",
        }.get(task_summary["kind"], "other")
        boundary = task_summary["boundary"]
        generic_tasks.append(
            {
                "program": "nwchem",
                "kind": generic_kind,
                "label": task_summary["label"],
                "energy_hartree": task_summary["total_energy_hartree"],
                "line_start": boundary["line_start"],
                "line_end": boundary["line_end"],
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
