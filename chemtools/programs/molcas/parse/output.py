from __future__ import annotations

import re
from typing import Any

from chemtools.core.common import make_metadata
from chemtools.programs.molcas.parse.scf import parse_scf
from chemtools.programs.molcas.parse.rasscf import parse_rasscf
from chemtools.programs.molcas.parse.caspt2 import parse_caspt2, assess_reference_weights
from chemtools.programs.molcas.parse.mrci import parse_mrci
from chemtools.programs.molcas.parse.ccsdt import parse_ccsdt
from chemtools.programs.molcas.parse.mos import parse_last_mo_block
from chemtools.programs.molcas.parse.rassi import parse_rassi

MODULE_RE = re.compile(r"---\s+Start Module:\s+([A-Za-z0-9_]+)")
STOP_MODULE_RE = re.compile(r"---\s+Stop Module:\s+([A-Za-z0-9_]+)\s+at[^/]*\/rc=(\S+)")
INTERNAL_MODULES = {"last_energy", "last_atoms", "emil"}

# Return codes that indicate normal flow (not failure). SLAPAF / numerical
# gradient drivers exit with these to hand control back to the loop.
_CLEAN_RETURN_CODES: set[str] = {
    "_RC_ALL_IS_WELL_",
    "_RC_INVOKED_OTHER_MODULE_",
    "_RC_CONTINUE_LOOP_",
    "_RC_CONTINUE_UNIX_LOOP_",
    "_RC_EXIT_LOOP_",
    "_RC_EXIT_EXPECTED_",
}


def classify_module(name: str) -> str:
    upper = name.upper()
    if upper in {"SEWARD", "GATEWAY"}:
        return "integrals"
    if upper == "SCF":
        return "scf"
    if upper == "RASSCF":
        return "rasscf"
    if upper == "CASPT2":
        return "caspt2"
    if upper == "RASSI":
        return "rassi"
    if upper in {"SLAPAF", "ALASKA", "NUMGRAD"}:
        return "optimization"
    if upper == "VIBROT":
        return "frequency"
    if "SO" in upper or "SPIN" in upper:
        return "spin_orbit"
    return "other"


def parse_tasks(path: str, contents: str) -> dict[str, Any]:
    """Identify Molcas module boundaries and emit a generic task list.

    Cheap pass — does not invoke per-module parsers. For deep extraction call
    parse_output_full().
    """
    tasks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    last_line = 0

    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        last_line = line_number
        stripped = raw_line.rstrip()
        match = MODULE_RE.search(stripped)
        if match:
            module = match.group(1).strip().upper()
            if module.lower() in INTERNAL_MODULES:
                continue
            if current is not None:
                current["line_end"] = line_number - 1
                tasks.append(current)
            current = {
                "kind": classify_module(module),
                "module": module,
                "line_start": line_number,
                "line_end": line_number,
            }
            continue
        # Capture Stop Module's return code so we can emit a per-task status
        stop = STOP_MODULE_RE.search(stripped)
        if stop and current is not None and stop.group(1).upper() == current["module"]:
            current["return_code"] = stop.group(2)

    if current is not None:
        current["line_end"] = max(last_line, current["line_start"])
        tasks.append(current)

    generic_tasks = []
    for task in tasks:
        return_code = task.get("return_code")
        if return_code in _CLEAN_RETURN_CODES:
            outcome = "success"
        elif return_code:
            outcome = "failed"
        else:
            outcome = "incomplete"
        generic_kind = {
            "optimization": "optimization",
            "frequency": "frequency",
            "scf": "scf",
        }.get(task["kind"], "other")
        generic_tasks.append(
            {
                "program": "molcas",
                "kind": generic_kind,
                "label": task["kind"].replace("_", " ").title(),
                "energy_hartree": None,
                "line_start": task["line_start"],
                "line_end": task["line_end"],
                "extra": {
                    "module": task["module"],
                    "return_code": return_code,
                    "outcome": outcome,
                },
            }
        )

    return {
        "metadata": make_metadata(path, contents, "molcas"),
        "generic_tasks": generic_tasks,
        "program_summary": {
            "kind": "molcas",
            "task_count": len(tasks),
            "raw": {"tasks": tasks},
        },
    }


# --- Top-level orchestrator ----------------------------------------------------

def parse_output_full(
    path: str,
    contents: str,
    *,
    parse_mo_coefficients: bool = False,
) -> dict[str, Any]:
    """Run the cheap pass + dispatch each task to its module-specific parser.

    Returns:
      {
        "metadata": ...,
        "tasks_overview": [generic_task, ...],          # from parse_tasks
        "task_payloads": [                              # one per task, in order
          {"task_index": 0, "module": "SCF", "kind": "scf",
           "line_range": [start, end], "return_code": "...",
           "details": <module-specific dict>},
          ...
        ],
        "energy_summary": {                             # roll-up across tasks
          "scf_total_hartree": ...,
          "rasscf_root_energies": [...],
          "caspt2_root_energies": [...],
          "ms_caspt2_root_energies": [...],
          "primary_energy_hartree": ...,
          "primary_label": "MS-CASPT2 root 1" / "RASSCF root 1" / "SCF",
        },
        "active_space_summary": ...,
        "warnings": [aggregated cross-task warnings],
      }
    """
    base = parse_tasks(path, contents)
    lines = contents.splitlines()

    payloads: list[dict[str, Any]] = []
    aggregated_warnings: list[dict[str, Any]] = []
    energy_summary: dict[str, Any] = {}
    active_space_summary: dict[str, Any] | None = None

    for idx, gtask in enumerate(base["generic_tasks"]):
        line_start = gtask["line_start"] - 1  # 0-indexed
        line_end = gtask["line_end"]          # inclusive in 1-indexed → exclusive in 0-indexed slice
        block_text = "\n".join(lines[line_start:line_end])
        module = gtask["extra"]["module"]
        rc = gtask["extra"].get("return_code")
        kind = gtask["kind"]
        details: dict[str, Any] = {}

        if module == "SCF":
            details = parse_scf(block_text)
            if (e := details.get("final_energy", {}).get("total")) is not None:
                energy_summary["scf_total_hartree"] = e
        elif module == "RASSCF":
            details = parse_rasscf(block_text)
            if details.get("root_energies"):
                energy_summary["rasscf_root_energies"] = details["root_energies"]
            if details.get("active_space_signature"):
                active_space_summary = {
                    "signature": details["active_space_signature"],
                    "wave_function": details.get("wave_function"),
                    "orbital_specs": details.get("orbital_specs"),
                    "natural_occupation_quality": [
                        w["summary"] for w in details.get("natural_occupation_warnings", [])
                    ],
                }
        elif module == "CASPT2":
            details = parse_caspt2(block_text)
            if details.get("ss_root_energies"):
                energy_summary["caspt2_root_energies"] = details["ss_root_energies"]
            if details.get("ms_root_energies"):
                energy_summary["ms_caspt2_root_energies"] = details["ms_root_energies"]
            quality = assess_reference_weights(details.get("per_group_results", []))
            if quality:
                details["reference_weight_quality"] = quality
            for w in details.get("warnings", []):
                aggregated_warnings.append({**w, "task_index": idx, "module": "CASPT2"})
        elif module == "MRCI":
            details = parse_mrci(block_text)
            if details.get("state_energies"):
                # Keep the lowest state of the latest MRCI/ACPF run.
                energy_summary["mrci_state_energies"] = details["state_energies"]
                energy_summary["mrci_variant"] = details.get("variant")
        elif module == "CCSDT":
            details = parse_ccsdt(block_text)
            if details.get("ccsd_energy_hartree") is not None:
                energy_summary["ccsd_energy_hartree"] = details["ccsd_energy_hartree"]
            if details.get("ccsd_t_energy_hartree") is not None:
                energy_summary["ccsd_t_energy_hartree"] = details["ccsd_t_energy_hartree"]
        elif module == "RASSI":
            details = parse_rassi(block_text)
            if details.get("spin_free_states", {}).get("rows"):
                energy_summary["rassi_spin_free_states"] = [
                    {"sf_state": r["sf_state_index"], "energy_hartree": r["absolute_energy_au"]}
                    for r in details["spin_free_states"]["rows"]
                ]
            if details.get("so_states", {}).get("rows"):
                energy_summary["rassi_so_states"] = [
                    {"so_state": r["so_state_index"], "energy_hartree": r["absolute_energy_au"]}
                    for r in details["so_states"]["rows"]
                ]

        # MO last-block (only for modules where MOs are emitted): SCF, RASSCF
        if module in {"SCF", "RASSCF"}:
            mo_block = parse_last_mo_block(
                block_text, parse_coefficients=parse_mo_coefficients
            )
            if mo_block:
                details["mo_block"] = mo_block

        if rc and rc not in _CLEAN_RETURN_CODES:
            aggregated_warnings.append(
                {
                    "code": "module_failed",
                    "severity": "high",
                    "message": f"Module {module} returned failure code: {rc}",
                    "task_index": idx,
                    "module": module,
                }
            )

        payloads.append(
            {
                "task_index": idx,
                "module": module,
                "kind": kind,
                "line_range": [gtask["line_start"], gtask["line_end"]],
                "return_code": rc,
                "details": details,
            }
        )

    # Pick a "primary" energy: prefer SO-RASSI > RASSI SF > MS-CASPT2 > CASPT2 > RASSCF > SCF
    primary_energy = None
    primary_label = None
    if energy_summary.get("rassi_so_states"):
        primary_energy = min(s["energy_hartree"] for s in energy_summary["rassi_so_states"])
        primary_label = "SO-RASSI ground"
    elif energy_summary.get("rassi_spin_free_states"):
        primary_energy = min(s["energy_hartree"] for s in energy_summary["rassi_spin_free_states"])
        primary_label = "RASSI spin-free ground"
    elif energy_summary.get("ccsd_t_energy_hartree") is not None:
        primary_energy = energy_summary["ccsd_t_energy_hartree"]
        primary_label = "CCSD(T)"
    elif energy_summary.get("ccsd_energy_hartree") is not None:
        primary_energy = energy_summary["ccsd_energy_hartree"]
        primary_label = "CCSD"
    elif energy_summary.get("mrci_state_energies"):
        primary_energy = min(s["energy_hartree"] for s in energy_summary["mrci_state_energies"])
        primary_label = f"{energy_summary.get('mrci_variant') or 'MRCI'} state 1"
    elif energy_summary.get("ms_caspt2_root_energies"):
        primary_energy = energy_summary["ms_caspt2_root_energies"][0]["energy_hartree"]
        primary_label = "MS-CASPT2 root 1"
    elif energy_summary.get("caspt2_root_energies"):
        primary_energy = energy_summary["caspt2_root_energies"][0]["energy_hartree"]
        primary_label = "CASPT2 root 1"
    elif energy_summary.get("rasscf_root_energies"):
        primary_energy = energy_summary["rasscf_root_energies"][0]["energy_hartree"]
        primary_label = "RASSCF root 1"
    elif energy_summary.get("scf_total_hartree") is not None:
        primary_energy = energy_summary["scf_total_hartree"]
        # Distinguish KS-DFT from HF/ROHF in the label so downstream tools
        # don't mislabel a DFT-only run as "SCF".
        scf_method = next(
            (
                p["details"].get("method")
                for p in payloads
                if p["module"] == "SCF" and isinstance(p.get("details"), dict)
            ),
            "SCF",
        )
        primary_label = "KS-DFT" if scf_method == "KSDFT" else "SCF"
    if primary_energy is not None:
        energy_summary["primary_energy_hartree"] = primary_energy
        energy_summary["primary_label"] = primary_label

    return {
        "metadata": base["metadata"],
        "tasks_overview": base["generic_tasks"],
        "task_payloads": payloads,
        "energy_summary": energy_summary,
        "active_space_summary": active_space_summary,
        "warnings": aggregated_warnings,
    }
