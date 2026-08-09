"""QE-side artifact evidence used before QMCPACK conversion."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
import posixpath
from typing import Any
from xml.etree import ElementTree

import numpy as np

from chemtools.core.hdf5 import hdf5_signature_offset
from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.qe._elements import element_from_label
from chemtools.programs.qe.control_paths import inspect_explicit_control_paths
from chemtools.programs.qe.input_geometry import normalize_pw_input_geometry
from chemtools.programs.qmcpack.particles import (
    collect_ion_geometries,
    collect_particle_sets,
    electron_particle_count,
    electron_spin_population,
    non_electron_particle_sets,
)
from chemtools.programs.qmcpack.pseudopotential import (
    collect_pseudopotential_references,
    inspect_qmcpack_pseudopotential,
)


def plan_qe_qmcpack_conversion(
    qe_input: str | Path,
    pwscf_h5: str | Path,
    parsed_input: dict[str, Any],
) -> dict[str, Any]:
    """Describe the declared QE-to-QMCPACK artifact handoff without running it."""
    input_path = Path(qe_input).expanduser().resolve()
    h5_path = Path(pwscf_h5).expanduser().resolve()
    preflight = [
        inspect_conversion_calculation(parsed_input),
        inspect_conversion_disk_io(parsed_input),
        inspect_conversion_k_points(parsed_input),
        inspect_conversion_isolation(parsed_input),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-plan/1",
        "qe_input": str(input_path),
        "pwscf_h5": str(h5_path),
        "readiness": conversion_readiness(preflight),
        "preflight": preflight,
        "steps": [
            {
                "id": "qe_scf",
                "program": "qe",
                "executable": "pw.x",
                "input": str(input_path),
                "required_evidence": "completed_converged_scf_output",
            },
            {
                "id": "pw2qmcpack_conversion",
                "program": "qe",
                "executable": "pw2qmcpack.x",
                "requires": {
                    "qe_input": str(input_path),
                    "scf_wavefunctions": "retained_by_qe",
                },
                "produces": {"pwscf_h5": str(h5_path)},
                "command_line": None,
            },
            {
                "id": "qmcpack_deck_validation",
                "program": "qmcpack",
                "requires": {"pwscf_h5": str(h5_path)},
                "tool": "inspect_qe_qmcpack_conversion_deck",
            },
        ],
        "scope_limit": (
            "This declares artifact provenance and the bounded preflight checks. "
            "It does not generate pw2qmcpack input, choose converter options, "
            "launch QE or QMCPACK, or inspect HDF5 contents."
        ),
    }


def inspect_qe_pw2qmcpack_control_paths(
    parsed_qe_input: dict[str, Any],
    parsed_converter_input: dict[str, Any],
) -> dict[str, Any]:
    """Compare explicit QE and converter ``prefix`` and ``outdir`` settings."""
    qe_paths = inspect_explicit_control_paths(parsed_qe_input)
    qe_prefix = qe_paths["prefix"]
    qe_outdir = qe_paths["outdir"]
    converter_values = parsed_converter_input.get("namelist")
    if not isinstance(converter_values, dict):
        converter_values = {}
    converter_prefix = _explicit_converter_path(converter_values.get("prefix"))
    converter_outdir = _explicit_converter_path(converter_values.get("outdir"))
    observed = {
        "qe": {"prefix": qe_prefix, "outdir": qe_outdir},
        "pw2qmcpack": {"prefix": converter_prefix, "outdir": converter_outdir},
    }
    if qe_prefix is None or qe_outdir is None:
        return {
            "name": "qe_pw2qmcpack_control_paths",
            "status": "review_required",
            "observed": observed,
            "message": "The QE input lacks explicit prefix or outdir handoff evidence.",
        }
    if converter_prefix is None or converter_outdir is None:
        return {
            "name": "qe_pw2qmcpack_control_paths",
            "status": "review_required",
            "observed": observed,
            "message": "The converter input lacks explicit prefix or outdir handoff evidence.",
        }
    paths_match = (
        qe_prefix == converter_prefix
        and posixpath.normpath(qe_outdir) == posixpath.normpath(converter_outdir)
    )
    return {
        "name": "qe_pw2qmcpack_control_paths",
        "status": "pass" if paths_match else "not_ready",
        "observed": {
            **observed,
            "normalized_outdirs": {
                "qe": posixpath.normpath(qe_outdir),
                "pw2qmcpack": posixpath.normpath(converter_outdir),
            },
        },
        "message": (
            "QE and pw2qmcpack use the same explicit prefix and outdir."
            if paths_match
            else "QE and pw2qmcpack use different prefix or outdir values."
        ),
    }


def _explicit_converter_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped and "\n" not in stripped and "\r" not in stripped else None


def inspect_pwscf_h5_artifact(
    pwscf_h5: str | Path,
    qe_input: str | Path,
    qe_output: str | Path,
) -> dict[str, Any]:
    artifact_path = Path(pwscf_h5).expanduser().resolve()
    input_path = Path(qe_input).expanduser().resolve()
    output_path = Path(qe_output).expanduser().resolve()
    try:
        artifact = artifact_path.stat()
    except OSError:
        return {
            "name": "pwscf_h5_artifact",
            "status": "not_ready",
            "observed": {"exists": False},
            "message": "The declared .pwscf.h5 artifact is missing.",
        }
    if not artifact_path.is_file() or artifact.st_size == 0:
        return {
            "name": "pwscf_h5_artifact",
            "status": "not_ready",
            "observed": {"exists": True, "size_bytes": artifact.st_size},
            "message": "The declared .pwscf.h5 artifact is empty or not a regular file.",
        }
    signature_offset = hdf5_signature_offset(artifact_path, artifact.st_size)
    if signature_offset is None:
        return {
            "name": "pwscf_h5_artifact",
            "status": "not_ready",
            "observed": {
                "exists": True,
                "size_bytes": artifact.st_size,
                "hdf5_signature_offset": None,
            },
            "message": (
                "The declared .pwscf.h5 artifact does not contain an HDF5 "
                "signature at a supported superblock offset."
            ),
        }
    newest_source_ns = max(input_path.stat().st_mtime_ns, output_path.stat().st_mtime_ns)
    is_current = artifact.st_mtime_ns >= newest_source_ns
    return {
        "name": "pwscf_h5_artifact",
        "status": "pass" if is_current else "not_ready",
        "observed": {
            "exists": True,
            "size_bytes": artifact.st_size,
            "modified_ns": artifact.st_mtime_ns,
            "hdf5_signature_offset": signature_offset,
            "current_against_qe_input_and_output": is_current,
        },
        "message": (
            "The declared .pwscf.h5 artifact is present and current."
            if is_current
            else "The declared .pwscf.h5 artifact is older than the QE input or output."
        ),
    }


def inspect_qe_scf_completion(parsed_output: dict[str, Any]) -> dict[str, Any]:
    if parsed_output["errors"] or not parsed_output["job_done"]:
        status = "not_ready"
        message = "QE output does not show a clean completed calculation."
    elif (
        parsed_output["calculation_mode"] != "scf"
        or not parsed_output["scf_converged"]
    ):
        status = "review_required"
        message = "QE output completed, but SCF convergence is not established."
    else:
        status = "pass"
        message = "QE output records a completed converged SCF calculation."
    return {
        "name": "qe_output_completion",
        "status": status,
        "observed": {
            "calculation_mode": parsed_output["calculation_mode"],
            "scf_converged": parsed_output["scf_converged"],
            "job_done": parsed_output["job_done"],
            "error_count": len(parsed_output["errors"]),
        },
        "message": message,
    }


def inspect_conversion_calculation(parsed_input: dict[str, Any]) -> dict[str, Any]:
    calculation = parsed_input["calculation"]
    return {
        "name": "scf_calculation",
        "status": "pass" if calculation == "scf" else "review_required",
        "observed": calculation,
        "message": (
            "The conversion workflow is documented for an SCF input."
            if calculation == "scf"
            else "The documented conversion precondition is an SCF input."
        ),
    }


def inspect_conversion_disk_io(parsed_input: dict[str, Any]) -> dict[str, Any]:
    disk_io = parsed_input["namelists"].get("control", {}).get("disk_io")
    normalized = disk_io.casefold() if isinstance(disk_io, str) else None
    if normalized in {"medium", "high"}:
        status = "pass"
        message = "disk_io retains wavefunctions for pw2qmcpack conversion."
    elif normalized in {"low", "none"}:
        status = "not_ready"
        message = "disk_io may omit the wavefunctions needed by pw2qmcpack."
    else:
        status = "review_required"
        message = "Set disk_io='medium' or confirm a supported higher setting."
    return {
        "name": "disk_io",
        "status": status,
        "observed": disk_io,
        "message": message,
    }


def inspect_conversion_k_points(parsed_input: dict[str, Any]) -> dict[str, Any]:
    sampling = parsed_input["k_points"]
    if sampling is None:
        return {
            "name": "qmcpack_gamma_representation",
            "status": "review_required",
            "observed": None,
            "message": "An explicit K_POINTS crystal gamma point is required.",
        }
    option = sampling["option"]
    if option == "gamma":
        return {
            "name": "qmcpack_gamma_representation",
            "status": "not_ready",
            "observed": {"option": option},
            "message": "K_POINTS gamma uses the unsupported gamma trick.",
        }
    points = sampling.get("points")
    if (
        option == "crystal"
        and isinstance(points, list)
        and len(points) == 1
        and points[0]["coordinates"] == [0.0, 0.0, 0.0]
        and points[0]["weight"] == 1.0
    ):
        return {
            "name": "qmcpack_gamma_representation",
            "status": "pass",
            "observed": {"option": option, "point": points[0]},
            "message": "The gamma point uses the explicit crystal representation.",
        }
    return {
        "name": "qmcpack_gamma_representation",
        "status": "review_required",
        "observed": {"option": option, "points": points},
        "message": "Confirm that this k-point representation is supported by QMCPACK.",
    }


def inspect_conversion_isolation(parsed_input: dict[str, Any]) -> dict[str, Any]:
    setting = parsed_input["namelists"].get("system", {}).get("assume_isolated")
    normalized = setting.casefold() if isinstance(setting, str) else None
    if normalized is None:
        status = "pass"
        message = "No Martyna-Tuckerman isolation setting is present."
    elif normalized == "m-t":
        status = "not_ready"
        message = "Martyna-Tuckerman isolation does not match QMCPACK Ewald conventions."
    else:
        status = "review_required"
        message = "Confirm that this isolation setting matches QMCPACK conventions."
    return {
        "name": "hamiltonian_convention",
        "status": status,
        "observed": setting,
        "message": message,
    }


def inspect_conversion_readiness(
    qe_input: str | Path,
    parsed_input: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the bounded QE input checks required before conversion."""
    input_path = Path(qe_input).expanduser().resolve()
    checks = [
        inspect_conversion_calculation(parsed_input),
        inspect_conversion_disk_io(parsed_input),
        inspect_conversion_k_points(parsed_input),
        inspect_conversion_isolation(parsed_input),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-readiness/1",
        "qe_input": str(input_path),
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This checks the QE input conditions for pw2qmcpack conversion. "
            "It does not prove that the SCF completed, that orbitals were "
            "written, or that a QMCPACK energy comparison is valid."
        ),
    }


def conversion_readiness(checks: list[dict[str, Any]]) -> str:
    """Reduce conversion check statuses using the public severity order."""
    statuses = {check["status"] for check in checks}
    return (
        "not_ready"
        if "not_ready" in statuses
        else "review_required"
        if "review_required" in statuses
        else "ready"
    )


def inspect_qe_qmcpack_electron_count(
    parsed_qe_output: dict[str, Any],
    qe_charge_spin: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    runtime_record = parsed_qe_output["system"].get("n_electrons")
    runtime_electrons = (
        runtime_record["value"]
        if isinstance(runtime_record, dict)
        and isinstance(runtime_record.get("value"), (int, float))
        else None
    )
    accounting = qe_charge_spin["electron_accounting"]
    input_electrons = (
        accounting["electron_count"]
        if accounting["status"] == "complete"
        else None
    )
    qmcpack_electrons = electron_particle_count(parsed_qmcpack, include_review)
    observed = {
        "qe_runtime_electrons": runtime_electrons,
        "qe_runtime_electron_line": (
            runtime_record.get("line") if isinstance(runtime_record, dict) else None
        ),
        "qe_input_valence_electrons": input_electrons,
        "qe_input_accounting_status": accounting["status"],
        "qmcpack": qmcpack_electrons,
    }
    if qmcpack_electrons["status"] != "complete":
        return {
            "name": "qe_qmcpack_electron_count",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QMCPACK electron-particle groups are not complete enough for an "
                "electron-count comparison."
            ),
        }
    qmc_count = qmcpack_electrons["electron_count"]
    qe_evidence = [
        value for value in (runtime_electrons, input_electrons) if value is not None
    ]
    if not qe_evidence:
        return {
            "name": "qe_qmcpack_electron_count",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QE provides no runtime electron count or complete UPF valence "
                "accounting for comparison with the QMCPACK deck."
            ),
        }
    if any(abs(value - qmc_count) > 1e-6 for value in qe_evidence):
        return {
            "name": "qe_qmcpack_electron_count",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QE electron-count evidence does not match the QMCPACK electron "
                "particle groups."
            ),
        }
    return {
        "name": "qe_qmcpack_electron_count",
        "status": "pass",
        "observed": observed,
        "message": (
            "QE electron-count evidence matches the QMCPACK electron-particle "
            "groups."
        ),
    }


def inspect_qe_qmcpack_atom_count(
    parsed_qe_input: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    qe_atoms = parsed_qe_input["system"].get("nat")
    qmcpack_atoms = non_electron_particle_sets(parsed_qmcpack, include_review)
    observed = {
        "qe_declared_atom_count": qe_atoms,
        "qmcpack": qmcpack_atoms,
    }
    if not isinstance(qe_atoms, int) or qmcpack_atoms["status"] != "complete":
        return {
            "name": "qe_qmcpack_atom_count",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QE or QMCPACK does not provide explicit particle-set sizes for "
                "an atom-count comparison."
            ),
        }
    qmcpack_atom_count = qmcpack_atoms["particle_count"]
    observed["qmcpack_declared_atom_count"] = qmcpack_atom_count
    if include_review["status"] == "incomplete":
        return {
            "name": "qe_qmcpack_atom_count",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK include graph is incomplete, so atom-count evidence "
                "may be missing."
            ),
        }
    if qe_atoms != qmcpack_atom_count:
        return {
            "name": "qe_qmcpack_atom_count",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QE's declared atom count does not match QMCPACK non-electron "
                "particle-set sizes."
            ),
        }
    return {
        "name": "qe_qmcpack_atom_count",
        "status": "pass",
        "observed": observed,
        "message": (
            "QE's declared atom count matches QMCPACK non-electron particle-set "
            "sizes."
        ),
    }


def inspect_qe_qmcpack_ion_species(
    parsed_qe_input: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    qe_atoms = parsed_qe_input.get("atomic_positions", {}).get("atoms", [])
    qe_elements = [
        element_from_label(str(atom.get("label") or ""))
        for atom in qe_atoms
        if isinstance(atom, dict)
    ]
    qmcpack_groups = _qmcpack_ion_groups(parsed_qmcpack, include_review)
    qmcpack_elements = [group["element"] for group in qmcpack_groups]
    observed = {
        "qe_atomic_elements": qe_elements,
        "qmcpack_ion_groups": qmcpack_groups,
        "include_review_status": include_review["status"],
    }
    if not qe_elements or any(element is None for element in qe_elements):
        return {
            "name": "qe_qmcpack_ion_species",
            "status": "review_required",
            "observed": observed,
            "message": "QE atomic-position labels cannot be normalized to elements.",
        }
    if (
        not qmcpack_groups
        or any(element is None for element in qmcpack_elements)
        or any(
            not isinstance(group["size"], str) or not group["size"].isdigit()
            for group in qmcpack_groups
        )
    ):
        return {
            "name": "qe_qmcpack_ion_species",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QMCPACK ion groups need normalized element labels and explicit "
                "sizes for a species comparison."
            ),
        }
    if include_review["status"] == "incomplete":
        return {
            "name": "qe_qmcpack_ion_species",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK include graph is incomplete, so ion-species "
                "evidence may be missing."
            ),
        }
    qe_counts = _element_counts(qe_elements)
    qmcpack_counts = _element_counts(
        element
        for group in qmcpack_groups
        for element in [group["element"]] * int(group["size"])
    )
    observed["qe_element_counts"] = qe_counts
    observed["qmcpack_element_counts"] = qmcpack_counts
    if qe_counts != qmcpack_counts:
        return {
            "name": "qe_qmcpack_ion_species",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QE atomic elements and QMCPACK ion-group elements do not "
                "have matching counts."
            ),
        }
    return {
        "name": "qe_qmcpack_ion_species",
        "status": "pass",
        "observed": observed,
        "message": "QE atomic elements match QMCPACK ion-group element counts.",
    }


def inspect_qmcpack_hdf5_deck_metadata(
    hdf5_inspection: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    """Compare recognized pw2qmcpack metadata with the QMCPACK XML deck."""
    status = hdf5_inspection.get("status")
    if status in {"unavailable", "incompatible", "tool_refused"}:
        return {
            "name": "qmcpack_hdf5_deck_metadata",
            "status": "not_applicable",
            "observed": {"hdf5_status": status},
            "message": "The optional companion HDF5 metadata inspection is unavailable.",
        }
    if status != "recognized" or hdf5_inspection.get("artifact_kind") != "pwscf_wavefunction":
        return {
            "name": "qmcpack_hdf5_deck_metadata",
            "status": "review_required",
            "observed": {
                "hdf5_status": status,
                "hdf5_artifact_kind": hdf5_inspection.get("artifact_kind"),
                "hdf5_message": hdf5_inspection.get("message"),
            },
            "message": (
                "The declared conversion artifact did not yield recognized "
                "pw2qmcpack wavefunction metadata for a deck cross-check."
            ),
        }

    wavefunction = hdf5_inspection.get("wavefunction")
    if not isinstance(wavefunction, dict):
        return {
            "name": "qmcpack_hdf5_deck_metadata",
            "status": "review_required",
            "observed": {"hdf5_status": status},
            "message": "Recognized wavefunction metadata is incomplete.",
        }
    hdf5_atoms = wavefunction.get("atoms")
    hdf5_electrons = wavefunction.get("electrons")
    if not isinstance(hdf5_atoms, dict) or not isinstance(hdf5_electrons, dict):
        return {
            "name": "qmcpack_hdf5_deck_metadata",
            "status": "review_required",
            "observed": {"hdf5_status": status},
            "message": "Recognized wavefunction atom or electron metadata is incomplete.",
        }

    qmcpack_atoms = non_electron_particle_sets(parsed_qmcpack, include_review)
    qmcpack_electrons = electron_particle_count(parsed_qmcpack, include_review)
    qmcpack_spin = electron_spin_population(parsed_qmcpack, include_review)
    hdf5_species = hdf5_atoms.get("species")
    hdf5_counts = _hdf5_species_counts(hdf5_species)
    qmcpack_groups = _qmcpack_ion_groups(parsed_qmcpack, include_review)
    qmcpack_counts = _qmcpack_group_element_counts(qmcpack_groups)
    spin_populations = hdf5_electrons.get("spin_populations")
    hdf5_total_electrons = (
        sum(spin_populations)
        if isinstance(spin_populations, list)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in spin_populations)
        else None
    )
    observed = {
        "hdf5": {
            "source": hdf5_inspection.get("source"),
            "atom_count": hdf5_atoms.get("count"),
            "species_counts": hdf5_counts,
            "spin_populations": spin_populations,
            "spin_count": hdf5_electrons.get("spin_count"),
            "total_electrons": hdf5_total_electrons,
        },
        "qmcpack": {
            "atoms": qmcpack_atoms,
            "species_counts": qmcpack_counts,
            "electrons": qmcpack_electrons,
            "spin": qmcpack_spin,
        },
        "include_review_status": include_review["status"],
    }
    missing = []
    mismatches = []
    hdf5_atom_count = hdf5_atoms.get("count")
    if not isinstance(hdf5_atom_count, int) or qmcpack_atoms["status"] != "complete":
        missing.append("atom_count")
    elif hdf5_atom_count != qmcpack_atoms["particle_count"]:
        mismatches.append("atom_count")
    if hdf5_counts is None or qmcpack_counts is None:
        missing.append("species_counts")
    elif hdf5_counts != qmcpack_counts:
        mismatches.append("species_counts")
    if hdf5_total_electrons is None or qmcpack_electrons["status"] != "complete":
        missing.append("electron_count")
    elif hdf5_total_electrons != qmcpack_electrons["electron_count"]:
        mismatches.append("electron_count")
    if isinstance(spin_populations, list) and len(spin_populations) == 2:
        if qmcpack_spin["status"] != "complete":
            missing.append("spin_populations")
        elif spin_populations != [
            qmcpack_spin["up_electrons"],
            qmcpack_spin["down_electrons"],
        ]:
            mismatches.append("spin_populations")

    if mismatches:
        return {
            "name": "qmcpack_hdf5_deck_metadata",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "pw2qmcpack HDF5 metadata does not match the QMCPACK deck for "
                f"{', '.join(mismatches)}."
            ),
        }
    if missing or include_review["status"] == "incomplete":
        return {
            "name": "qmcpack_hdf5_deck_metadata",
            "status": "review_required",
            "observed": observed,
            "message": (
                "pw2qmcpack HDF5 metadata is recognized, but the QMCPACK deck "
                "does not provide enough complete evidence for every cross-check."
            ),
        }
    return {
        "name": "qmcpack_hdf5_deck_metadata",
        "status": "pass",
        "observed": observed,
        "message": (
            "pw2qmcpack HDF5 atom, species, electron, and spin metadata matches "
            "the QMCPACK deck."
        ),
    }


def inspect_qe_qmcpack_collinear_spin(
    qe_charge_spin: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    qe_spin = qe_charge_spin["spin"]
    qmcpack_spin = electron_spin_population(parsed_qmcpack, include_review)
    observed = {
        "qe_spin": qe_spin,
        "qmcpack_spin": qmcpack_spin,
    }
    if qe_spin["mode"] != "collinear":
        return {
            "name": "qe_qmcpack_collinear_spin",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QE-to-QMCPACK spin comparison supports only collinear nspin=2 "
                "inputs."
            ),
        }
    qe_magnetization = qe_spin["tot_magnetization"]
    if qe_magnetization is None:
        return {
            "name": "qe_qmcpack_collinear_spin",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QE does not set an explicit fixed total magnetization for a "
                "collinear spin comparison."
            ),
        }
    if qmcpack_spin["status"] != "complete":
        return {
            "name": "qe_qmcpack_collinear_spin",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QMCPACK does not provide exactly one selected electron "
                "particle set with explicit u and d groups."
            ),
        }
    qmcpack_imbalance = qmcpack_spin["spin_imbalance"]
    observed["qe_fixed_magnetization"] = qe_magnetization
    observed["qmcpack_spin_imbalance"] = qmcpack_imbalance
    if abs(qe_magnetization - qmcpack_imbalance) > 1e-6:
        return {
            "name": "qe_qmcpack_collinear_spin",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QE fixed total magnetization does not match the QMCPACK u-d "
                "electron imbalance."
            ),
        }
    return {
        "name": "qe_qmcpack_collinear_spin",
        "status": "pass",
        "observed": observed,
        "message": (
            "QE fixed total magnetization matches the QMCPACK u-d electron "
            "imbalance."
        ),
    }


def inspect_qe_qmcpack_charge(
    qe_charge_spin: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    qe_accounting = qe_charge_spin["electron_accounting"]
    qmcpack_electrons = electron_particle_count(parsed_qmcpack, include_review)
    qmcpack_ions = _qmcpack_ion_valence(parsed_qmcpack, include_review)
    observed = {
        "qe_charge": qe_charge_spin["charge"],
        "qe_electron_accounting": qe_accounting,
        "qmcpack_electrons": qmcpack_electrons,
        "qmcpack_ions": qmcpack_ions,
        "include_review_status": include_review["status"],
    }
    if qe_accounting["status"] != "complete":
        return {
            "name": "qe_qmcpack_charge",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QE UPF valence accounting is incomplete for a charge comparison."
            ),
        }
    if qmcpack_electrons["status"] != "complete" or qmcpack_ions["status"] != "complete":
        return {
            "name": "qe_qmcpack_charge",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QMCPACK electron or ion-valence evidence is incomplete for a "
                "charge comparison."
            ),
        }
    if include_review["status"] == "incomplete":
        return {
            "name": "qe_qmcpack_charge",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK include graph is incomplete, so charge evidence "
                "may be missing."
            ),
        }
    qe_valence = qe_accounting["valence_electrons_before_charge"]
    qmcpack_valence = qmcpack_ions["valence_electrons"]
    qmcpack_charge = qmcpack_valence - qmcpack_electrons["electron_count"]
    qe_charge = qe_charge_spin["charge"]["tot_charge"]
    observed["qe_valence_electrons"] = qe_valence
    observed["qmcpack_valence_electrons"] = qmcpack_valence
    observed["qmcpack_net_charge"] = qmcpack_charge
    if abs(qe_valence - qmcpack_valence) > 1e-6:
        return {
            "name": "qe_qmcpack_charge",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QE UPF valence accounting does not match QMCPACK ion valence "
                "parameters."
            ),
        }
    if abs(qe_charge - qmcpack_charge) > 1e-6:
        return {
            "name": "qe_qmcpack_charge",
            "status": "not_ready",
            "observed": observed,
            "message": "QE total charge does not match the QMCPACK net charge.",
        }
    return {
        "name": "qe_qmcpack_charge",
        "status": "pass",
        "observed": observed,
        "message": "QE total charge matches the QMCPACK net charge.",
    }


def inspect_qe_qmcpack_pseudopotential_species(
    parsed_qe_input: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
    qmcpack_input: str | Path,
) -> dict[str, Any]:
    qe_species = [{
        "label": species.get("label"),
        "element": element_from_label(str(species.get("label") or "")),
    } for species in parsed_qe_input.get("atomic_species", [])]
    qmcpack_references = collect_pseudopotential_references(
        parsed_qmcpack,
        include_review,
        qmcpack_input,
    )
    qmcpack_species = [{
        "element": element_from_label(str(reference.get("element") or "")),
        "reference": reference,
    } for entry in qmcpack_references for reference in entry["references"]]
    observed = {
        "qe_atomic_species": qe_species,
        "qmcpack_pseudopotential_species": qmcpack_species,
        "include_review_status": include_review["status"],
    }
    qe_elements = {species["element"] for species in qe_species if species["element"]}
    qmcpack_elements = {
        species["element"] for species in qmcpack_species if species["element"]
    }
    unresolved = [
        *[species for species in qe_species if species["element"] is None],
        *[species for species in qmcpack_species if species["element"] is None],
    ]
    if not qmcpack_species:
        return {
            "name": "qe_qmcpack_pseudopotential_species",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK deck declares no pseudopotential elementType values; "
                "confirm whether an all-electron Hamiltonian is intended."
            ),
        }
    if unresolved:
        return {
            "name": "qe_qmcpack_pseudopotential_species",
            "status": "review_required",
            "observed": {**observed, "unresolved_species": unresolved},
            "message": (
                "At least one QE species label or QMCPACK pseudopotential "
                "elementType cannot be normalized to an element."
            ),
        }
    extra_qmcpack_elements = sorted(qmcpack_elements - qe_elements)
    missing_qmcpack_elements = sorted(qe_elements - qmcpack_elements)
    observed["qe_elements"] = sorted(qe_elements)
    observed["qmcpack_elements"] = sorted(qmcpack_elements)
    observed["missing_qmcpack_elements"] = missing_qmcpack_elements
    observed["extra_qmcpack_elements"] = extra_qmcpack_elements
    if extra_qmcpack_elements:
        return {
            "name": "qe_qmcpack_pseudopotential_species",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QMCPACK declares a pseudopotential element absent from the QE "
                "input species list."
            ),
        }
    if include_review["status"] == "incomplete":
        return {
            "name": "qe_qmcpack_pseudopotential_species",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK include graph is incomplete, so pseudopotential "
                "species evidence may be missing."
            ),
        }
    if missing_qmcpack_elements:
        return {
            "name": "qe_qmcpack_pseudopotential_species",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QMCPACK lacks a declared pseudopotential element for at least "
                "one QE input species."
            ),
        }
    return {
        "name": "qe_qmcpack_pseudopotential_species",
        "status": "pass",
        "observed": observed,
        "message": (
            "QE atomic-species elements match the QMCPACK pseudopotential "
            "elementType declarations."
        ),
    }


def inspect_qe_qmcpack_pseudopotential_valence(
    qe_pseudopotentials: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
    qmcpack_input: str | Path,
) -> dict[str, Any]:
    qe_entries = [
        {
            "species_label": entry.get("species_label"),
            "element": element_from_label(str(entry.get("species_label") or "")),
            "status": entry.get("status"),
            "z_valence": (
                entry["upf"].get("z_valence")
                if isinstance(entry.get("upf"), dict)
                else None
            ),
        }
        for entry in qe_pseudopotentials.get("entries", [])
        if isinstance(entry, dict)
    ]
    qmcpack_entries = []
    for reference in collect_pseudopotential_references(
        parsed_qmcpack,
        include_review,
        qmcpack_input,
    ):
        try:
            inspection = inspect_qmcpack_pseudopotential(reference["path"])
        except (OSError, ValueError) as error:
            qmcpack_entries.append({
                **reference,
                "status": "not_ready",
                "error": str(error),
            })
            continue
        header = inspection["header"]
        qmcpack_entries.append({
            **reference,
            "status": "parsed",
            "header_element": element_from_label(str(header.get("symbol") or "")),
            "zval": header["zval"],
        })

    observed = {
        "qe_pseudopotentials": qe_entries,
        "qmcpack_pseudopotentials": qmcpack_entries,
        "include_review_status": include_review["status"],
    }
    if not qe_entries or qe_pseudopotentials.get("status") != "parsed":
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QE UPF headers are not complete enough for a valence comparison."
            ),
        }
    if not qmcpack_entries:
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK deck declares no pseudopotential XML files for a "
                "valence comparison."
            ),
        }
    if any(entry["status"] != "parsed" for entry in qmcpack_entries):
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "At least one declared QMCPACK pseudopotential cannot be parsed "
                "for its header valence."
            ),
        }

    qe_by_element = _single_valence_by_element(qe_entries, "element", "z_valence")
    qmcpack_by_element = _qmcpack_valence_by_element(qmcpack_entries)
    observed["qe_valence_by_element"] = qe_by_element
    observed["qmcpack_valence_by_element"] = qmcpack_by_element
    if qe_by_element is None or qmcpack_by_element is None:
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The declared pseudopotentials do not provide one unambiguous "
                "valence value for each element."
            ),
        }

    qe_elements = set(qe_by_element)
    qmcpack_elements = set(qmcpack_by_element)
    extra_qmcpack_elements = sorted(qmcpack_elements - qe_elements)
    missing_qmcpack_elements = sorted(qe_elements - qmcpack_elements)
    observed["missing_qmcpack_elements"] = missing_qmcpack_elements
    observed["extra_qmcpack_elements"] = extra_qmcpack_elements
    if extra_qmcpack_elements:
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "not_ready",
            "observed": observed,
            "message": "QMCPACK declares a pseudopotential element absent from QE.",
        }
    if include_review["status"] == "incomplete":
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK include graph is incomplete, so valence evidence "
                "may be missing."
            ),
        }
    if missing_qmcpack_elements:
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "not_ready",
            "observed": observed,
            "message": "QMCPACK lacks valence evidence for at least one QE element.",
        }
    mismatches = {
        element: {"qe_z_valence": qe_by_element[element], "qmcpack_zval": qmcpack_by_element[element]}
        for element in sorted(qe_elements)
        if abs(qe_by_element[element] - qmcpack_by_element[element]) > 1e-6
    }
    observed["mismatches"] = mismatches
    if mismatches:
        return {
            "name": "qe_qmcpack_pseudopotential_valence",
            "status": "not_ready",
            "observed": observed,
            "message": "QE UPF z_valence does not match the QMCPACK XML zval.",
        }
    return {
        "name": "qe_qmcpack_pseudopotential_valence",
        "status": "pass",
        "observed": observed,
        "message": "QE UPF z_valence matches the QMCPACK XML zval for every element.",
    }


def inspect_qe_qmcpack_projector_evidence(
    qe_pseudopotentials: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
    qmcpack_input: str | Path,
) -> dict[str, Any]:
    dmc_blocks = _dmc_qmc_blocks(
        parsed_qmcpack,
        include_review,
        Path(qmcpack_input).expanduser().resolve(),
    )
    entries = [
        {
            "species_label": entry.get("species_label"),
            "path": entry.get("path"),
            "status": entry.get("status"),
            "pseudo_type": (
                entry["upf"].get("pseudo_type")
                if isinstance(entry.get("upf"), dict)
                else None
            ),
            "projector_channel_evidence": (
                entry["upf"].get("projector_channel_evidence")
                if isinstance(entry.get("upf"), dict)
                else None
            ),
        }
        for entry in qe_pseudopotentials.get("entries", [])
        if isinstance(entry, dict)
    ]
    observed = {
        "qmcpack_dmc_blocks": dmc_blocks,
        "qe_pseudopotentials": entries,
        "include_review_status": include_review["status"],
    }
    if not dmc_blocks:
        if include_review["status"] == "incomplete":
            return {
                "name": "qe_qmcpack_projector_evidence",
                "status": "review_required",
                "observed": observed,
                "message": (
                    "No DMC block was found in the primary QMCPACK XML or reviewed "
                    "includes, but the incomplete include graph may omit controls."
                ),
            }
        return {
            "name": "qe_qmcpack_projector_evidence",
            "status": "not_applicable",
            "observed": observed,
            "message": "No DMC block was found in the primary QMCPACK XML or reviewed includes.",
        }
    if not entries or qe_pseudopotentials.get("status") != "parsed":
        return {
            "name": "qe_qmcpack_projector_evidence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QE UPF headers are not complete enough to review projector "
                "evidence for the declared DMC input."
            ),
        }
    non_semilocal_sources = [
        {
            "species_label": entry["species_label"],
            "path": entry["path"],
            "pseudo_type": entry["pseudo_type"],
        }
        for entry in entries
        if not isinstance(entry["pseudo_type"], str)
        or entry["pseudo_type"].casefold() not in {"nc", "sl"}
    ]
    observed["non_semilocal_qe_pseudopotentials"] = non_semilocal_sources
    if non_semilocal_sources:
        return {
            "name": "qe_qmcpack_projector_evidence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "At least one QE source UPF is not declared NC or semilocal; "
                "confirm that QMCPACK DMC uses a separately generated semilocal "
                "potential."
            ),
        }
    incomplete_entries = [
        entry
        for entry in entries
        if entry["pseudo_type"].casefold() == "nc"
        and (
            not isinstance(entry["projector_channel_evidence"], dict)
            or entry["projector_channel_evidence"].get("status") != "complete"
        )
    ]
    observed["incomplete_qe_pseudopotentials"] = incomplete_entries
    if incomplete_entries:
        return {
            "name": "qe_qmcpack_projector_evidence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "Bounded QE UPF evidence does not establish projector counts for "
                "every nonlocal NC species used by the declared DMC input."
            ),
        }
    multi_projector_channels = [
        {
            "species_label": entry["species_label"],
            "path": entry["path"],
            "counts_by_angular_momentum": {
                angular_momentum: count
                for angular_momentum, count in entry[
                    "projector_channel_evidence"
                ]["counts_by_angular_momentum"].items()
                if count > 1
            },
        }
        for entry in entries
        if any(
            count > 1
            for count in entry["projector_channel_evidence"][
                "counts_by_angular_momentum"
            ].values()
        )
    ]
    observed["multi_projector_channels"] = multi_projector_channels
    if multi_projector_channels:
        return {
            "name": "qe_qmcpack_projector_evidence",
            "status": "review_required",
            "observed": observed,
            "message": (
                "At least one QE UPF has multiple projectors in an angular channel; "
                "confirm that QMCPACK DMC uses a separately generated semilocal "
                "potential rather than a reconstructed projector form."
            ),
        }
    return {
        "name": "qe_qmcpack_projector_evidence",
        "status": "pass",
        "observed": observed,
        "message": (
            "The bounded QE UPF evidence has no multi-projector angular channel "
            "for the declared DMC input."
        ),
    }


def _dmc_qmc_blocks(
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
    primary_input: Path,
) -> list[dict[str, Any]]:
    blocks = [
        {"source_path": str(primary_input), "qmc_block_index": index}
        for index, block in enumerate(parsed_qmcpack.get("qmc_blocks", []))
        if isinstance(block, dict) and block.get("method") == "dmc"
    ]
    for entry in include_review.get("entries", []):
        if (
            not isinstance(entry, dict)
            or entry.get("status") != "present"
            or entry.get("scan_status") == "too_large"
        ):
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        try:
            root = ElementTree.parse(path).getroot()
        except (ElementTree.ParseError, OSError):
            continue
        qmc_elements = [
            item for item in root.iter() if item.tag.rsplit("}", 1)[-1] == "qmc"
        ]
        blocks.extend(
            {"source_path": path, "qmc_block_index": index}
            for index, element in enumerate(qmc_elements)
            if element.get("method") == "dmc"
        )
    return blocks


def inspect_qe_qmcpack_geometry(
    parsed_qe_input: dict[str, Any],
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    qe_geometry = normalize_pw_input_geometry(parsed_qe_input)
    targets = sorted({
        hamiltonian["target"]
        for hamiltonian in parsed_qmcpack.get("hamiltonians", [])
        if isinstance(hamiltonian, dict) and hamiltonian.get("target")
    })
    geometries = collect_ion_geometries(parsed_qmcpack, include_review)
    candidates = [
        geometry
        for geometry in geometries
        if geometry.get("particle_set") not in targets
    ]
    observed = {
        "qe_geometry": qe_geometry,
        "hamiltonian_targets": targets,
        "qmcpack_ion_geometries": candidates,
        "include_review_status": include_review["status"],
    }
    if qe_geometry["status"] != "available":
        return {
            "name": "qe_qmcpack_geometry",
            "status": "review_required",
            "observed": observed,
            "message": "QE input geometry cannot be normalized for comparison.",
        }
    if len(targets) != 1 or len(candidates) != 1:
        return {
            "name": "qe_qmcpack_geometry",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QMCPACK does not provide exactly one non-electron particle "
                "geometry selected against one Hamiltonian target."
            ),
        }
    qmcpack_geometry = candidates[0]
    if qmcpack_geometry.get("status") != "complete":
        return {
            "name": "qe_qmcpack_geometry",
            "status": "review_required",
            "observed": observed,
            "message": "QMCPACK ion geometry is incomplete.",
        }
    if include_review["status"] == "incomplete":
        return {
            "name": "qe_qmcpack_geometry",
            "status": "review_required",
            "observed": observed,
            "message": (
                "The QMCPACK include graph is incomplete, so geometry evidence "
                "may be missing."
            ),
        }
    cell = qmcpack_geometry["cell"]
    lattice = cell["lattice"] if isinstance(cell, dict) else None
    if (
        not isinstance(lattice, dict)
        or lattice.get("units") != "bohr"
        or cell.get("boundary_conditions") != ["p", "p", "p"]
    ):
        return {
            "name": "qe_qmcpack_geometry",
            "status": "review_required",
            "observed": observed,
            "message": (
                "QMCPACK geometry comparison requires an explicit bohr lattice "
                "with p p p boundary conditions."
            ),
        }
    qmcpack_cell = np.asarray(lattice["vectors"], dtype=float) * ANGSTROM_PER_BOHR
    qe_cell = np.asarray(qe_geometry["cell_vectors_angstrom"], dtype=float)
    if qmcpack_cell.shape != (3, 3) or qe_cell.shape != (3, 3):
        return {
            "name": "qe_qmcpack_geometry",
            "status": "review_required",
            "observed": observed,
            "message": "QE or QMCPACK does not provide a complete three-vector cell.",
        }
    qe_cell_volume = _periodic_cell_volume(qe_cell)
    qmcpack_cell_volume = _periodic_cell_volume(qmcpack_cell)
    observed["qe_cell_volume_angstrom3"] = qe_cell_volume
    observed["qmcpack_cell_volume_angstrom3"] = qmcpack_cell_volume
    if qe_cell_volume is None or qmcpack_cell_volume is None:
        return {
            "name": "qe_qmcpack_geometry",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QE or QMCPACK provides a non-finite or singular periodic cell."
            ),
        }
    cell_matches = bool(np.allclose(qe_cell, qmcpack_cell, rtol=0.0, atol=1e-6))
    observed["cell_matches"] = cell_matches
    if not cell_matches:
        return {
            "name": "qe_qmcpack_geometry",
            "status": "not_ready",
            "observed": observed,
            "message": "QE and QMCPACK cells do not match within 1e-6 angstrom.",
        }
    qmcpack_atoms = [{
        "element": element_from_label(str(atom.get("label") or "")),
        "coordinates_angstrom": np.asarray(atom["coordinates"], dtype=float)
        * ANGSTROM_PER_BOHR,
    } for atom in qmcpack_geometry["atoms"]]
    if any(atom["element"] is None for atom in qmcpack_atoms):
        return {
            "name": "qe_qmcpack_geometry",
            "status": "review_required",
            "observed": observed,
            "message": "A QMCPACK ion group name cannot be normalized to an element.",
        }
    qe_signature = _periodic_atom_signature(qe_geometry["atoms"], qe_cell)
    qmcpack_signature = _periodic_atom_signature(qmcpack_atoms, qmcpack_cell)
    observed["qe_periodic_atom_signature"] = qe_signature
    observed["qmcpack_periodic_atom_signature"] = qmcpack_signature
    if qe_signature != qmcpack_signature:
        return {
            "name": "qe_qmcpack_geometry",
            "status": "not_ready",
            "observed": observed,
            "message": (
                "QE and QMCPACK atom elements or positions do not match modulo "
                "periodic lattice translations."
            ),
        }
    return {
        "name": "qe_qmcpack_geometry",
        "status": "pass",
        "observed": observed,
        "message": (
            "QE and QMCPACK cells, atom elements, and positions agree modulo "
            "periodic lattice translations."
        ),
    }


def _periodic_atom_signature(
    atoms: list[dict[str, Any]],
    cell: np.ndarray,
) -> list[tuple[str, int, int, int]]:
    inverse = np.linalg.inv(cell)
    signature = []
    for atom in atoms:
        if "coordinates_angstrom" in atom:
            coordinates = atom["coordinates_angstrom"]
        else:
            coordinates = np.asarray([atom["x"], atom["y"], atom["z"]])
        fractional = np.mod(np.asarray(coordinates) @ inverse, 1.0)
        signature.append((
            str(atom["element"]),
            *(int(round(value * 10_000_000)) % 10_000_000 for value in fractional),
        ))
    return sorted(signature)


def _periodic_cell_volume(cell: np.ndarray) -> float | None:
    if not np.isfinite(cell).all():
        return None
    volume = abs(float(np.linalg.det(cell)))
    return volume if np.isfinite(volume) and volume > 1e-12 else None


def _qmcpack_ion_valence(
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> dict[str, Any]:
    groups = _qmcpack_ion_groups(parsed_qmcpack, include_review)
    if (
        not groups
        or any(
            not isinstance(group["size"], str)
            or not group["size"].isdigit()
            or not isinstance(group["valence"], float)
            or not isfinite(group["valence"])
            for group in groups
        )
    ):
        return {"status": "incomplete", "groups": groups}
    return {
        "status": "complete",
        "groups": groups,
        "valence_electrons": sum(
            int(group["size"]) * group["valence"] for group in groups
        ),
    }


def _qmcpack_ion_groups(
    parsed_qmcpack: dict[str, Any],
    include_review: dict[str, Any],
) -> list[dict[str, Any]]:
    targets = {
        hamiltonian["target"]
        for hamiltonian in parsed_qmcpack.get("hamiltonians", [])
        if isinstance(hamiltonian, dict) and hamiltonian.get("target")
    }
    groups = []
    for particle_set in collect_particle_sets(parsed_qmcpack, include_review):
        if particle_set.get("name") in targets:
            continue
        particle_groups = particle_set.get("groups", [])
        for group in particle_groups:
            size = group.get("size")
            if size is None and len(particle_groups) == 1:
                size = particle_set.get("size")
            parameters = group.get("parameters")
            valence_text = parameters.get("valence") if isinstance(parameters, dict) else None
            try:
                valence = float(valence_text)
            except (TypeError, ValueError):
                valence = None
            groups.append({
                "particle_set": particle_set.get("name"),
                "label": group.get("name"),
                "element": element_from_label(str(group.get("name") or "")),
                "size": size,
                "valence": valence,
            })
    return groups


def _hdf5_species_counts(species: Any) -> dict[str, int] | None:
    if not isinstance(species, list) or not species:
        return None
    counts: dict[str, int] = {}
    for entry in species:
        if not isinstance(entry, dict):
            return None
        element = element_from_label(str(entry.get("name") or ""))
        atom_count = entry.get("atom_count")
        if element is None or not isinstance(atom_count, int) or atom_count < 0:
            return None
        counts[element] = counts.get(element, 0) + atom_count
    return dict(sorted(counts.items()))


def _qmcpack_group_element_counts(
    groups: list[dict[str, Any]],
) -> dict[str, int] | None:
    if not groups:
        return None
    counts: dict[str, int] = {}
    for group in groups:
        element = group.get("element")
        size = group.get("size")
        if (
            not isinstance(element, str)
            or not isinstance(size, str)
            or not size.isdigit()
        ):
            return None
        counts[element] = counts.get(element, 0) + int(size)
    return dict(sorted(counts.items()))


def _element_counts(elements: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        counts[element] = counts.get(element, 0) + 1
    return dict(sorted(counts.items()))


def _single_valence_by_element(
    entries: list[dict[str, Any]],
    element_key: str,
    valence_key: str,
) -> dict[str, float] | None:
    values: dict[str, set[float]] = {}
    for entry in entries:
        element = entry.get(element_key)
        valence = entry.get(valence_key)
        if not isinstance(element, str) or not isinstance(valence, (int, float)):
            return None
        values.setdefault(element, set()).add(float(valence))
    if any(len(element_values) != 1 for element_values in values.values()):
        return None
    return {element: next(iter(element_values)) for element, element_values in values.items()}


def _qmcpack_valence_by_element(
    entries: list[dict[str, Any]],
) -> dict[str, float] | None:
    values: dict[str, set[float]] = {}
    for entry in entries:
        header_element = entry.get("header_element")
        valence = entry.get("zval")
        references = entry.get("references")
        if (
            not isinstance(header_element, str)
            or not isinstance(valence, (int, float))
            or not isinstance(references, list)
        ):
            return None
        for reference in references:
            declared_element = element_from_label(str(reference.get("element") or ""))
            if declared_element != header_element:
                return None
            values.setdefault(declared_element, set()).add(float(valence))
    if any(len(element_values) != 1 for element_values in values.values()):
        return None
    return {element: next(iter(element_values)) for element, element_values in values.items()}


__all__ = [
    "inspect_conversion_calculation",
    "inspect_conversion_disk_io",
    "inspect_conversion_isolation",
    "inspect_conversion_k_points",
    "inspect_qe_pw2qmcpack_control_paths",
    "inspect_pwscf_h5_artifact",
    "inspect_qe_scf_completion",
    "inspect_qe_qmcpack_atom_count",
    "inspect_qe_qmcpack_charge",
    "inspect_qe_qmcpack_collinear_spin",
    "inspect_qe_qmcpack_electron_count",
    "inspect_qe_qmcpack_pseudopotential_species",
    "inspect_qe_qmcpack_pseudopotential_valence",
    "inspect_qe_qmcpack_projector_evidence",
    "inspect_qe_qmcpack_geometry",
    "inspect_qmcpack_hdf5_deck_metadata",
    "inspect_qe_qmcpack_ion_species",
]
