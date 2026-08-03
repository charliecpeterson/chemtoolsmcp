"""Build an explicitly incomplete PySCF comparison reference from NWChem evidence.

The adapter keeps NWChem facts separate from caller-declared PySCF settings.
It prepares a draft for ``compare_pyscf_reference_calculation`` but never
silently fills values whose NWChem meaning is not equivalent to PySCF's.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from chemtools.core.units import ANGSTROM_PER_BOHR
from chemtools.programs.nwchem.parse.input import (
    extract_nwchem_geometry_block,
    extract_nwchem_module_block,
    inspect_all_nwchem_basis_blocks,
    inspect_nwchem_input,
)
from chemtools.programs.nwchem.strategy.diagnose import parse_scf


NWCHEM_PYSCF_REFERENCE_DRAFT_SCHEMA = "chemtools.nwchem-pyscf-reference-draft/1"
_GEOMETRY_UNITS_RE = re.compile(
    r"\bunits\s+(angstroms?|au|a\.u\.?|bohrs?)\b",
    re.IGNORECASE,
)
_XC_RE = re.compile(r"^\s*xc\s+(.+?)\s*$", re.IGNORECASE)
_LIBRARY_RE = re.compile(
    r"^\s*[A-Za-z*][A-Za-z0-9*]*\s+library\s+(\S+)",
    re.IGNORECASE,
)
_PYSCF_METHODS = {"rhf", "uhf", "rks", "uks"}


def draft_nwchem_pyscf_reference(
    input_path: str,
    *,
    output_path: str | None = None,
    label: str | None = None,
    pyscf_method: str | None = None,
    pyscf_xc: str | None = None,
    density_fit: bool | None = None,
    electron_total: int | None = None,
) -> dict[str, Any]:
    """Prepare a comparison reference draft from one NWChem calculation.

    PySCF settings are caller declarations because generic NWChem text cannot
    safely establish their semantic equivalence. The result is only
    comparison-ready when every required field has direct evidence or a caller
    declaration.
    """
    input_summary = inspect_nwchem_input(input_path)
    geometry, geometry_evidence = _geometry_from_input(input_path, input_summary)
    basis, basis_evidence = _basis_from_input(input_path)
    nwchem_method = _last_electronic_task(input_summary)
    nwchem_xc, nwchem_xc_evidence = _xc_from_input(input_path, nwchem_method)
    output_evidence, scf_converged, total_hartree = _scf_from_output(output_path)

    declared_method = _declared_pyscf_method(pyscf_method)
    declared_xc = _declared_pyscf_xc(pyscf_xc, declared_method)
    declared_density_fit = _declared_density_fit(density_fit)
    declared_electrons = _declared_electron_total(electron_total)
    reference = {
        "label": label or f"NWChem reference: {Path(input_path).name}",
        "geometry": geometry,
        "calculation": {
            "method": declared_method,
            "basis": basis,
            "xc": declared_xc,
            "density_fit": declared_density_fit,
            "charge": input_summary["charge"],
            "multiplicity": input_summary["multiplicity"],
        },
        "scf": {"converged": scf_converged},
        "energy": {"total_hartree": total_hartree},
        "electrons": {"total": declared_electrons},
    }
    field_sources = {
        "geometry": geometry_evidence,
        "calculation.method": {
            "status": "caller_declared" if declared_method else "missing",
            "value": declared_method,
            "nwchem_task_module": nwchem_method,
            "reason": (
                "NWChem task modules do not identify the corresponding "
                "PySCF SCF flavour."
            ),
        },
        "calculation.basis": basis_evidence,
        "calculation.xc": {
            "status": (
                "caller_declared"
                if declared_xc is not None
                else "not_applicable"
                if declared_method in {"rhf", "uhf"}
                else "missing"
            ),
            "value": declared_xc,
            "nwchem_xc": nwchem_xc,
            "nwchem_evidence": nwchem_xc_evidence,
            "reason": (
                "The NWChem xc declaration is retained as evidence but is "
                "not treated as a PySCF functional-equivalence mapping."
            ),
        },
        "calculation.density_fit": {
            "status": (
                "caller_declared" if declared_density_fit is not None else "missing"
            ),
            "value": declared_density_fit,
            "reason": (
                "NWChem integral and fitting directives are not a safe "
                "equivalence mapping to PySCF density_fit."
            ),
        },
        "calculation.charge": _input_field_source(input_summary, "charge"),
        "calculation.multiplicity": _input_field_source(
            input_summary,
            "multiplicity",
        ),
        "scf.converged": output_evidence["scf_converged"],
        "energy.total_hartree": output_evidence["total_hartree"],
        "electrons.total": {
            "status": "caller_declared" if declared_electrons is not None else "missing",
            "value": declared_electrons,
            "reason": (
                "Electron count is not inferred because ECPs and nonstandard "
                "center charges can change the effective electron count."
            ),
        },
    }
    missing_required_fields = [
        field_name
        for field_name, value in _required_values(reference).items()
        if value is None
    ]
    return {
        "schema_version": NWCHEM_PYSCF_REFERENCE_DRAFT_SCHEMA,
        "status": "drafted",
        "comparison_ready": not missing_required_fields,
        "missing_required_fields": missing_required_fields,
        "reference_draft": reference,
        "field_sources": field_sources,
        "evidence": {
            "input": {
                "path": input_summary["file"],
                "tasks": input_summary["tasks"],
                "geometry_block_count": input_summary["geometry_block_count"],
            },
            "output": output_evidence["output"],
        },
        "cautions": _cautions(
            nwchem_method=nwchem_method,
            geometry_evidence=geometry_evidence,
            basis_evidence=basis_evidence,
            output_path=output_path,
        ),
    }


def _geometry_from_input(
    input_path: str,
    input_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    if input_summary["geometry_block_count"] != 1:
        return None, {
            "status": "missing",
            "reason": "The input must contain exactly one Cartesian geometry block.",
        }
    try:
        geometry = extract_nwchem_geometry_block(input_path)
    except ValueError as exc:
        return None, {"status": "missing", "reason": str(exc)}
    units = _geometry_units(geometry["header_line"])
    if units is None:
        return None, {
            "status": "missing",
            "reason": "The Cartesian geometry block does not declare coordinate units.",
        }
    factor = ANGSTROM_PER_BOHR if units == "bohr" else 1.0
    atoms = [
        {
            "element": atom["element"],
            "x": atom["x"] * factor,
            "y": atom["y"] * factor,
            "z": atom["z"] * factor,
        }
        for atom in geometry["atoms"]
    ]
    return atoms, {
        "status": "extracted",
        "path": geometry["file"],
        "block_index": geometry["block_index"],
        "source_units": units,
        "normalized_units": "angstrom",
        "atom_count": geometry["atom_count"],
    }


def _basis_from_input(input_path: str) -> tuple[str | None, dict[str, Any]]:
    blocks = inspect_all_nwchem_basis_blocks(input_path)
    if not blocks:
        return None, {"status": "missing", "reason": "No NWChem basis block was found."}
    if any(block["has_manual_content"] for block in blocks):
        return None, {
            "status": "missing",
            "reason": "A manual basis definition cannot be represented as one PySCF basis name.",
        }
    names = {
        match.group(1)
        for block in blocks
        for line in block["body_lines"]
        if (match := _LIBRARY_RE.match(line.split("#", 1)[0]))
    }
    if len(names) != 1:
        return None, {
            "status": "missing",
            "reason": "The input does not declare one unambiguous library basis name.",
            "basis_names": sorted(names),
        }
    basis = names.pop()
    return basis, {
        "status": "extracted",
        "value": basis,
        "block_count": len(blocks),
        "reason": "One library basis name was used across all parsed basis blocks.",
    }


def _last_electronic_task(input_summary: dict[str, Any]) -> str | None:
    modules = [
        task["module"].lower()
        for task in input_summary["tasks"]
        if task["module"].lower() in {"scf", "dft"}
    ]
    return modules[-1] if modules else None


def _xc_from_input(
    input_path: str,
    nwchem_method: str | None,
) -> tuple[str | None, dict[str, Any]]:
    if nwchem_method != "dft":
        return None, {
            "status": "not_applicable",
            "value": None,
            "reason": "The selected NWChem task is not DFT.",
        }
    try:
        block = extract_nwchem_module_block(input_path, module="dft")
    except ValueError:
        return None, {
            "status": "missing",
            "reason": "The selected DFT task has no readable DFT block.",
        }
    declarations = [
        match.group(1).split("#", 1)[0].strip()
        for line in block["body_lines"]
        if (match := _XC_RE.match(line))
    ]
    if len(declarations) != 1:
        return None, {
            "status": "missing",
            "reason": "The DFT block must contain exactly one xc declaration.",
        }
    return declarations[0], {
        "status": "extracted",
        "value": declarations[0],
        "block_selection": "last",
    }


def _scf_from_output(
    output_path: str | None,
) -> tuple[dict[str, Any], bool | None, float | None]:
    if output_path is None:
        missing = {"status": "missing", "reason": "No NWChem output file was supplied."}
        return {
            "output": None,
            "scf_converged": missing,
            "total_hartree": missing,
        }, None, None
    scf = parse_scf(output_path)
    status = scf["status"]
    converged = True if status == "converged" else False if status in {"failed", "incomplete"} else None
    energy = scf["total_energy_hartree"] if converged is True else None
    return {
        "output": {
            "path": scf["metadata"]["file"],
            "scf_status": status,
            "run_count": scf["run_count"],
        },
        "scf_converged": {
            "status": "extracted" if converged is not None else "missing",
            "value": converged,
            "scf_status": status,
        },
        "total_hartree": {
            "status": "extracted" if energy is not None else "missing",
            "value": energy,
            "reason": (
                None if energy is not None else "A final energy is retained only for a converged SCF result."
            ),
        },
    }, converged, energy


def _declared_pyscf_method(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("pyscf_method must be one of " + ", ".join(sorted(_PYSCF_METHODS)))
    normalized = value.lower()
    if normalized not in _PYSCF_METHODS:
        raise ValueError(
            "pyscf_method must be one of " + ", ".join(sorted(_PYSCF_METHODS))
        )
    return normalized


def _declared_density_fit(value: bool | None) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("density_fit must be a Boolean")
    return value


def _declared_pyscf_xc(
    value: str | None,
    method: str | None,
) -> str | None:
    if method is None:
        if value is not None:
            raise ValueError("pyscf_xc requires an explicit pyscf_method")
        return None
    if method in {"rhf", "uhf"}:
        if value is not None:
            raise ValueError("pyscf_xc must be null for RHF and UHF")
        return None
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError("RKS and UKS require a non-empty pyscf_xc up to 200 characters")
    return value.strip()


def _declared_electron_total(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("electron_total must be a positive integer")
    return value


def _input_field_source(input_summary: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = input_summary[field_name]
    return {
        "status": "extracted" if value is not None else "missing",
        "value": value,
        "path": input_summary["file"],
        "source": input_summary.get(f"{field_name}_source"),
    }


def _required_values(reference: dict[str, Any]) -> dict[str, Any]:
    required = {
        "geometry": reference["geometry"],
        "calculation.method": reference["calculation"]["method"],
        "calculation.basis": reference["calculation"]["basis"],
        "calculation.density_fit": reference["calculation"]["density_fit"],
        "calculation.charge": reference["calculation"]["charge"],
        "calculation.multiplicity": reference["calculation"]["multiplicity"],
        "scf.converged": reference["scf"]["converged"],
        "energy.total_hartree": reference["energy"]["total_hartree"],
        "electrons.total": reference["electrons"]["total"],
    }
    if reference["calculation"]["method"] in {"rks", "uks"}:
        required["calculation.xc"] = reference["calculation"]["xc"]
    return required


def _geometry_units(header_line: str) -> str | None:
    match = _GEOMETRY_UNITS_RE.search(header_line)
    if match is None:
        return None
    return "angstrom" if match.group(1).lower().startswith("angstrom") else "bohr"


def _cautions(
    *,
    nwchem_method: str | None,
    geometry_evidence: dict[str, Any],
    basis_evidence: dict[str, Any],
    output_path: str | None,
) -> list[str]:
    cautions = [
        "This is an evidence draft, not a correctness verdict or an automatic NWChem-to-PySCF method conversion.",
    ]
    if nwchem_method is not None:
        cautions.append(
            f"NWChem's selected task module is '{nwchem_method}'; declare the matching PySCF method explicitly."
        )
    if geometry_evidence["status"] != "extracted":
        cautions.append("Supply a single Cartesian geometry block with explicit units before comparing geometries.")
    if basis_evidence["status"] != "extracted":
        cautions.append("A multi-basis or manual NWChem basis must be represented manually for the current PySCF runner.")
    if output_path is None:
        cautions.append("Supply the completed NWChem output to include SCF outcome and final energy evidence.")
    return cautions


__all__ = [
    "NWCHEM_PYSCF_REFERENCE_DRAFT_SCHEMA",
    "draft_nwchem_pyscf_reference",
]
