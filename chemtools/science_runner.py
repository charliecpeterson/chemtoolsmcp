"""Fixed companion-runtime operations for bounded scientific inspection."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any


RESULT_SENTINEL = "CHEMTOOLS_SCIENCE_RESULT="
COMPANION_RUNTIME_PROVENANCE_SCHEMA = "chemtools.companion-runtime-provenance/1"
RDKIT_PREFLIGHT_REQUEST_SCHEMA = "chemtools.rdkit-preflight-request/1"
RDKIT_PREFLIGHT_RESULT_SCHEMA = "chemtools.rdkit-preflight-result/1"
OPENBABEL_CONVERSION_REQUEST_SCHEMA = "chemtools.openbabel-conversion-request/1"
OPENBABEL_CONVERSION_RESULT_SCHEMA = "chemtools.openbabel-conversion-result/1"
ORBITRON_PERIODIC_REQUEST_SCHEMA = (
    "chemtools.orbitron-periodic-electronic-structure-request/1"
)
ORBITRON_PERIODIC_RESULT_SCHEMA = (
    "chemtools.orbitron-periodic-electronic-structure-result/1"
)
ORBITRON_STRUCTURE_IDENTITY_REQUEST_SCHEMA = (
    "chemtools.orbitron-structure-identity-request/1"
)
ORBITRON_STRUCTURE_IDENTITY_RESULT_SCHEMA = (
    "chemtools.orbitron-structure-identity-result/1"
)
ORBITRON_NBO_REQUEST_SCHEMA = "chemtools.orbitron-nbo-request/1"
ORBITRON_NBO_RESULT_SCHEMA = "chemtools.orbitron-nbo-result/1"
QMCPACK_HDF5_INSPECTION_REQUEST_SCHEMA = "chemtools.qmcpack-hdf5-inspection-request/1"
PYSCF_SINGLE_POINT_REQUEST_SCHEMA = "chemtools.pyscf-single-point-request/1"
PYSCF_SINGLE_POINT_RESULT_SCHEMA = "chemtools.pyscf-single-point-result/1"
MAX_REQUEST_BYTES = 1_048_576
MAX_OPENBABEL_CONVERSION_TEXT_BYTES = 128 * 1024
MAX_ORBITRON_PERIODIC_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
SCIENCE_RUNTIME_LOCK_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "science"
    / "chemtools-science-linux-64.explicit.txt"
)


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    if arguments not in (
        ["rdkit-preflight"],
        ["openbabel-convert"],
        ["orbitron-periodic-electronic-structure"],
        ["orbitron-structure-identity"],
        ["orbitron-nbo"],
        ["qmcpack-hdf5-inspect"],
        ["pyscf-single-point"],
    ):
        return _write_result(_with_runtime_provenance({
            "schema_version": "chemtools.science-runner-error/1",
            "status": "invalid_operation",
            "message": (
                "operation must be rdkit-preflight, openbabel-convert, "
                "orbitron-periodic-electronic-structure, "
                "orbitron-structure-identity, orbitron-nbo, qmcpack-hdf5-inspect, or "
                "pyscf-single-point"
            ),
        }))
    try:
        request = _read_request()
    except ValueError as error:
        return _write_result(_with_runtime_provenance({
            "schema_version": "chemtools.science-runner-error/1",
            "status": "invalid_request",
            "message": str(error),
        }))
    if arguments[0] == "rdkit-preflight":
        result = rdkit_preflight(request)
    elif arguments[0] == "openbabel-convert":
        result = openbabel_convert(request)
    elif arguments[0] == "orbitron-periodic-electronic-structure":
        result = orbitron_periodic_electronic_structure(request)
    elif arguments[0] == "orbitron-structure-identity":
        result = orbitron_structure_identity(request)
    elif arguments[0] == "orbitron-nbo":
        result = orbitron_nbo(request)
    elif arguments[0] == "qmcpack-hdf5-inspect":
        result = qmcpack_hdf5_inspect(request)
    else:
        result = pyscf_single_point(request)
    return _write_result(_with_runtime_provenance(
        result,
        operation=arguments[0],
        request=request,
    ))


def rdkit_preflight(request: Any) -> dict[str, Any]:
    try:
        source_format, source = _rdkit_request(request)
    except ValueError as error:
        return _rdkit_error("invalid_request", str(error))

    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception as error:
        return _rdkit_error("runtime_error", _error_text(error))

    try:
        if source_format == "smiles":
            molecule = Chem.MolFromSmiles(source, sanitize=True)
        else:
            molecule = Chem.MolFromMolBlock(
                source,
                sanitize=True,
                removeHs=False,
                strictParsing=True,
            )
    except Exception as error:
        return _rdkit_error("invalid_molecule", _error_text(error))
    if molecule is None:
        return _rdkit_error(
            "invalid_molecule",
            "RDKit could not parse and sanitize the submitted molecule",
        )

    return {
        "schema_version": RDKIT_PREFLIGHT_RESULT_SCHEMA,
        "status": "valid",
        "input": {
            "format": source_format,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "source_preserved": True,
        },
        "rdkit": {
            "version": _module_version(rdkit),
            **_rdkit_molecular_evidence(Chem, rdMolDescriptors, molecule),
        },
        "warnings": _rdkit_warnings(Chem, molecule),
    }


def openbabel_convert(request: Any) -> dict[str, Any]:
    try:
        source_format, source, output_format = _openbabel_request(request)
    except ValueError as error:
        return _openbabel_error("invalid_request", str(error))

    try:
        import rdkit
        from openbabel import openbabel as ob
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception as error:
        return _openbabel_error("runtime_error", _error_text(error))

    try:
        source_molecule = _rdkit_parse(Chem, source_format, source)
    except ValueError as error:
        return _openbabel_error("invalid_molecule", str(error))

    conversion = ob.OBConversion()
    if not conversion.SetInAndOutFormats(
        _openbabel_format(source_format),
        _openbabel_format(output_format),
    ):
        return _openbabel_error("runtime_error", "Open Babel rejected a fixed format")
    if output_format == "smiles":
        conversion.AddOption("c", ob.OBConversion.OUTOPTIONS)
    molecule = ob.OBMol()
    if not conversion.ReadString(molecule, source):
        return _openbabel_error(
            "invalid_molecule",
            "Open Babel could not parse the submitted molecule",
        )
    converted = conversion.WriteString(molecule)
    if not converted.strip():
        return _openbabel_error(
            "runtime_error",
            "Open Babel produced empty converted text",
        )
    if len(converted.encode("utf-8")) > MAX_OPENBABEL_CONVERSION_TEXT_BYTES:
        return _openbabel_error(
            "output_too_large",
            "Open Babel converted text exceeds the 128 KiB limit",
        )

    try:
        converted_molecule = _rdkit_parse(Chem, output_format, converted)
    except ValueError as error:
        return _openbabel_error(
            "uninspectable_output",
            f"Open Babel output could not be independently inspected: {error}",
        )

    source_evidence = _rdkit_molecular_evidence(
        Chem,
        rdMolDescriptors,
        source_molecule,
    )
    converted_evidence = _rdkit_molecular_evidence(
        Chem,
        rdMolDescriptors,
        converted_molecule,
    )
    differences = {
        key: {"source": source_evidence[key], "converted": converted_evidence[key]}
        for key in source_evidence
        if source_evidence[key] != converted_evidence[key]
    }
    return {
        "schema_version": OPENBABEL_CONVERSION_RESULT_SCHEMA,
        "status": "completed",
        "input": {
            "format": source_format,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "source_preserved": True,
        },
        "converted": {
            "format": output_format,
            "text": converted,
            "sha256": hashlib.sha256(converted.encode("utf-8")).hexdigest(),
            "coordinate_status": _converted_coordinate_status(
                source_format,
                output_format,
            ),
        },
        "provenance": {
            "openbabel_version": ob.OBReleaseVersion(),
            "rdkit_version": _module_version(rdkit),
        },
        "comparison": {
            "status": "matched" if not differences else "different",
            "source_rdkit": source_evidence,
            "converted_rdkit": converted_evidence,
            "differences": differences,
        },
        "warnings": _rdkit_warnings(Chem, converted_molecule),
    }


def orbitron_periodic_electronic_structure(request: Any) -> dict[str, Any]:
    try:
        source_path = _orbitron_source_request(
            request,
            schema=ORBITRON_PERIODIC_REQUEST_SCHEMA,
            operation="periodic electronic-structure",
        )
    except ValueError as error:
        return _orbitron_periodic_error("invalid_request", str(error))
    try:
        import orbitron
    except Exception as error:
        return _orbitron_periodic_error("runtime_error", _error_text(error))

    try:
        source = _orbitron_source_evidence(source_path)
    except OSError as error:
        return _orbitron_periodic_error("tool_refused", _error_text(error))
    try:
        scene = orbitron.load(str(source_path))
        periodic = scene.periodic_electronic_structure(
            include_projections=False,
        )
    except Exception as error:
        return _orbitron_periodic_error("tool_refused", _error_text(error), source)
    if periodic is None:
        return {
            "schema_version": ORBITRON_PERIODIC_RESULT_SCHEMA,
            "status": "unavailable_data",
            "source": source,
            "provenance": _orbitron_provenance(orbitron),
            "message": "Orbitron found no periodic electronic-structure data",
        }
    try:
        summary = _periodic_electronic_structure_summary(periodic)
    except ValueError as error:
        return _orbitron_periodic_error("runtime_error", str(error), source)
    return {
        "schema_version": ORBITRON_PERIODIC_RESULT_SCHEMA,
        "status": "completed",
        "source": source,
        "provenance": _orbitron_provenance(orbitron),
        "periodic_electronic_structure": summary,
    }


def orbitron_structure_identity(request: Any) -> dict[str, Any]:
    try:
        source_path = _orbitron_source_request(
            request,
            schema=ORBITRON_STRUCTURE_IDENTITY_REQUEST_SCHEMA,
            operation="structure-identity",
        )
    except ValueError as error:
        return _orbitron_structure_identity_error("invalid_request", str(error))
    try:
        import orbitron
    except Exception as error:
        return _orbitron_structure_identity_error("runtime_error", _error_text(error))

    try:
        source = _orbitron_source_evidence(source_path)
    except OSError as error:
        return _orbitron_structure_identity_error("tool_refused", _error_text(error))
    try:
        scene = orbitron.load(str(source_path))
        summary = _orbitron_structure_identity_summary(orbitron.Orbitron(), scene)
    except ValueError as error:
        return _orbitron_structure_identity_error("runtime_error", str(error), source)
    except Exception as error:
        return _orbitron_structure_identity_error("tool_refused", _error_text(error), source)
    return {
        "schema_version": ORBITRON_STRUCTURE_IDENTITY_RESULT_SCHEMA,
        "status": "completed",
        "source": source,
        "provenance": _orbitron_provenance(orbitron),
        "structure_identity": summary,
    }


def orbitron_nbo(request: Any) -> dict[str, Any]:
    try:
        source_path = _orbitron_source_request(
            request,
            schema=ORBITRON_NBO_REQUEST_SCHEMA,
            operation="NBO",
        )
    except ValueError as error:
        return _orbitron_nbo_error("invalid_request", str(error))
    try:
        import orbitron
    except Exception as error:
        return _orbitron_nbo_error("runtime_error", _error_text(error))

    try:
        source = _orbitron_source_evidence(source_path)
    except OSError as error:
        return _orbitron_nbo_error("tool_refused", _error_text(error))
    try:
        scene = orbitron.load(str(source_path))
        nbo = orbitron.Orbitron().analyze_nbo(scene, top_atoms=5)
    except Exception as error:
        return _orbitron_nbo_error("tool_refused", _error_text(error), source)
    if nbo is None:
        return {
            "schema_version": ORBITRON_NBO_RESULT_SCHEMA,
            "status": "unavailable_data",
            "source": source,
            "provenance": _orbitron_provenance(orbitron),
            "message": "Orbitron found no Natural Bond Orbital data",
        }
    try:
        summary = _orbitron_nbo_summary(nbo, scene.atom_count())
    except ValueError as error:
        return _orbitron_nbo_error("runtime_error", str(error), source)
    return {
        "schema_version": ORBITRON_NBO_RESULT_SCHEMA,
        "status": "completed",
        "source": source,
        "provenance": _orbitron_provenance(orbitron),
        "nbo": summary,
    }


def qmcpack_hdf5_inspect(request: Any) -> dict[str, Any]:
    try:
        source = _qmcpack_hdf5_request(request)
    except ValueError as error:
        return {
            "schema_version": "chemtools.qmcpack-hdf5-inspection/1",
            "status": "invalid_request",
            "message": str(error),
        }
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        # The fixed runner is executed by path from a separate interpreter.
        sys.path.insert(0, str(package_root))
    from chemtools.programs.qmcpack.hdf5 import inspect_qmcpack_hdf5

    return inspect_qmcpack_hdf5(source)


def pyscf_single_point(request: Any) -> dict[str, Any]:
    try:
        specification = _pyscf_request(request)
    except ValueError as error:
        return _pyscf_error("invalid_request", str(error))
    try:
        from pyscf import dft, gto, scf
        import pyscf
    except Exception as error:
        return _pyscf_error("runtime_error", _error_text(error))

    try:
        molecule = gto.M(
            atom="; ".join(
                f"{atom['element']} {atom['x']} {atom['y']} {atom['z']}"
                for atom in specification["atoms"]
            ),
            unit="Angstrom",
            charge=specification["charge"],
            spin=specification["multiplicity"] - 1,
            basis=specification["basis"],
            max_memory=specification["max_memory_mb"],
            verbose=0,
        )
        method = specification["method"]
        if method == "rhf":
            mean_field = scf.RHF(molecule)
        elif method == "uhf":
            mean_field = scf.UHF(molecule)
        elif method == "rks":
            mean_field = dft.RKS(molecule)
            mean_field.xc = specification["xc"]
        else:
            mean_field = dft.UKS(molecule)
            mean_field.xc = specification["xc"]
        mean_field.max_cycle = specification["max_cycles"]
        mean_field.conv_tol = specification["convergence_tolerance"]
        if specification["density_fit"]:
            mean_field = mean_field.density_fit()
        mean_field.kernel()
    except Exception as error:
        return _pyscf_error("runtime_error", _error_text(error), specification)

    electrons_alpha, electrons_beta = molecule.nelec
    density_cube, density_warnings = _density_cube_artifact(
        molecule,
        mean_field,
        specification,
    )
    orbital_cubes, orbital_warnings = _orbital_cube_artifacts(
        molecule,
        mean_field,
        specification,
    )
    warnings = (
        [] if mean_field.converged else [{
            "code": "scf_not_converged",
            "message": "PySCF stopped without satisfying the SCF convergence criterion",
        }]
    )
    warnings.extend(density_warnings)
    warnings.extend(orbital_warnings)
    response = {
        "schema_version": PYSCF_SINGLE_POINT_RESULT_SCHEMA,
        "status": "completed",
        "calculation": {
            "method": method,
            "basis": specification["basis"],
            "xc": specification["xc"],
            "density_fit": specification["density_fit"],
            "charge": specification["charge"],
            "multiplicity": specification["multiplicity"],
            "atom_count": len(specification["atoms"]),
        },
        "geometry": specification["atoms"],
        "provenance": {
            "pyscf_version": pyscf.__version__,
            "python_version": sys.version.split()[0],
        },
        "scf": {
            "converged": bool(mean_field.converged),
            "cycles": int(mean_field.cycles),
            "convergence_tolerance": specification["convergence_tolerance"],
        },
        "energy": {
            "total_hartree": float(mean_field.e_tot),
        },
        "electrons": {
            "total": electrons_alpha + electrons_beta,
            "alpha": electrons_alpha,
            "beta": electrons_beta,
        },
        "warnings": warnings,
    }
    if density_cube is not None:
        response["density_cube"] = density_cube
    if orbital_cubes is not None:
        response["orbital_cubes"] = orbital_cubes
    return response


def _read_request() -> Any:
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("request exceeds the 1 MiB limit")
    try:
        return json.loads(payload, parse_constant=_reject_nonfinite_json)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("request must be valid JSON") from error


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not permitted: {value}")


def _rdkit_request(request: Any) -> tuple[str, str]:
    if not isinstance(request, dict):
        raise ValueError("RDKit request must be an object")
    if request.get("schema_version") != RDKIT_PREFLIGHT_REQUEST_SCHEMA:
        raise ValueError("unsupported RDKit request schema")
    if set(request) != {"schema_version", "format", "source"}:
        raise ValueError("RDKit request contains unsupported fields")
    source_format = request["format"]
    source = request["source"]
    if source_format not in {"smiles", "molblock"}:
        raise ValueError("RDKit format must be smiles or molblock")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("RDKit source must be non-empty text")
    return source_format, source


def _openbabel_request(request: Any) -> tuple[str, str, str]:
    if not isinstance(request, dict):
        raise ValueError("Open Babel request must be an object")
    if request.get("schema_version") != OPENBABEL_CONVERSION_REQUEST_SCHEMA:
        raise ValueError("unsupported Open Babel request schema")
    if set(request) != {"schema_version", "format", "source", "output_format"}:
        raise ValueError("Open Babel request contains unsupported fields")
    source_format = request["format"]
    output_format = request["output_format"]
    source = request["source"]
    if source_format not in {"smiles", "molblock"}:
        raise ValueError("Open Babel format must be smiles or molblock")
    if output_format not in {"smiles", "molblock"}:
        raise ValueError("Open Babel output_format must be smiles or molblock")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Open Babel source must be non-empty text")
    if len(source.encode("utf-8")) > MAX_OPENBABEL_CONVERSION_TEXT_BYTES:
        raise ValueError("Open Babel source exceeds the 128 KiB limit")
    return source_format, source, output_format


def _orbitron_source_request(
    request: Any,
    *,
    schema: str,
    operation: str,
) -> Path:
    if not isinstance(request, dict):
        raise ValueError(f"Orbitron {operation} request must be an object")
    if request.get("schema_version") != schema:
        raise ValueError(f"unsupported Orbitron {operation} request schema")
    if set(request) != {"schema_version", "path"}:
        raise ValueError(f"Orbitron {operation} request contains unsupported fields")
    raw_path = request["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Orbitron {operation} path must be non-empty text")
    if not Path(raw_path).is_absolute():
        raise ValueError(f"Orbitron {operation} path must be absolute")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"Orbitron {operation} path is not a file: {path}")
    if path.stat().st_size > MAX_ORBITRON_PERIODIC_SOURCE_BYTES:
        raise ValueError(f"Orbitron {operation} source exceeds the 2 GiB limit")
    return path


def _orbitron_periodic_request(request: Any) -> Path:
    return _orbitron_source_request(
        request,
        schema=ORBITRON_PERIODIC_REQUEST_SCHEMA,
        operation="periodic electronic-structure",
    )


def _qmcpack_hdf5_request(request: Any) -> Path:
    if not isinstance(request, dict):
        raise ValueError("QMCPACK HDF5 inspection request must be an object")
    if request.get("schema_version") != QMCPACK_HDF5_INSPECTION_REQUEST_SCHEMA:
        raise ValueError("unsupported QMCPACK HDF5 inspection request schema")
    if set(request) != {"schema_version", "path"}:
        raise ValueError("QMCPACK HDF5 inspection request contains unsupported fields")
    raw_path = request["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("QMCPACK HDF5 inspection path must be non-empty text")
    if not Path(raw_path).is_absolute():
        raise ValueError("QMCPACK HDF5 inspection path must be absolute")
    return Path(raw_path).resolve()


def _orbitron_source_evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _orbitron_provenance(orbitron: Any) -> dict[str, str]:
    return {
        "orbitron_version": str(orbitron.__version__),
        "python_version": sys.version.split()[0],
    }


def _orbitron_structure_identity_summary(
    service: Any,
    scene: Any,
) -> dict[str, Any]:
    atom_count = scene.atom_count()
    bond_count = scene.bond_count()
    bonds = scene.bonds()
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 1
        or isinstance(bond_count, bool)
        or not isinstance(bond_count, int)
        or bond_count < 0
        or not isinstance(bonds, list)
        or len(bonds) != bond_count
    ):
        raise ValueError("Orbitron structure counts are invalid")
    bond_order_counts: dict[str, int] = {}
    for bond in bonds:
        if not isinstance(bond, dict) or set(bond) != {"a", "b", "order"}:
            raise ValueError("Orbitron bond evidence is invalid")
        order = bond["order"]
        if not isinstance(order, str) or not order:
            raise ValueError("Orbitron bond order is invalid")
        bond_order_counts[order] = bond_order_counts.get(order, 0) + 1
    return {
        "atom_count": atom_count,
        "bond_count": bond_count,
        "bond_order_counts": dict(sorted(bond_order_counts.items())),
        "identifiers": {
            name: _orbitron_identifier(service, name, scene)
            for name in ("formula", "inchi", "inchikey", "smiles")
        },
    }


def _orbitron_identifier(
    service: Any,
    name: str,
    scene: Any,
) -> dict[str, str]:
    try:
        value = getattr(service, name)(scene)
    except Exception as error:
        return {"status": "unavailable", "message": _error_text(error)}
    if not isinstance(value, str) or not value:
        raise ValueError(f"Orbitron {name} identifier is invalid")
    return {"status": "available", "value": value}


def _orbitron_nbo_summary(nbo: Any, atom_count: Any) -> dict[str, Any]:
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 1
        or not isinstance(nbo, dict)
    ):
        raise ValueError("Orbitron NBO evidence is invalid")
    orbitals = nbo.get("orbitals")
    per_atom = nbo.get("per_atom")
    if not isinstance(orbitals, list) or not orbitals or not isinstance(per_atom, dict):
        raise ValueError("Orbitron NBO evidence is incomplete")

    normalized_orbitals = [
        _orbitron_nbo_orbital(orbital, atom_count)
        for orbital in orbitals
    ]
    numbers = [orbital["number"] for orbital in normalized_orbitals]
    if len(set(numbers)) != len(numbers):
        raise ValueError("Orbitron NBO orbital numbers are not unique")
    return {
        "orbital_count": len(normalized_orbitals),
        "orbital_type_counts": _count_nbo_orbital_types(normalized_orbitals),
        "occupancy_range": {
            "minimum": min(orbital["occupancy"] for orbital in normalized_orbitals),
            "maximum": max(orbital["occupancy"] for orbital in normalized_orbitals),
        },
        "per_atom_entry_counts": _nbo_per_atom_entry_counts(per_atom, atom_count),
        "bonding_orbital_samples": [
            orbital
            for orbital in normalized_orbitals
            if orbital["orbital_type"].startswith(("BD", "LP"))
        ][:12],
    }


def _orbitron_nbo_orbital(orbital: Any, atom_count: int) -> dict[str, Any]:
    if not isinstance(orbital, dict):
        raise ValueError("Orbitron NBO orbital evidence is invalid")
    number = orbital.get("number")
    label = orbital.get("label")
    orbital_type = orbital.get("orbital_type")
    occupancy = _finite_or_none(orbital.get("occupancy"))
    atoms = orbital.get("atoms")
    if (
        isinstance(number, bool)
        or not isinstance(number, int)
        or number < 1
        or not isinstance(label, str)
        or not label
        or not isinstance(orbital_type, str)
        or not orbital_type
        or occupancy is None
        or not isinstance(atoms, list)
        or not 1 <= len(atoms) <= 5
    ):
        raise ValueError("Orbitron NBO orbital evidence is invalid")
    return {
        "number": number,
        "label": label,
        "orbital_type": orbital_type,
        "occupancy": occupancy,
        "atoms": [_orbitron_nbo_atom(atom, atom_count) for atom in atoms],
    }


def _orbitron_nbo_atom(atom: Any, atom_count: int) -> dict[str, Any]:
    if not isinstance(atom, dict):
        raise ValueError("Orbitron NBO atom evidence is invalid")
    atom_index = atom.get("atom_index")
    element = atom.get("element")
    weight = _finite_or_none(atom.get("weight"))
    is_positive = atom.get("is_positive")
    if (
        isinstance(atom_index, bool)
        or not isinstance(atom_index, int)
        or not 0 <= atom_index < atom_count
        or not isinstance(element, str)
        or not element
        or weight is None
        or not 0.0 <= weight <= 1.0
        or not isinstance(is_positive, bool)
    ):
        raise ValueError("Orbitron NBO atom evidence is invalid")
    return {
        "atom_index": atom_index,
        "element": element,
        "weight": weight,
        "is_positive": is_positive,
    }


def _count_nbo_orbital_types(orbitals: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for orbital in orbitals:
        orbital_type = orbital["orbital_type"]
        counts[orbital_type] = counts.get(orbital_type, 0) + 1
    return dict(sorted(counts.items()))


def _nbo_per_atom_entry_counts(
    per_atom: dict[Any, Any],
    atom_count: int,
) -> list[dict[str, int]]:
    counts = []
    for raw_index, entries in per_atom.items():
        try:
            atom_index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError("Orbitron NBO per-atom index is invalid") from error
        if not 0 <= atom_index < atom_count or not isinstance(entries, list):
            raise ValueError("Orbitron NBO per-atom evidence is invalid")
        counts.append({"atom_index": atom_index, "entry_count": len(entries)})
    return sorted(counts, key=lambda value: value["atom_index"])


def _periodic_electronic_structure_summary(periodic: Any) -> dict[str, Any]:
    if not isinstance(periodic, dict):
        raise ValueError("Orbitron periodic electronic structure must be an object")
    band_structure = periodic.get("band_structure")
    density_of_states = periodic.get("density_of_states")
    return {
        "fermi_energy_ev": _finite_or_none(periodic.get("fermi_energy_ev")),
        "total_magnetization_bohr": _finite_or_none(
            periodic.get("total_magnetization_bohr")
        ),
        "band_gap": _band_gap_summary(periodic.get("band_gap")),
        "band_structure": _band_structure_summary(band_structure),
        "density_of_states": _density_of_states_summary(density_of_states),
        "projected_data": "omitted",
    }


def _band_gap_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "unavailable"}
    if not isinstance(value, dict):
        raise ValueError("Orbitron band-gap evidence must be an object")
    gap = _finite_or_none(value.get("value_ev"))
    direct = value.get("is_direct")
    if gap is None or not isinstance(direct, bool):
        raise ValueError("Orbitron band-gap evidence is incomplete")
    return {
        "status": "available",
        "value_ev": gap,
        "is_direct": direct,
    }


def _band_structure_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "unavailable"}
    if not isinstance(value, dict):
        raise ValueError("Orbitron band-structure evidence must be an object")
    kpoints = value.get("kpoints_fractional")
    spin_channels = value.get("spin_channels")
    eigenvalues = value.get("eigenvalues_ev")
    sampling = value.get("sampling")
    if (
        not isinstance(kpoints, list)
        or not isinstance(spin_channels, list)
        or not isinstance(eigenvalues, list)
        or not isinstance(sampling, str)
    ):
        raise ValueError("Orbitron band-structure evidence is incomplete")
    if len(spin_channels) != len(eigenvalues):
        raise ValueError("Orbitron band-structure spin channels do not match eigenvalues")
    bands_per_spin = []
    for spin_eigenvalues in eigenvalues:
        if not isinstance(spin_eigenvalues, list):
            raise ValueError("Orbitron band-structure eigenvalues are invalid")
        counts = set()
        for kpoint_eigenvalues in spin_eigenvalues:
            if not isinstance(kpoint_eigenvalues, list):
                raise ValueError("Orbitron band-structure k-point values are invalid")
            counts.add(len(kpoint_eigenvalues))
        bands_per_spin.append(next(iter(counts)) if len(counts) == 1 else None)
    return {
        "status": "available",
        "sampling": sampling,
        "spin_channels": spin_channels,
        "kpoint_count": len(kpoints),
        "band_count_per_spin": bands_per_spin,
        "label_count": len(value.get("kpoint_labels", [])),
        "segment_count": len(value.get("segments", [])),
    }


def _density_of_states_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "unavailable"}
    if not isinstance(value, dict):
        raise ValueError("Orbitron density-of-states evidence must be an object")
    energies = value.get("energies_ev")
    spin_channels = value.get("spin_channels")
    densities = value.get("densities")
    if (
        not isinstance(energies, list)
        or not isinstance(spin_channels, list)
        or not isinstance(densities, list)
    ):
        raise ValueError("Orbitron density-of-states evidence is incomplete")
    if len(spin_channels) != len(densities):
        raise ValueError("Orbitron density-of-states spin channels do not match densities")
    finite_energies = [_finite_or_none(value) for value in energies]
    if any(value is None for value in finite_energies):
        raise ValueError("Orbitron density-of-states energies are invalid")
    return {
        "status": "available",
        "spin_channels": spin_channels,
        "energy_point_count": len(energies),
        "energy_min_ev": min(finite_energies) if finite_energies else None,
        "energy_max_ev": max(finite_energies) if finite_energies else None,
        "integrated_available": value.get("integrated") is not None,
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Orbitron numeric evidence must be finite")
    if not math.isfinite(value):
        raise ValueError("Orbitron numeric evidence must be finite")
    return float(value)


def _openbabel_format(format_name: str) -> str:
    return {"smiles": "smi", "molblock": "mol"}[format_name]


def _rdkit_parse(Chem: Any, source_format: str, source: str) -> Any:
    try:
        if source_format == "smiles":
            molecule = Chem.MolFromSmiles(source, sanitize=True)
        else:
            molecule = Chem.MolFromMolBlock(
                source,
                sanitize=True,
                removeHs=False,
                strictParsing=True,
            )
    except Exception as error:
        raise ValueError(_error_text(error)) from error
    if molecule is None:
        raise ValueError("RDKit could not parse and sanitize the molecule")
    return molecule


def _rdkit_molecular_evidence(
    Chem: Any,
    rdMolDescriptors: Any,
    molecule: Any,
) -> dict[str, Any]:
    radical_electrons = sum(
        atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
    )
    return {
        "canonical_smiles": Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        ),
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "atom_count": molecule.GetNumAtoms(),
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "bond_count": molecule.GetNumBonds(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "radical_electrons": radical_electrons,
        "fragment_count": len(Chem.GetMolFrags(molecule)),
        "aromatic_atom_count": sum(
            atom.GetIsAromatic() for atom in molecule.GetAtoms()
        ),
        "stereocenter_count": len(
            Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
        ),
        "stereo_bond_count": sum(
            bond.GetStereo() != Chem.BondStereo.STEREONONE
            for bond in molecule.GetBonds()
        ),
    }


def _rdkit_warnings(Chem: Any, molecule: Any) -> list[dict[str, Any]]:
    fragments = Chem.GetMolFrags(molecule)
    radical_electrons = sum(
        atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
    )
    warnings = []
    if len(fragments) > 1:
        warnings.append({
            "code": "multiple_fragments",
            "message": "RDKit found multiple disconnected molecular fragments",
        })
    if radical_electrons:
        warnings.append({
            "code": "radical_electrons",
            "message": "RDKit reports explicit radical electrons",
            "count": radical_electrons,
        })
    return warnings


def _converted_coordinate_status(
    source_format: str,
    output_format: str,
) -> str:
    if output_format != "molblock":
        return "not_applicable"
    if source_format == "smiles":
        return "not_generated"
    return "source_preserved_by_converter"


def _pyscf_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("PySCF request must be an object")
    required = {
        "schema_version",
        "atoms",
        "charge",
        "multiplicity",
        "method",
        "basis",
        "xc",
        "density_fit",
        "max_cycles",
        "convergence_tolerance",
        "max_memory_mb",
    }
    optional = {
        "density_cube_grid_points",
        "orbital_cube_grid_points",
        "orbital_cube_requests",
    }
    if not required <= set(request) <= required | optional:
        raise ValueError("PySCF request contains unsupported or missing fields")
    if request["schema_version"] != PYSCF_SINGLE_POINT_REQUEST_SCHEMA:
        raise ValueError("unsupported PySCF request schema")
    atoms = request["atoms"]
    if not isinstance(atoms, list) or not 1 <= len(atoms) <= 500:
        raise ValueError("atoms must contain between 1 and 500 entries")
    normalized_atoms = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, dict) or set(atom) != {"element", "x", "y", "z"}:
            raise ValueError(f"atoms[{index}] must define element, x, y, and z")
        element = atom["element"]
        if not isinstance(element, str) or not element.isalpha() or len(element) > 2:
            raise ValueError(f"atoms[{index}].element must be an element symbol")
        coordinates = []
        for field_name in ("x", "y", "z"):
            coordinate = atom[field_name]
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError(f"atoms[{index}].{field_name} must be numeric")
            if not math.isfinite(coordinate):
                raise ValueError(f"atoms[{index}].{field_name} must be finite")
            coordinates.append(float(coordinate))
        normalized_atoms.append({
            "element": element,
            "x": coordinates[0],
            "y": coordinates[1],
            "z": coordinates[2],
        })
    charge = request["charge"]
    multiplicity = request["multiplicity"]
    if isinstance(charge, bool) or not isinstance(charge, int):
        raise ValueError("charge must be an integer")
    if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError("multiplicity must be a positive integer")
    method = request["method"]
    if method not in {"rhf", "uhf", "rks", "uks"}:
        raise ValueError("method must be rhf, uhf, rks, or uks")
    if method in {"rhf", "rks"} and multiplicity != 1:
        raise ValueError("restricted methods require multiplicity 1")
    basis = request["basis"]
    if not isinstance(basis, str) or not basis.strip() or len(basis) > 200:
        raise ValueError("basis must be non-empty text up to 200 characters")
    xc = request["xc"]
    if method in {"rks", "uks"}:
        if not isinstance(xc, str) or not xc.strip() or len(xc) > 200:
            raise ValueError("DFT methods require a non-empty xc functional")
    elif xc is not None:
        raise ValueError("HF methods require xc to be null")
    if not isinstance(request["density_fit"], bool):
        raise ValueError("density_fit must be a boolean")
    max_cycles = request["max_cycles"]
    if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or not 1 <= max_cycles <= 500:
        raise ValueError("max_cycles must be an integer between 1 and 500")
    tolerance = request["convergence_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance <= 1e-4:
        raise ValueError("convergence_tolerance must be finite and between 0 and 1e-4")
    max_memory_mb = request["max_memory_mb"]
    if isinstance(max_memory_mb, bool) or not isinstance(max_memory_mb, int) or not 64 <= max_memory_mb <= 262_144:
        raise ValueError("max_memory_mb must be an integer between 64 and 262144")
    density_cube_grid_points = request.get("density_cube_grid_points")
    _cube_grid_points(density_cube_grid_points, "density_cube_grid_points")
    orbital_cube_grid_points = request.get("orbital_cube_grid_points")
    _cube_grid_points(orbital_cube_grid_points, "orbital_cube_grid_points")
    orbital_cube_requests = _orbital_cube_requests(
        request.get("orbital_cube_requests"),
        method,
    )
    if (orbital_cube_grid_points is None) != (orbital_cube_requests is None):
        raise ValueError(
            "orbital_cube_grid_points and orbital_cube_requests must be supplied together"
        )
    return {
        "atoms": normalized_atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "method": method,
        "basis": basis.strip(),
        "xc": xc.strip() if isinstance(xc, str) else None,
        "density_fit": request["density_fit"],
        "max_cycles": max_cycles,
        "convergence_tolerance": float(tolerance),
        "max_memory_mb": max_memory_mb,
        "density_cube_grid_points": density_cube_grid_points,
        "orbital_cube_grid_points": orbital_cube_grid_points,
        "orbital_cube_requests": orbital_cube_requests,
    }


def _cube_grid_points(value: Any, field_name: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 20 <= value <= 120
    ):
        raise ValueError(f"{field_name} must be an integer between 20 and 120")


def _orbital_cube_requests(
    value: Any,
    method: str,
) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("orbital_cube_requests must contain between 1 and 8 selectors")
    allowed_spins = {"restricted"} if method in {"rhf", "rks"} else {"alpha", "beta"}
    normalized = []
    seen = set()
    for index, selector in enumerate(value):
        if not isinstance(selector, dict) or set(selector) != {"spin", "orbital_index"}:
            raise ValueError(
                f"orbital_cube_requests[{index}] must contain spin and orbital_index"
            )
        spin = selector["spin"]
        if not isinstance(spin, str):
            raise ValueError(f"orbital_cube_requests[{index}].spin must be text")
        if spin not in allowed_spins:
            allowed = ", ".join(sorted(allowed_spins))
            raise ValueError(
                f"orbital_cube_requests[{index}].spin must be one of: {allowed}"
            )
        orbital_index = selector["orbital_index"]
        if (
            isinstance(orbital_index, bool)
            or not isinstance(orbital_index, int)
            or not 0 <= orbital_index <= 20_000
        ):
            raise ValueError(
                f"orbital_cube_requests[{index}].orbital_index must be between 0 and 20000"
            )
        identity = (spin, orbital_index)
        if identity in seen:
            raise ValueError("orbital_cube_requests cannot contain duplicate selectors")
        seen.add(identity)
        normalized.append({"spin": spin, "orbital_index": orbital_index})
    return normalized


def _density_cube_artifact(
    molecule: Any,
    mean_field: Any,
    specification: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    grid_points = specification["density_cube_grid_points"]
    if grid_points is None:
        return None, []
    if not mean_field.converged:
        return {
            "status": "not_written",
            "reason": "scf_not_converged",
        }, [{
            "code": "density_cube_not_written_scf_not_converged",
            "message": "PySCF did not write a density CUBE because SCF did not converge",
        }]
    try:
        from pyscf.tools import cubegen

        artifact_name = _density_cube_name(specification)
        artifact_path = Path.cwd() / artifact_name
        if artifact_path.exists():
            raise FileExistsError(f"density CUBE already exists: {artifact_path}")
        cubegen.density(
            molecule,
            str(artifact_path),
            mean_field.make_rdm1(),
            nx=grid_points,
            ny=grid_points,
            nz=grid_points,
        )
        return {
            "status": "written",
            "path": str(artifact_path.resolve()),
            "sha256": _file_sha256(artifact_path),
            "grid_points": [grid_points, grid_points, grid_points],
            "density_value_unit": "electron_per_bohr3",
        }, []
    except Exception as error:
        return {
            "status": "failed",
            "message": _error_text(error),
        }, [{
            "code": "density_cube_generation_failed",
            "message": "PySCF completed but could not write the requested density CUBE",
        }]


def _density_cube_name(specification: dict[str, Any]) -> str:
    request_digest = hashlib.sha256(
        json.dumps(specification, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return f"pyscf_density_{request_digest[:16]}.cube"


def _orbital_cube_artifacts(
    molecule: Any,
    mean_field: Any,
    specification: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, str]]]:
    selectors = specification["orbital_cube_requests"]
    if selectors is None:
        return None, []
    if not mean_field.converged:
        return [
            {
                **selector,
                "status": "not_written",
                "reason": "scf_not_converged",
            }
            for selector in selectors
        ], [{
            "code": "orbital_cubes_not_written_scf_not_converged",
            "message": "PySCF did not write orbital CUBEs because SCF did not converge",
        }]
    grid_points = specification["orbital_cube_grid_points"]
    artifacts = []
    warnings = []
    try:
        from pyscf.tools import cubegen
    except Exception as error:
        return [
            {
                **selector,
                "status": "failed",
                "message": _error_text(error),
            }
            for selector in selectors
        ], [{
            "code": "orbital_cube_generation_failed",
            "message": "PySCF completed but could not import its CUBE generator",
        }]
    for selector in selectors:
        try:
            coefficients, energies, occupations = _orbital_data(mean_field, selector)
            orbital_index = selector["orbital_index"]
            if orbital_index >= coefficients.shape[1]:
                raise ValueError(
                    f"orbital_index {orbital_index} is outside the available range "
                    f"0 through {coefficients.shape[1] - 1}"
                )
            artifact_path = Path.cwd() / _orbital_cube_name(specification, selector)
            if artifact_path.exists():
                raise FileExistsError(f"orbital CUBE already exists: {artifact_path}")
            cubegen.orbital(
                molecule,
                str(artifact_path),
                coefficients[:, orbital_index],
                nx=grid_points,
                ny=grid_points,
                nz=grid_points,
            )
            artifacts.append({
                **selector,
                "status": "written",
                "path": str(artifact_path.resolve()),
                "sha256": _file_sha256(artifact_path),
                "orbital_label": _orbital_label(selector),
                "orbital_energy_hartree": float(energies[orbital_index]),
                "occupation": float(occupations[orbital_index]),
                "grid_points": [grid_points, grid_points, grid_points],
                "orbital_value_unit": "bohr_to_minus_three_halves",
            })
        except Exception as error:
            artifacts.append({
                **selector,
                "status": "failed",
                "message": _error_text(error),
            })
            warnings.append({
                "code": "orbital_cube_generation_failed",
                "message": (
                    "PySCF completed but could not write requested orbital CUBE "
                    f"{_orbital_label(selector)}"
                ),
            })
    return artifacts, warnings


def _orbital_data(mean_field: Any, selector: dict[str, Any]) -> tuple[Any, Any, Any]:
    if selector["spin"] == "restricted":
        return mean_field.mo_coeff, mean_field.mo_energy, mean_field.mo_occ
    spin_index = 0 if selector["spin"] == "alpha" else 1
    return (
        mean_field.mo_coeff[spin_index],
        mean_field.mo_energy[spin_index],
        mean_field.mo_occ[spin_index],
    )


def _orbital_label(selector: dict[str, Any]) -> str:
    return f"{selector['spin']} orbital {selector['orbital_index']}"


def _orbital_cube_name(
    specification: dict[str, Any],
    selector: dict[str, Any],
) -> str:
    request_digest = hashlib.sha256(
        json.dumps(specification, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return (
        f"pyscf_orbital_{request_digest[:16]}_"
        f"{selector['spin']}_{selector['orbital_index']}.cube"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _rdkit_error(status: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": RDKIT_PREFLIGHT_RESULT_SCHEMA,
        "status": status,
        "message": message,
    }


def _openbabel_error(status: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": OPENBABEL_CONVERSION_RESULT_SCHEMA,
        "status": status,
        "message": message,
    }


def _orbitron_periodic_error(
    status: str,
    message: str,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "schema_version": ORBITRON_PERIODIC_RESULT_SCHEMA,
        "status": status,
        "message": message,
    }
    if source is not None:
        response["source"] = source
    return response


def _orbitron_structure_identity_error(
    status: str,
    message: str,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "schema_version": ORBITRON_STRUCTURE_IDENTITY_RESULT_SCHEMA,
        "status": status,
        "message": message,
    }
    if source is not None:
        response["source"] = source
    return response


def _orbitron_nbo_error(
    status: str,
    message: str,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "schema_version": ORBITRON_NBO_RESULT_SCHEMA,
        "status": status,
        "message": message,
    }
    if source is not None:
        response["source"] = source
    return response


def _pyscf_error(
    status: str,
    message: str,
    specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = {
        "schema_version": PYSCF_SINGLE_POINT_RESULT_SCHEMA,
        "status": status,
        "message": message,
    }
    if specification is not None:
        response["calculation"] = {
            "method": specification["method"],
            "basis": specification["basis"],
            "xc": specification["xc"],
            "density_fit": specification["density_fit"],
            "charge": specification["charge"],
            "multiplicity": specification["multiplicity"],
            "atom_count": len(specification["atoms"]),
        }
    return response


def _module_version(module: Any) -> str | None:
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _with_runtime_provenance(
    result: dict[str, Any],
    *,
    operation: str | None = None,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if "runtime_provenance" in result:
        raise ValueError("science-runner result already has runtime provenance")
    provenance = _runtime_provenance()
    if operation is not None:
        provenance["runner_operation"] = operation
    if request is not None:
        provenance["request"] = {
            "schema_version": request.get("schema_version"),
            "sha256": hashlib.sha256(
                json.dumps(request, sort_keys=True, allow_nan=False).encode("utf-8")
            ).hexdigest(),
        }
    return {
        **result,
        "runtime_provenance": provenance,
    }


def _runtime_provenance() -> dict[str, Any]:
    return {
        "schema_version": COMPANION_RUNTIME_PROVENANCE_SCHEMA,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation().lower(),
            "version": sys.version.split()[0],
        },
        "environment_lock": _runtime_lock_evidence(),
        "packages": {
            name: _installed_package_evidence(distribution, module_name)
            for name, (distribution, module_name) in {
                "pyscf": ("pyscf", "pyscf"),
                "rdkit": ("rdkit", "rdkit"),
                "openbabel": ("openbabel", "openbabel"),
                "h5py": ("h5py", "h5py"),
                "orbitron": ("orbitron", "orbitron"),
            }.items()
        },
    }


def _runtime_lock_evidence() -> dict[str, str]:
    if not SCIENCE_RUNTIME_LOCK_PATH.is_file():
        return {
            "status": "unavailable",
            "identifier": SCIENCE_RUNTIME_LOCK_PATH.name,
        }
    return {
        "status": "available",
        "identifier": SCIENCE_RUNTIME_LOCK_PATH.name,
        "sha256": _file_sha256(SCIENCE_RUNTIME_LOCK_PATH),
    }


def _installed_package_evidence(
    distribution: str,
    module_name: str,
) -> dict[str, str]:
    module = None
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return {"status": "unavailable"}
        version = getattr(module, "__version__", None)
    if version is None:
        return {"status": "unavailable"}
    evidence = {"status": "available", "version": str(version)}
    if module_name == "orbitron":
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                return evidence
        native_module = getattr(module, "_orbitron_py", None)
        native_path = Path(getattr(native_module, "__file__", ""))
        if native_path.is_file():
            evidence["native_module_sha256"] = _file_sha256(native_path)
    return evidence


def _error_text(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        message = type(error).__name__
    return f"{type(error).__name__}: {message[:1_024]}"


def _write_result(result: dict[str, Any]) -> int:
    print(RESULT_SENTINEL + json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
