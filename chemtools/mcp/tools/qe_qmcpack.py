"""QE-to-QMCPACK conversion checks and their MCP tool definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.application.run_inspection import inspect_run
from chemtools.integrations.science_runtime import (
    ScienceRuntimeClient,
    ScienceRuntimeCommandError,
    ScienceRuntimeProtocolError,
    ScienceRuntimeUnavailableError,
)
from chemtools.mcp.decorator import _tool
from chemtools.mcp.tools.qe_qmcpack_definitions import qe_tool_definitions
from chemtools.programs.qe import QE
from chemtools.programs.qe.charge_spin import inspect_charge_spin
from chemtools.programs.qe.input import parse_pw_input
from chemtools.programs.qe.output import parse_pw_output
from chemtools.programs.qe.pseudopotentials import inspect_input_pseudopotentials
from chemtools.programs.qe.phonon import draft_ph_x_input
from chemtools.programs.qe.pw2qmcpack import (
    draft_pw2qmcpack_input,
    inspect_pw2qmcpack_input_scope,
    parse_pw2qmcpack_input,
)
from chemtools.programs.qe.qmcpack import (
    conversion_readiness,
    inspect_conversion_calculation,
    inspect_conversion_readiness,
    inspect_conversion_disk_io,
    inspect_conversion_isolation,
    inspect_conversion_k_points,
    inspect_pwscf_h5_artifact,
    inspect_qe_pw2qmcpack_control_paths,
    inspect_qe_scf_completion,
    inspect_qe_qmcpack_atom_count,
    inspect_qe_qmcpack_charge,
    inspect_qe_qmcpack_collinear_spin,
    inspect_qe_qmcpack_electron_count,
    inspect_qe_qmcpack_geometry,
    inspect_qe_qmcpack_ion_species,
    inspect_qmcpack_hdf5_deck_metadata,
    inspect_qe_qmcpack_pseudopotential_species,
    inspect_qe_qmcpack_pseudopotential_valence,
    inspect_qe_qmcpack_projector_evidence,
    plan_qe_qmcpack_conversion,
)
from chemtools.programs.qmcpack.includes import inspect_xml_includes
from chemtools.programs.qmcpack.input import (
    parse_qmcpack_input,
)
from chemtools.programs.qmcpack.pseudopotential import (
    inspect_referenced_pseudopotentials,
)
from chemtools.programs.qmcpack.sidecars import (
    inspect_hdf5_sidecars,
    inspect_pwscf_h5_reference,
)
from chemtools.science_runner import QMCPACK_HDF5_INSPECTION_REQUEST_SCHEMA


@_tool("check_qe_qmcpack_conversion_ready", program="qe")
def _handle_check_qe_qmcpack_conversion_ready(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    source = Path(arguments["qe_input"]).expanduser().resolve()
    return inspect_conversion_readiness(source, parse_pw_input(source))


@_tool("plan_qe_qmcpack_conversion", program="qe")
def _handle_plan_qe_qmcpack_conversion(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    return plan_qe_qmcpack_conversion(
        qe_input,
        arguments["pwscf_h5"],
        parse_pw_input(qe_input),
    )


@_tool("draft_pw2qmcpack_input", program="qe")
def _handle_draft_pw2qmcpack_input(arguments: dict[str, Any]) -> dict[str, Any]:
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    return {
        "qe_input": str(qe_input),
        **draft_pw2qmcpack_input(parse_pw_input(qe_input)),
    }


@_tool("draft_ph_x_input", program="qe")
def _handle_draft_ph_x_input(arguments: dict[str, Any]) -> dict[str, Any]:
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    return {
        "qe_input": str(qe_input),
        **draft_ph_x_input(
            parse_pw_input(qe_input),
            arguments["title"],
            arguments["q_point"],
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_artifacts", program="qe")
def _handle_inspect_qe_qmcpack_conversion_artifacts(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qe_output = Path(arguments["qe_output"]).expanduser().resolve()
    pwscf_h5 = Path(arguments["pwscf_h5"]).expanduser().resolve()
    parsed_input = parse_pw_input(qe_input)
    parsed_output = parse_pw_output(qe_output)
    checks = [
        inspect_conversion_calculation(parsed_input),
        inspect_conversion_disk_io(parsed_input),
        inspect_conversion_k_points(parsed_input),
        inspect_conversion_isolation(parsed_input),
        inspect_qe_scf_completion(parsed_output),
        inspect_pwscf_h5_artifact(pwscf_h5, qe_input, qe_output),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-artifacts/1",
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This checks declared QE-to-QMCPACK artifact lineage. It does not "
            "read HDF5 contents, run pw2qmcpack, inspect a QMCPACK input, or "
            "establish an energy comparison."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion", program="qe")
def _handle_inspect_qe_qmcpack_conversion(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qe_output = Path(arguments["qe_output"]).expanduser().resolve()
    pwscf_h5 = Path(arguments["pwscf_h5"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    parsed_qe_output = parse_pw_output(qe_output)
    qe_pseudopotentials = inspect_input_pseudopotentials(qe_input, parsed_qe_input)
    qe_charge_spin = inspect_charge_spin(parsed_qe_input, qe_pseudopotentials)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    sidecar_review = inspect_hdf5_sidecars(
        qmcpack_input,
        parsed_qmcpack,
        include_review,
    )
    hdf5_metadata = _inspect_conversion_hdf5_metadata(pwscf_h5)
    checks = [
        inspect_conversion_calculation(parsed_qe_input),
        inspect_conversion_disk_io(parsed_qe_input),
        inspect_conversion_k_points(parsed_qe_input),
        inspect_conversion_isolation(parsed_qe_input),
        inspect_qe_scf_completion(parsed_qe_output),
        inspect_pwscf_h5_artifact(pwscf_h5, qe_input, qe_output),
        inspect_pwscf_h5_reference(sidecar_review, include_review, pwscf_h5),
        inspect_qmcpack_hdf5_deck_metadata(
            hdf5_metadata,
            parsed_qmcpack,
            include_review,
        ),
        inspect_referenced_pseudopotentials(
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
        inspect_qe_qmcpack_pseudopotential_species(
            parsed_qe_input,
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
        inspect_qe_qmcpack_pseudopotential_valence(
            qe_pseudopotentials,
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
        inspect_qe_qmcpack_projector_evidence(
            qe_pseudopotentials,
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
        inspect_qe_qmcpack_electron_count(
            parsed_qe_output,
            qe_charge_spin,
            parsed_qmcpack,
            include_review,
        ),
        inspect_qe_qmcpack_atom_count(
            parsed_qe_input,
            parsed_qmcpack,
            include_review,
        ),
        inspect_qe_qmcpack_ion_species(
            parsed_qe_input,
            parsed_qmcpack,
            include_review,
        ),
        inspect_qe_qmcpack_geometry(
            parsed_qe_input,
            parsed_qmcpack,
            include_review,
        ),
        inspect_qe_qmcpack_collinear_spin(
            qe_charge_spin,
            parsed_qmcpack,
            include_review,
        ),
        inspect_qe_qmcpack_charge(
            qe_charge_spin,
            parsed_qmcpack,
            include_review,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion/1",
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This combines Chemtools' bounded QE-to-QMCPACK conversion "
            "evidence in one response. When the optional science runtime is "
            "configured, it also compares fixed pw2qmcpack HDF5 metadata with "
            "the QMCPACK deck. It does not decode coefficients or arbitrary "
            "HDF5 datasets, run pw2qmcpack or QMCPACK, establish pseudopotential-family "
            "equivalence, or validate an energy comparison."
        ),
    }


def _inspect_conversion_hdf5_metadata(pwscf_h5: Path) -> dict[str, Any]:
    request = {
        "schema_version": QMCPACK_HDF5_INSPECTION_REQUEST_SCHEMA,
        "path": str(pwscf_h5),
    }
    try:
        return ScienceRuntimeClient().qmcpack_hdf5_inspect(request)
    except ScienceRuntimeUnavailableError as error:
        return {"status": "unavailable", "message": str(error)}
    except ScienceRuntimeProtocolError as error:
        return {"status": "incompatible", "message": str(error)}
    except ScienceRuntimeCommandError as error:
        return {
            "status": "tool_refused",
            "message": str(error),
            "returncode": error.returncode,
        }


@_tool("inspect_qe_qmcpack_conversion_execution", program="qe")
def _handle_inspect_qe_qmcpack_conversion_execution(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    conversion = _handle_inspect_qe_qmcpack_conversion(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    converter_input = Path(arguments["pw2qmcpack_input"]).expanduser().resolve()
    converter_output = Path(arguments["pw2qmcpack_output"]).expanduser().resolve()
    pwscf_h5 = Path(arguments["pwscf_h5"]).expanduser().resolve()
    converter_text = converter_input.read_text(encoding="utf-8", errors="replace")
    converter_inspection = inspect_run(
        QE,
        converter_output,
        resolved_by="explicit",
        artifact_files=(converter_input, pwscf_h5),
    )
    control_path_check = inspect_qe_pw2qmcpack_control_paths(
        parse_pw_input(qe_input),
        parse_pw2qmcpack_input(str(converter_input)),
    )
    input_scope_check = inspect_pw2qmcpack_input_scope(converter_text)
    converter_check = _converter_execution_check(converter_inspection)
    checks = [
        *conversion["checks"],
        input_scope_check,
        control_path_check,
        converter_check,
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-execution/1",
        "qe_input": conversion["qe_input"],
        "qe_output": conversion["qe_output"],
        "pw2qmcpack_input": str(converter_input),
        "pw2qmcpack_output": str(converter_output),
        "pwscf_h5": conversion["pwscf_h5"],
        "qmcpack_input": conversion["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "pw2qmcpack_inspection": converter_inspection,
        "scope_limit": (
            "This combines bounded QE, pw2qmcpack, HDF5-path, and QMCPACK-deck "
            "evidence. It does not decode HDF5, establish pseudopotential-family "
            "equivalence, or validate an energy comparison."
        ),
    }


def _converter_execution_check(
    converter_inspection: dict[str, Any],
) -> dict[str, Any]:
    verdict = converter_inspection["assessment"]["verdict"]["label"]
    task = converter_inspection["evidence"]["tasks"][0]
    consistency = converter_inspection["evidence"]["input_output_consistency"]
    observed = {
        "verdict": verdict,
        "task_outcome": task["outcome"],
        "input_output_consistency": consistency,
    }
    if verdict != "converter_completed":
        return {
            "name": "pw2qmcpack_execution",
            "status": "not_ready",
            "observed": observed,
            "message": "pw2qmcpack did not report completed converter output.",
        }
    if consistency["status"] == "mismatch":
        return {
            "name": "pw2qmcpack_execution",
            "status": "not_ready",
            "observed": observed,
            "message": "The converter input, output, and supplied HDF5 artifact disagree.",
        }
    if (
        consistency["status"] != "checked"
        or consistency["summary"]["not_checked"]
    ):
        return {
            "name": "pw2qmcpack_execution",
            "status": "review_required",
            "observed": observed,
            "message": "The converter completed, but its artifact lineage is incomplete.",
        }
    return {
        "name": "pw2qmcpack_execution",
        "status": "pass",
        "observed": observed,
        "message": "pw2qmcpack completed and its declared HDF5 artifact matches.",
    }


@_tool("inspect_qe_qmcpack_conversion_deck", program="qe")
def _handle_inspect_qe_qmcpack_conversion_deck(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    artifact_inspection = _handle_inspect_qe_qmcpack_conversion_artifacts(arguments)
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    sidecar_review = inspect_hdf5_sidecars(
        qmcpack_input,
        parsed_qmcpack,
        include_review,
    )
    checks = [
        *artifact_inspection["checks"],
        inspect_pwscf_h5_reference(
            sidecar_review,
            include_review,
            Path(artifact_inspection["pwscf_h5"]),
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-deck/1",
        "qe_input": artifact_inspection["qe_input"],
        "qe_output": artifact_inspection["qe_output"],
        "pwscf_h5": artifact_inspection["pwscf_h5"],
        "qmcpack_input": str(qmcpack_input),
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This checks that the declared QMCPACK XML deck resolves an HDF5 "
            "reference to the declared QE conversion artifact. It does not read "
            "HDF5 contents, merge XML trees, compare particle or pseudopotential "
            "semantics, run pw2qmcpack, or compare energies."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_pseudopotentials", program="qe")
def _handle_inspect_qe_qmcpack_conversion_pseudopotentials(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_referenced_pseudopotentials(
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-pseudopotentials/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This checks the declared QMCPACK pseudopotential XML files for the "
            "supported semilocal structural evidence. It does not establish "
            "pseudopotential transferability, family equivalence with the QE UPF, "
            "or a valid energy comparison."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_species", program="qe")
def _handle_inspect_qe_qmcpack_conversion_species(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_pseudopotential_species(
            parsed_qe_input,
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-species/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares QE atomic-species elements with QMCPACK "
            "pseudopotential elementType declarations. It does not establish "
            "pseudopotential family equivalence, valence equivalence, spin "
            "state, or coordinate consistency."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_valence", program="qe")
def _handle_inspect_qe_qmcpack_conversion_valence(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    qe_pseudopotentials = inspect_input_pseudopotentials(qe_input, parsed_qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_pseudopotential_valence(
            qe_pseudopotentials,
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-valence/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares parsed QE UPF z_valence values with QMCPACK XML "
            "zval headers. It does not establish pseudopotential family "
            "equivalence, scattering equivalence, spin state, or coordinate "
            "consistency."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_projectors", program="qe")
def _handle_inspect_qe_qmcpack_conversion_projectors(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    qe_pseudopotentials = inspect_input_pseudopotentials(qe_input, parsed_qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_projector_evidence(
            qe_pseudopotentials,
            parsed_qmcpack,
            include_review,
            qmcpack_input,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-projectors/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This reports bounded QE UPF projector evidence when a QMCPACK DMC "
            "block is declared. It does not establish pseudopotential family "
            "equivalence, prove the QMCPACK card's source, or validate DMC."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_electrons", program="qe")
def _handle_inspect_qe_qmcpack_conversion_electrons(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qe_output = Path(arguments["qe_output"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    parsed_qe_output = parse_pw_output(qe_output)
    qe_pseudopotentials = inspect_input_pseudopotentials(qe_input, parsed_qe_input)
    qe_charge_spin = inspect_charge_spin(parsed_qe_input, qe_pseudopotentials)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_electron_count(
            parsed_qe_output,
            qe_charge_spin,
            parsed_qmcpack,
            include_review,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-electrons/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares explicit QE electron-count evidence with the QMCPACK "
            "electron-particle groups. It does not establish a physical charge "
            "state, spin state, orbital occupancy, or pseudopotential equivalence."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_atoms", program="qe")
def _handle_inspect_qe_qmcpack_conversion_atoms(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_atom_count(parsed_qe_input, parsed_qmcpack, include_review),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-atoms/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares QE's declared atom count with QMCPACK non-electron "
            "particle-set sizes. It does not compare element identities, cell "
            "parameters, or coordinates."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_ion_species", program="qe")
def _handle_inspect_qe_qmcpack_conversion_ion_species(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_ion_species(
            parsed_qe_input,
            parsed_qmcpack,
            include_review,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-ion-species/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares QE atomic elements with QMCPACK explicitly sized "
            "non-electron ion groups, including bounded XML includes. It does "
            "not compare coordinates, pseudopotential identity, charge, or spin."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_geometry", program="qe")
def _handle_inspect_qe_qmcpack_conversion_geometry(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_geometry(
            parsed_qe_input,
            parsed_qmcpack,
            include_review,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-geometry/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares one explicit periodic QE geometry with one explicit "
            "QMCPACK ion geometry in bohr, including bounded XML includes. It "
            "does not decode HDF5, compare pseudopotentials, spin state, or "
            "physical conventions beyond the stated cell and coordinates."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_spin", program="qe")
def _handle_inspect_qe_qmcpack_conversion_spin(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_collinear_spin(
            inspect_charge_spin(parsed_qe_input),
            parsed_qmcpack,
            include_review,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-spin/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares only an explicit QE collinear fixed total "
            "magnetization with the QMCPACK u-d electron imbalance. It does "
            "not establish a physical spin state, compare spin densities, or "
            "support noncollinear or spin-orbit calculations."
        ),
    }


@_tool("inspect_qe_qmcpack_conversion_charge", program="qe")
def _handle_inspect_qe_qmcpack_conversion_charge(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    deck_inspection = _handle_inspect_qe_qmcpack_conversion_deck(arguments)
    qe_input = Path(arguments["qe_input"]).expanduser().resolve()
    qmcpack_input = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_qe_input = parse_pw_input(qe_input)
    qe_pseudopotentials = inspect_input_pseudopotentials(qe_input, parsed_qe_input)
    parsed_qmcpack = parse_qmcpack_input(qmcpack_input)
    include_review = inspect_xml_includes(qmcpack_input, parsed_qmcpack)
    checks = [
        *deck_inspection["checks"],
        inspect_qe_qmcpack_charge(
            inspect_charge_spin(parsed_qe_input, qe_pseudopotentials),
            parsed_qmcpack,
            include_review,
        ),
    ]
    return {
        "schema_version": "chemtools.qe-qmcpack-conversion-charge/1",
        "qe_input": deck_inspection["qe_input"],
        "qe_output": deck_inspection["qe_output"],
        "pwscf_h5": deck_inspection["pwscf_h5"],
        "qmcpack_input": deck_inspection["qmcpack_input"],
        "readiness": conversion_readiness(checks),
        "checks": checks,
        "scope_limit": (
            "This compares QE UPF valence accounting and total charge with "
            "QMCPACK ion valence parameters and selected electron particles. "
            "It does not establish pseudopotential family equivalence, a "
            "physical charge state, or spin-state compatibility."
        ),
    }


__all__ = [
    "_handle_check_qe_qmcpack_conversion_ready",
    "_handle_plan_qe_qmcpack_conversion",
    "_handle_draft_pw2qmcpack_input",
    "_handle_draft_ph_x_input",
    "_handle_inspect_qe_qmcpack_conversion",
    "_handle_inspect_qe_qmcpack_conversion_execution",
    "_handle_inspect_qe_qmcpack_conversion_artifacts",
    "_handle_inspect_qe_qmcpack_conversion_deck",
    "_handle_inspect_qe_qmcpack_conversion_atoms",
    "_handle_inspect_qe_qmcpack_conversion_charge",
    "_handle_inspect_qe_qmcpack_conversion_electrons",
    "_handle_inspect_qe_qmcpack_conversion_geometry",
    "_handle_inspect_qe_qmcpack_conversion_ion_species",
    "_handle_inspect_qe_qmcpack_conversion_pseudopotentials",
    "_handle_inspect_qe_qmcpack_conversion_projectors",
    "_handle_inspect_qe_qmcpack_conversion_spin",
    "_handle_inspect_qe_qmcpack_conversion_species",
    "_handle_inspect_qe_qmcpack_conversion_valence",
    "qe_tool_definitions",
]
