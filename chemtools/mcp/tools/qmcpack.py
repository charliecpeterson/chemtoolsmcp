"""QMCPACK input, scalar, population, and pseudopotential analysis tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.application.qmcpack_execution import (
    launch_qmcpack_with_service,
    render_qmcpack_launch,
)
from chemtools.integrations.science_runtime import (
    ScienceRuntimeClient,
    ScienceRuntimeCommandError,
    ScienceRuntimeProtocolError,
    ScienceRuntimeUnavailableError,
)
from chemtools.mcp.decorator import _tool, get_execution_service
from chemtools.programs.qmcpack.dmc import (
    analyze_dmc_input_series,
    analyze_dmc_series,
    compare_tmove_locality_shift,
    compare_tmove_locality_shift_from_input,
    inspect_dmc_population,
    inspect_dmc_population_from_input,
)
from chemtools.programs.qmcpack.gates import (
    check_vmc_energy_gate,
    inspect_determinant_only_vmc_offsets,
)
from chemtools.programs.qmcpack.includes import inspect_xml_includes
from chemtools.programs.qmcpack.input import parse_qmcpack_input
from chemtools.programs.qmcpack.pseudopotential import (
    inspect_qmcpack_pseudopotential,
    inspect_referenced_pseudopotentials,
)
from chemtools.programs.qmcpack.scalar import parse_scalar_file
from chemtools.science_runner import QMCPACK_HDF5_INSPECTION_REQUEST_SCHEMA


@_tool("render_qmcpack_launch", needs="runner_profile", program="qmcpack")
def _handle_render_qmcpack_launch(arguments: dict[str, Any]) -> dict[str, Any]:
    preview, _ = render_qmcpack_launch(
        input_path=arguments["qmcpack_input"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
    )
    return preview


@_tool("launch_qmcpack_run", needs="executable", program="qmcpack")
def _handle_launch_qmcpack_run(arguments: dict[str, Any]) -> dict[str, Any]:
    return launch_qmcpack_with_service(
        get_execution_service(),
        input_path=arguments["qmcpack_input"],
        profile=arguments["profile"],
        profiles_path=arguments.get("profiles_path"),
        job_name=arguments.get("job_name"),
        resource_overrides=arguments.get("resource_overrides"),
        env_overrides=arguments.get("env_overrides"),
        dry_run=arguments.get("dry_run", False),
        qmcpack_dry_run=arguments.get("qmcpack_dry_run", False),
    )


@_tool("inspect_qmcpack_scalar", program="qmcpack")
def _handle_inspect_qmcpack_scalar(arguments: dict[str, Any]) -> dict[str, Any]:
    return parse_scalar_file(arguments["scalar_file"])


@_tool("inspect_qmcpack_hdf5", program="qmcpack")
def _handle_inspect_qmcpack_hdf5(arguments: dict[str, Any]) -> dict[str, Any]:
    request = {
        "schema_version": QMCPACK_HDF5_INSPECTION_REQUEST_SCHEMA,
        "path": arguments["hdf5_file"],
    }
    try:
        return ScienceRuntimeClient().qmcpack_hdf5_inspect(request)
    except ScienceRuntimeUnavailableError as error:
        return _qmcpack_hdf5_runtime_error("unavailable", error)
    except ScienceRuntimeProtocolError as error:
        return _qmcpack_hdf5_runtime_error("incompatible", error)
    except ScienceRuntimeCommandError as error:
        response = _qmcpack_hdf5_runtime_error("tool_refused", error)
        response["returncode"] = error.returncode
        if error.stderr:
            response["stderr"] = error.stderr
        return response


@_tool("inspect_qmcpack_pseudopotential", program="qmcpack")
def _handle_inspect_qmcpack_pseudopotential(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return inspect_qmcpack_pseudopotential(arguments["pseudopotential_file"])


@_tool("inspect_qmcpack_referenced_pseudopotentials", program="qmcpack")
def _handle_inspect_qmcpack_referenced_pseudopotentials(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    source = Path(arguments["qmcpack_input"]).expanduser().resolve()
    parsed_input = parse_qmcpack_input(source)
    include_review = inspect_xml_includes(source, parsed_input)
    inspection = inspect_referenced_pseudopotentials(
        parsed_input,
        include_review,
        source,
    )
    return {
        "schema_version": "chemtools.qmcpack-referenced-pseudopotentials/1",
        "qmcpack_input": str(source),
        "status": inspection["status"],
        "inspection": inspection,
        "scope_limit": (
            "This follows the bounded QMCPACK XML include graph and checks "
            "declared elementType values against pseudopotential header symbols. "
            "It does not establish pseudopotential family equivalence or "
            "transferability."
        ),
    }


@_tool("analyze_qmcpack_dmc_series", program="qmcpack")
def _handle_analyze_qmcpack_dmc_series(arguments: dict[str, Any]) -> dict[str, Any]:
    return analyze_dmc_series(
        arguments["runs"],
        discard_fraction=arguments.get("discard_fraction", 0.25),
        reblock_count=arguments.get("reblock_count", 32),
    )


@_tool("analyze_qmcpack_dmc_input_series", program="qmcpack")
def _handle_analyze_qmcpack_dmc_input_series(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return analyze_dmc_input_series(
        arguments["qmcpack_input"],
        arguments["runs"],
        discard_fraction=arguments.get("discard_fraction", 0.25),
        reblock_count=arguments.get("reblock_count", 32),
    )


@_tool("inspect_qmcpack_dmc_population", program="qmcpack")
def _handle_inspect_qmcpack_dmc_population(arguments: dict[str, Any]) -> dict[str, Any]:
    return inspect_dmc_population(
        arguments["dmc_file"],
        target_walkers=arguments.get("target_walkers"),
        discard_fraction=arguments.get("discard_fraction", 0.25),
    )


@_tool("inspect_qmcpack_dmc_population_from_input", program="qmcpack")
def _handle_inspect_qmcpack_dmc_population_from_input(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return inspect_dmc_population_from_input(
        arguments["qmcpack_input"],
        arguments["dmc_file"],
        arguments["qmc_block_index"],
        discard_fraction=arguments.get("discard_fraction", 0.25),
    )


@_tool("check_qmcpack_vmc_energy_gate", program="qmcpack")
def _handle_check_qmcpack_vmc_energy_gate(arguments: dict[str, Any]) -> dict[str, Any]:
    return check_vmc_energy_gate(
        arguments["scalar_file"],
        trial_scf_energy_hartree=arguments["trial_scf_energy_hartree"],
        discard_fraction=arguments.get("discard_fraction", 0.0),
    )


@_tool("inspect_qmcpack_determinant_vmc_offsets", program="qmcpack")
def _handle_inspect_qmcpack_determinant_vmc_offsets(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return inspect_determinant_only_vmc_offsets(
        arguments["runs"],
        discard_fraction=arguments.get("discard_fraction", 0.0),
    )


@_tool("compare_qmcpack_tmove_locality_shift", program="qmcpack")
def _handle_compare_qmcpack_tmove_locality_shift(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return compare_tmove_locality_shift(
        arguments["tmove"],
        arguments["no_tmove"],
        discard_fraction=arguments.get("discard_fraction", 0.25),
        reblock_count=arguments.get("reblock_count", 32),
    )


@_tool("compare_qmcpack_tmove_locality_shift_from_input", program="qmcpack")
def _handle_compare_qmcpack_tmove_locality_shift_from_input(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return compare_tmove_locality_shift_from_input(
        arguments["qmcpack_input"],
        arguments["tmove"],
        arguments["no_tmove"],
        discard_fraction=arguments.get("discard_fraction", 0.25),
        reblock_count=arguments.get("reblock_count", 32),
    )


def _qmcpack_hdf5_runtime_error(status: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "chemtools.qmcpack-hdf5-inspection/1",
        "status": status,
        "message": str(error),
    }


def qmcpack_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "render_qmcpack_launch",
            "description": (
                "Render the configured QMCPACK local or scheduler command without "
                "starting it. Use this to verify the selected runner profile, "
                "resources, environment, and expected output paths."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qmcpack_input": {
                        "type": "string",
                        "description": "Path to a QMCPACK XML input file.",
                    },
                    "profile": {
                        "type": "string",
                        "description": "Named Chemtools runner profile.",
                    },
                    "profiles_path": {
                        "type": "string",
                        "description": "Optional runner-profile YAML or JSON path.",
                    },
                    "job_name": {
                        "type": "string",
                        "description": "Optional output-file stem.",
                    },
                    "resource_overrides": {
                        "type": "object",
                        "description": "Optional profile resource overrides.",
                    },
                    "env_overrides": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Optional environment-value overrides.",
                    },
                },
                "required": ["qmcpack_input", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "launch_qmcpack_run",
            "description": (
                "Launch QMCPACK with a configured local or Slurm runner profile. "
                "The launch records its effective command, resources, and output "
                "paths. Set dry_run=true to inspect the profile without starting it; "
                "set qmcpack_dry_run=true to initialize QMCPACK while skipping QMC "
                "sections."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qmcpack_input": {"type": "string"},
                    "profile": {"type": "string"},
                    "profiles_path": {"type": "string"},
                    "job_name": {"type": "string"},
                    "resource_overrides": {"type": "object"},
                    "env_overrides": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "dry_run": {"type": "boolean", "default": False},
                    "qmcpack_dry_run": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["qmcpack_input", "profile"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_qmcpack_pseudopotential",
            "description": (
                "Inspect a QMCPACK semilocal pseudopotential XML card. Returns "
                "header metadata, grid details, local channel, per-channel data "
                "counts, final r*V values compared with minus zval, and structural "
                "evidence for the expected semilocal XML form. This does not "
                "establish pseudopotential transferability or DMC compatibility."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pseudopotential_file": {
                        "type": "string",
                        "description": "Path to one QMCPACK pseudopotential XML file.",
                    },
                },
                "required": ["pseudopotential_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_qmcpack_referenced_pseudopotentials",
            "description": (
                "Follow a bounded QMCPACK XML include graph and inspect each "
                "referenced semilocal pseudopotential card. Checks supported XML "
                "structure and whether each declared elementType matches the card's "
                "header symbol. This does not establish pseudopotential family "
                "equivalence or transferability."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qmcpack_input": {
                        "type": "string",
                        "description": "Path to a QMCPACK XML input or included fragment.",
                    },
                },
                "required": ["qmcpack_input"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_qmcpack_scalar",
            "description": (
                "Parse one QMCPACK .scalar.dat block file. Returns the observed "
                "columns, valid block count, malformed-row count, and compact "
                "estimator summaries. LocalEnergy includes a BlockWeight-weighted "
                "mean only when every block has a positive weight. This does not "
                "reblock, estimate autocorrelation, or extrapolate DMC to zero "
                "time step."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scalar_file": {
                        "type": "string",
                        "description": "Path to one QMCPACK .scalar.dat file.",
                    },
                },
                "required": ["scalar_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_qmcpack_hdf5",
            "description": (
                "Inspect one recognized QMCPACK HDF5 artifact through the optional "
                "companion science runtime. It identifies fixed QE/pw2qmcpack "
                "wavefunction, variational-parameter, walker-configuration, and "
                "statistics layouts and returns bounded metadata only. It does not "
                "read orbital coefficients, density grids, walker coordinates, "
                "estimator values, or arbitrary datasets."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hdf5_file": {
                        "type": "string",
                        "description": "Absolute path to one QMCPACK HDF5 artifact.",
                    },
                },
                "required": ["hdf5_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_qmcpack_dmc_series",
            "description": (
                "Reblock LocalEnergy from explicitly labelled QMCPACK DMC scalar "
                "files and perform separate inverse-variance linear fits to zero "
                "time step for T-move and no-T-move groups. Each run must provide "
                "the time step and nonlocalmoves setting from its input, because "
                "sequential scalar filenames do not identify a chained QMC block. "
                "This is a statistical summary, not a convergence verdict."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "runs": {
                        "type": "array",
                        "minItems": 1,
                        "description": "DMC scalar files with controls copied from their input blocks.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "scalar_file": {
                                    "type": "string",
                                    "description": "Path to one DMC .scalar.dat file.",
                                },
                                "timestep": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "description": "DMC time step recorded in the matching input block.",
                                },
                                "nonlocalmoves": {
                                    "type": "boolean",
                                    "description": "The matching DMC block's nonlocalmoves setting.",
                                },
                                "target_walkers": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "description": "Optional targetWalkers value from the matching input block.",
                                },
                                "potential_label": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": (
                                        "Optional caller-supplied potential identity. "
                                        "Provide it for every run to establish "
                                        "same-potential evidence."
                                    ),
                                },
                            },
                            "required": ["scalar_file", "timestep", "nonlocalmoves"],
                            "additionalProperties": False,
                        },
                    },
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0.25,
                        "description": "Leading scalar-block fraction discarded before reblocking.",
                    },
                    "reblock_count": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 32,
                        "description": "Maximum number of contiguous reblocks per run.",
                    },
                },
                "required": ["runs"],
                "additionalProperties": False,
            },
        },
        {
            "name": "analyze_qmcpack_dmc_input_series",
            "description": (
                "Reblock and fit QMCPACK DMC scalar files using the time step, "
                "nonlocalmoves setting, and walker target from explicitly selected "
                "DMC blocks in the primary QMCPACK XML. The caller supplies each "
                "scalar file and QMC-block index because scalar files do not record "
                "their source block. Included XML is not merged, and this does not "
                "prove that an association is correct."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qmcpack_input": {
                        "type": "string",
                        "description": "Path to the primary QMCPACK XML input containing direct DMC blocks.",
                    },
                    "runs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "scalar_file": {
                                    "type": "string",
                                    "description": "Path to one DMC .scalar.dat file.",
                                },
                                "qmc_block_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": "Zero-based direct <qmc> index in the primary XML for the DMC block that produced this file.",
                                },
                                "potential_label": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "Optional caller-supplied potential identity for fit comparability.",
                                },
                            },
                            "required": ["scalar_file", "qmc_block_index"],
                            "additionalProperties": False,
                        },
                    },
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0.25,
                        "description": "Leading scalar-block fraction discarded before reblocking.",
                    },
                    "reblock_count": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 32,
                        "description": "Maximum number of contiguous reblocks per run.",
                    },
                },
                "required": ["qmcpack_input", "runs"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_qmcpack_dmc_population_from_input",
            "description": (
                "Inspect a QMCPACK DMC population file using the walker target "
                "from one selected direct DMC block in the primary QMCPACK XML. The "
                "caller supplies the population-file-to-QMC-block association because "
                "the file does not record its source block. Included XML is not "
                "merged, and this does not prove that association."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qmcpack_input": {
                        "type": "string",
                        "description": "Path to the primary QMCPACK XML input containing the direct DMC block.",
                    },
                    "dmc_file": {
                        "type": "string",
                        "description": "Path to one DMC .dmc.dat population file.",
                    },
                    "qmc_block_index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Zero-based direct <qmc> index in the primary XML for the DMC block that produced this file.",
                    },
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0.25,
                        "description": "Leading DMC-record fraction discarded before summarizing.",
                    },
                },
                "required": ["qmcpack_input", "dmc_file", "qmc_block_index"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_qmcpack_dmc_population",
            "description": (
                "Inspect one QMCPACK .dmc.dat population record after a leading "
                "block discard. Returns NumOfWalkers, LivingFraction, and DiffEff "
                "summaries when present. An optional target_walkers value from the "
                "matching DMC input block adds observed mean and final population "
                "deviations. It also reports block-index continuity and malformed "
                "or truncated input warnings. The tool reports measurements and "
                "does not assign a population-control convergence threshold."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dmc_file": {
                        "type": "string",
                        "description": "Path to one QMCPACK .dmc.dat file.",
                    },
                    "target_walkers": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Optional targetWalkers value from the matching input block.",
                    },
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0.25,
                        "description": "Leading DMC-record fraction discarded before summarizing.",
                    },
                },
                "required": ["dmc_file"],
                "additionalProperties": False,
            },
        },
        {
            "name": "check_qmcpack_vmc_energy_gate",
            "description": (
                "Compare the retained mean QMCPACK VMC LocalEnergy with the "
                "matching trial SCF energy in Hartree. The gate passes only when "
                "VMC is at or below the trial energy, as required before using an "
                "optimized Jastrow for DMC. It reports scalar-input quality warnings "
                "but does not establish autocorrelation or statistical convergence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scalar_file": {
                        "type": "string",
                        "description": "Path to the post-optimization VMC .scalar.dat file.",
                    },
                    "trial_scf_energy_hartree": {
                        "type": "number",
                        "description": "Matching trial SCF energy in Hartree.",
                    },
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0,
                        "description": "Leading scalar-block fraction discarded before comparison.",
                    },
                },
                "required": ["scalar_file", "trial_scf_energy_hartree"],
                "additionalProperties": False,
            },
        },
        {
            "name": "inspect_qmcpack_determinant_vmc_offsets",
            "description": (
                "Summarize determinant-only VMC-minus-trial-SCF offsets across "
                "at least two caller-labelled states. Reports whether all offsets "
                "are positive and their strict trend in the supplied state order. "
                "It reports scalar-input quality warnings but does not set a "
                "small-offset threshold or prove Hamiltonian consistency."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "runs": {
                        "type": "array",
                        "minItems": 2,
                        "description": (
                            "Determinant-only VMC scalar files in the caller's "
                            "expected state order."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "state_label": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": "Caller label for one state.",
                                },
                                "scalar_file": {
                                    "type": "string",
                                    "minLength": 1,
                                    "description": (
                                        "Path to one determinant-only VMC "
                                        ".scalar.dat file."
                                    ),
                                },
                                "trial_scf_energy_hartree": {
                                    "type": "number",
                                    "description": (
                                        "Matching trial SCF energy in Hartree."
                                    ),
                                },
                            },
                            "required": [
                                "state_label",
                                "scalar_file",
                                "trial_scf_energy_hartree",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0,
                        "description": (
                            "Leading scalar-block fraction discarded before "
                            "calculating each offset."
                        ),
                    },
                },
                "required": ["runs"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_qmcpack_tmove_locality_shift_from_input",
            "description": (
                "Compare caller-bound T-move and no-T-move DMC scalar files using "
                "time-step and walker controls from selected direct blocks in the "
                "primary QMCPACK XML. It rejects selected blocks with the wrong "
                "nonlocalmoves setting. Included XML is not merged. The caller "
                "supplies the scalar-file-to-QMC-block bindings, which remain "
                "unverified provenance."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qmcpack_input": {
                        "type": "string",
                        "description": "Path to the primary QMCPACK XML input containing both direct DMC controls.",
                    },
                    "tmove": {"$ref": "#/$defs/inputBoundDmcRun"},
                    "no_tmove": {"$ref": "#/$defs/inputBoundDmcRun"},
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0.25,
                        "description": "Leading scalar-block fraction discarded before reblocking.",
                    },
                    "reblock_count": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 32,
                        "description": "Maximum number of contiguous reblocks per run.",
                    },
                },
                "required": ["qmcpack_input", "tmove", "no_tmove"],
                "additionalProperties": False,
                "$defs": {
                    "inputBoundDmcRun": {
                        "type": "object",
                        "properties": {
                            "scalar_file": {
                                "type": "string",
                                "description": "Path to one DMC .scalar.dat file.",
                            },
                            "qmc_block_index": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Zero-based direct <qmc> index in the primary XML for the DMC block that produced this file.",
                            },
                            "potential_label": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Optional potential identity; supplied labels must match across the pair.",
                            },
                        },
                        "required": ["scalar_file", "qmc_block_index"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        {
            "name": "compare_qmcpack_tmove_locality_shift",
            "description": (
                "Compare matched QMCPACK DMC T-move and no-T-move scalar files "
                "at one time step. Returns the signed locality shift as no-T-move "
                "minus T-move in Hartree, propagated reblocked uncertainty, and "
                "target-walker comparability. The two inputs must record the same "
                "time step. When potential labels are supplied, they must match; "
                "the result does not establish autocorrelation convergence."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tmove": {"$ref": "#/$defs/dmcRun"},
                    "no_tmove": {"$ref": "#/$defs/dmcRun"},
                    "discard_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMaximum": 1,
                        "default": 0.25,
                        "description": "Leading scalar-block fraction discarded before reblocking.",
                    },
                    "reblock_count": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 32,
                        "description": "Maximum number of contiguous reblocks per run.",
                    },
                },
                "required": ["tmove", "no_tmove"],
                "additionalProperties": False,
                "$defs": {
                    "dmcRun": {
                        "type": "object",
                        "properties": {
                            "scalar_file": {
                                "type": "string",
                                "description": "Path to one DMC .scalar.dat file.",
                            },
                            "timestep": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "DMC time step recorded in the matching input block.",
                            },
                            "target_walkers": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "Optional targetWalkers value from the matching input block.",
                            },
                            "potential_label": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Optional potential identity; supplied labels must match across the pair.",
                            },
                        },
                        "required": ["scalar_file", "timestep"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    ]


__all__ = [
    "_handle_analyze_qmcpack_dmc_input_series",
    "_handle_analyze_qmcpack_dmc_series",
    "_handle_launch_qmcpack_run",
    "_handle_render_qmcpack_launch",
    "_handle_check_qmcpack_vmc_energy_gate",
    "_handle_compare_qmcpack_tmove_locality_shift",
    "_handle_compare_qmcpack_tmove_locality_shift_from_input",
    "_handle_inspect_qmcpack_determinant_vmc_offsets",
    "_handle_inspect_qmcpack_dmc_population",
    "_handle_inspect_qmcpack_dmc_population_from_input",
    "_handle_inspect_qmcpack_pseudopotential",
    "_handle_inspect_qmcpack_referenced_pseudopotentials",
    "_handle_inspect_qmcpack_scalar",
    "qmcpack_tool_definitions",
]
