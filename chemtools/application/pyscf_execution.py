"""Run bounded molecular PySCF single points through the companion runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chemtools.application.execution import ExecutionService
from chemtools.core.execution import (
    ExecutionTarget,
    HardwareDescription,
    LaunchPlan,
    ProgramInstallation,
    ResourceRequest,
)
from chemtools.integrations.science_runtime import (
    ScienceRuntimeProtocolError,
    parse_science_runner_output,
    resolve_science_runtime_python,
    science_runner_path,
)
from chemtools.core.common import ELEMENT_TO_Z
from chemtools.science_runner import PYSCF_SINGLE_POINT_REQUEST_SCHEMA


PYSCF_LAUNCH_SCHEMA = "chemtools.pyscf-launch/2"


def render_pyscf_single_point(
    *,
    atoms: list[dict[str, Any]],
    charge: int,
    multiplicity: int,
    method: str,
    basis: str,
    working_directory: str,
    xc: str | None = None,
    density_fit: bool = False,
    max_cycles: int = 100,
    convergence_tolerance: float = 1e-9,
    max_memory_mb: int = 2_048,
    density_cube_grid_points: int | None = None,
    orbital_cube_grid_points: int | None = None,
    orbital_cube_requests: list[dict[str, Any]] | None = None,
    omp_threads: int = 1,
    timeout_seconds: float = 120.0,
    job_name: str = "pyscf_single_point",
) -> tuple[LaunchPlan, ExecutionTarget, dict[str, Any]]:
    work_dir = Path(working_directory).resolve()
    if not work_dir.is_dir():
        raise ValueError(f"working_directory does not exist: {work_dir}")
    if isinstance(omp_threads, bool) or not isinstance(omp_threads, int) or not 1 <= omp_threads <= 128:
        raise ValueError("omp_threads must be an integer between 1 and 128")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 1 <= timeout_seconds <= 3_600
    ):
        raise ValueError("timeout_seconds must be between 1 and 3600 seconds")
    if density_cube_grid_points is not None and (
        isinstance(density_cube_grid_points, bool)
        or not isinstance(density_cube_grid_points, int)
        or not 20 <= density_cube_grid_points <= 120
    ):
        raise ValueError(
            "density_cube_grid_points must be an integer between 20 and 120"
        )
    if orbital_cube_grid_points is not None and (
        isinstance(orbital_cube_grid_points, bool)
        or not isinstance(orbital_cube_grid_points, int)
        or not 20 <= orbital_cube_grid_points <= 120
    ):
        raise ValueError(
            "orbital_cube_grid_points must be an integer between 20 and 120"
        )
    if (orbital_cube_grid_points is None) != (orbital_cube_requests is None):
        raise ValueError(
            "orbital_cube_grid_points and orbital_cube_requests must be supplied together"
        )
    request = {
        "schema_version": PYSCF_SINGLE_POINT_REQUEST_SCHEMA,
        "atoms": atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "method": method,
        "basis": basis,
        "xc": xc,
        "density_fit": density_fit,
        "max_cycles": max_cycles,
        "convergence_tolerance": convergence_tolerance,
        "max_memory_mb": max_memory_mb,
    }
    if density_cube_grid_points is not None:
        request["density_cube_grid_points"] = density_cube_grid_points
    if orbital_cube_grid_points is not None:
        request["orbital_cube_grid_points"] = orbital_cube_grid_points
        request["orbital_cube_requests"] = orbital_cube_requests
    request_text = json.dumps(request, sort_keys=True, allow_nan=False)
    interpreter = resolve_science_runtime_python()
    target = ExecutionTarget(
        name="chemtools-science",
        executor="local",
        allowed_work_roots=(work_dir,),
        hardware=HardwareDescription(cores_per_node=omp_threads),
        programs={
            "pyscf": ProgramInstallation(executable_argv=(str(interpreter),)),
        },
    )
    plan = LaunchPlan(
        job_name=job_name,
        program="pyscf",
        program_arguments=(str(science_runner_path()), "pyscf-single-point"),
        environment={
            "OMP_NUM_THREADS": str(omp_threads),
            "OPENBLAS_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(omp_threads),
            "PYSCF_TMPDIR": str(work_dir),
        },
        working_directory=work_dir,
        staged_files=(),
        expected_artifacts=(),
        resources=ResourceRequest(omp_threads=omp_threads),
        stdin_text=request_text,
        timeout_seconds=float(timeout_seconds),
    )
    preview = {
        "schema_version": PYSCF_LAUNCH_SCHEMA,
        "program": "pyscf",
        "target": target.name,
        "working_directory": str(work_dir),
        "command": [str(interpreter), *plan.program_arguments],
        "environment": dict(plan.environment),
        "request_sha256": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
        "resources": {"omp_threads": omp_threads},
        "timeout_seconds": plan.timeout_seconds,
        **_atomic_state_control(atoms),
    }
    return plan, target, preview


def run_pyscf_single_point(
    service: ExecutionService,
    **arguments: Any,
) -> dict[str, Any]:
    plan, target, preview = render_pyscf_single_point(**arguments)
    executed = service.run_to_completion(plan, target)
    response = {
        **preview,
        "execution": {
            "launch_id": executed.record.launch_id,
            "status": executed.result.status,
            "return_code": executed.result.return_code,
            "elapsed_seconds": executed.result.elapsed_seconds,
            "argv": list(executed.record.argv),
        },
    }
    if executed.result.status != "completed":
        response["result"] = {
            "status": "execution_failed",
            "stdout": executed.result.stdout,
            "stderr": executed.result.stderr,
        }
        return response
    try:
        response["result"] = parse_science_runner_output(executed.result.stdout)
    except ScienceRuntimeProtocolError as error:
        response["result"] = {
            "status": "incompatible",
            "error": "science_runtime_protocol_error",
            "message": str(error),
            "stderr": executed.result.stderr,
        }
    return response


def _atomic_state_control(atoms: list[dict[str, Any]]) -> dict[str, object]:
    if not isinstance(atoms, list) or len(atoms) != 1:
        return {
            "atomic_state_control": {"status": "not_applicable"},
            "warnings": [],
        }
    atom = atoms[0] if isinstance(atoms[0], dict) else {}
    element_value = atom.get("element")
    element = (
        element_value[:1].upper() + element_value[1:].lower()
        if isinstance(element_value, str) and element_value
        else ""
    )
    atomic_number = ELEMENT_TO_Z.get(element)
    warnings = [{
        "code": "atomic_configuration_unconstrained",
        "message": (
            "The bounded PySCF runner sets charge and spin but does not set "
            "atomic symmetry, irrep occupations, or a post-SCF population check."
        ),
    }]
    if atomic_number is not None and (
        57 <= atomic_number <= 71 or 89 <= atomic_number <= 103
    ):
        warnings.append({
            "code": "fblock_catalog_transfer_unsupported",
            "message": (
                "This runner cannot preserve or validate a cataloged f-block "
                "occupation; use the catalog only as a GRASP2018 reference."
            ),
        })
    return {
        "atomic_state_control": {
            "status": "unconstrained",
            "atomic_symmetry": False,
            "irrep_occupations": False,
            "post_scf_population_check": False,
            "catalog_state_supported": False,
        },
        "warnings": warnings,
    }


__all__ = ["render_pyscf_single_point", "run_pyscf_single_point"]
