"""Run one explicitly declared PySCF match against NWChem evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemtools.application.execution import ExecutionService
from chemtools.application.pyscf_execution import (
    render_pyscf_single_point,
    run_pyscf_single_point,
)
from chemtools.core.pyscf_comparison import compare_pyscf_reference_calculation
from chemtools.programs.nwchem.pyscf_reference import (
    draft_nwchem_pyscf_reference,
)


NWCHEM_PYSCF_MATCHED_RUN_SCHEMA = "chemtools.nwchem-pyscf-matched-run/1"


def run_nwchem_pyscf_matched_reference(
    service: ExecutionService,
    *,
    input_path: str,
    output_path: str,
    working_directory: str,
    pyscf_method: str,
    density_fit: bool,
    electron_total: int,
    pyscf_xc: str | None = None,
    reference_density_cube: dict[str, Any] | None = None,
    density_cube_grid_points: int | None = None,
    label: str | None = None,
    max_cycles: int = 100,
    convergence_tolerance: float = 1e-9,
    max_memory_mb: int = 2_048,
    omp_threads: int = 1,
    timeout_seconds: float = 120.0,
    job_name: str = "nwchem_pyscf_match",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compare one explicitly declared PySCF calculation with NWChem evidence.

    The calculation is not started until the NWChem reference draft satisfies
    the existing strict comparison contract.
    """
    draft = draft_nwchem_pyscf_reference(
        input_path,
        output_path=output_path,
        label=label,
        pyscf_method=pyscf_method,
        pyscf_xc=pyscf_xc,
        density_fit=density_fit,
        electron_total=electron_total,
    )
    density_cube = _density_cube_request(
        reference_density_cube,
        density_cube_grid_points,
    )
    if density_cube is not None:
        draft["reference_draft"]["density_cube"] = density_cube
        draft["field_sources"]["density_cube"] = {
            "status": "caller_declared",
            **density_cube,
        }
    response = {
        "schema_version": NWCHEM_PYSCF_MATCHED_RUN_SCHEMA,
        "reference_draft": draft,
    }
    if not draft["comparison_ready"]:
        return {
            **response,
            "status": "reference_incomplete",
            "comparison": None,
        }

    reference = draft["reference_draft"]
    calculation = reference["calculation"]
    pyscf_arguments = {
        "atoms": reference["geometry"],
        "charge": calculation["charge"],
        "multiplicity": calculation["multiplicity"],
        "method": calculation["method"],
        "basis": calculation["basis"],
        "xc": calculation["xc"],
        "density_fit": calculation["density_fit"],
        "working_directory": working_directory,
        "max_cycles": max_cycles,
        "convergence_tolerance": convergence_tolerance,
        "max_memory_mb": max_memory_mb,
        "omp_threads": omp_threads,
        "timeout_seconds": timeout_seconds,
        "job_name": job_name,
    }
    if density_cube is not None:
        pyscf_arguments["density_cube_grid_points"] = density_cube_grid_points
    if dry_run:
        _, _, preview = render_pyscf_single_point(**pyscf_arguments)
        return {
            **response,
            "status": "previewed",
            "pyscf_launch": preview,
            "comparison": None,
        }

    pyscf_run = run_pyscf_single_point(service, **pyscf_arguments)
    response["pyscf_run"] = pyscf_run
    if pyscf_run["execution"]["status"] != "completed":
        return {
            **response,
            "status": "pyscf_execution_failed",
            "comparison": None,
        }
    pyscf_result = pyscf_run.get("result")
    if not isinstance(pyscf_result, dict) or pyscf_result.get("status") != "completed":
        return {
            **response,
            "status": "pyscf_result_unavailable",
            "comparison": None,
        }
    return {
        **response,
        "status": "compared",
        "comparison": compare_pyscf_reference_calculation(
            pyscf_result,
            reference,
        ),
    }


def _density_cube_request(
    reference_density_cube: dict[str, Any] | None,
    density_cube_grid_points: int | None,
) -> dict[str, str] | None:
    if (reference_density_cube is None) != (density_cube_grid_points is None):
        raise ValueError(
            "reference_density_cube and density_cube_grid_points must be supplied together"
        )
    if reference_density_cube is None:
        return None
    if set(reference_density_cube) != {"path", "density_value_unit"}:
        raise ValueError(
            "reference_density_cube must contain path and density_value_unit"
        )
    path = reference_density_cube["path"]
    unit = reference_density_cube["density_value_unit"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("reference_density_cube.path must be a non-empty string")
    if not Path(path).is_file():
        raise ValueError(f"reference_density_cube.path is not a file: {path}")
    if unit not in {"electron_per_bohr3", "electron_per_angstrom3"}:
        raise ValueError("reference_density_cube.density_value_unit is not supported")
    if (
        isinstance(density_cube_grid_points, bool)
        or not isinstance(density_cube_grid_points, int)
        or not 20 <= density_cube_grid_points <= 120
    ):
        raise ValueError(
            "density_cube_grid_points must be an integer between 20 and 120"
        )
    return {
        "path": path,
        "density_value_unit": unit,
    }


__all__ = [
    "NWCHEM_PYSCF_MATCHED_RUN_SCHEMA",
    "run_nwchem_pyscf_matched_reference",
]
