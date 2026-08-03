"""Contracts for bounded PySCF companion-runtime execution."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemtools.application import pyscf_execution


def _interpreter(tmp_path: Path) -> Path:
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    return interpreter


def _arguments(tmp_path: Path) -> dict:
    return {
        "atoms": [
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
        ],
        "charge": 0,
        "multiplicity": 1,
        "method": "rhf",
        "basis": "sto-3g",
        "working_directory": str(tmp_path),
    }


def test_render_pyscf_single_point_uses_fixed_companion_contract(
    tmp_path,
    monkeypatch,
):
    interpreter = _interpreter(tmp_path)
    runner = tmp_path / "science_runner.py"
    runner.write_text("# fixed runner\n")
    monkeypatch.setattr(
        pyscf_execution,
        "resolve_science_runtime_python",
        lambda: interpreter,
    )
    monkeypatch.setattr(pyscf_execution, "science_runner_path", lambda: runner)

    plan, target, preview = pyscf_execution.render_pyscf_single_point(
        **_arguments(tmp_path),
    )

    assert target.executor == "local"
    assert target.allowed_work_roots == (tmp_path.resolve(),)
    assert plan.program_arguments == (str(runner), "pyscf-single-point")
    assert plan.environment["PYSCF_TMPDIR"] == str(tmp_path.resolve())
    assert preview["command"] == [str(interpreter), str(runner), "pyscf-single-point"]


def test_render_pyscf_single_point_can_request_a_fixed_density_cube(
    tmp_path,
    monkeypatch,
):
    interpreter = _interpreter(tmp_path)
    runner = tmp_path / "science_runner.py"
    runner.write_text("# fixed runner\n")
    monkeypatch.setattr(
        pyscf_execution,
        "resolve_science_runtime_python",
        lambda: interpreter,
    )
    monkeypatch.setattr(pyscf_execution, "science_runner_path", lambda: runner)

    plan, _, _ = pyscf_execution.render_pyscf_single_point(
        **_arguments(tmp_path),
        density_cube_grid_points=80,
    )

    assert json.loads(plan.stdin_text)["density_cube_grid_points"] == 80


def test_render_pyscf_single_point_rejects_an_unbounded_density_cube(
    tmp_path,
):
    with pytest.raises(ValueError, match="between 20 and 120"):
        pyscf_execution.render_pyscf_single_point(
            **_arguments(tmp_path),
            density_cube_grid_points=121,
        )


def test_render_pyscf_single_point_can_request_selected_orbital_cubes(
    tmp_path,
    monkeypatch,
):
    interpreter = _interpreter(tmp_path)
    runner = tmp_path / "science_runner.py"
    runner.write_text("# fixed runner\n")
    monkeypatch.setattr(
        pyscf_execution,
        "resolve_science_runtime_python",
        lambda: interpreter,
    )
    monkeypatch.setattr(pyscf_execution, "science_runner_path", lambda: runner)

    plan, _, _ = pyscf_execution.render_pyscf_single_point(
        **_arguments(tmp_path),
        orbital_cube_grid_points=80,
        orbital_cube_requests=[{"spin": "restricted", "orbital_index": 0}],
    )

    assert json.loads(plan.stdin_text)["orbital_cube_requests"] == [{
        "spin": "restricted",
        "orbital_index": 0,
    }]


def test_run_pyscf_single_point_keeps_execution_and_scf_results_separate(
    tmp_path,
    monkeypatch,
):
    interpreter = _interpreter(tmp_path)
    runner = tmp_path / "science_runner.py"
    runner.write_text("# fixed runner\n")
    monkeypatch.setattr(
        pyscf_execution,
        "resolve_science_runtime_python",
        lambda: interpreter,
    )
    monkeypatch.setattr(pyscf_execution, "science_runner_path", lambda: runner)
    runner_result = {
        "schema_version": "chemtools.pyscf-single-point-result/1",
        "status": "completed",
        "scf": {"converged": False},
    }

    class Service:
        def run_to_completion(self, plan, target):
            return SimpleNamespace(
                record=SimpleNamespace(
                    launch_id="123e4567-e89b-12d3-a456-426614174000",
                    argv=(str(interpreter), str(runner), "pyscf-single-point"),
                ),
                result=SimpleNamespace(
                    status="completed",
                    return_code=0,
                    elapsed_seconds=0.5,
                    stdout=(
                        "CHEMTOOLS_SCIENCE_RESULT="
                        + __import__("json").dumps(runner_result)
                    ),
                    stderr="",
                ),
            )

    response = pyscf_execution.run_pyscf_single_point(
        Service(),
        **_arguments(tmp_path),
    )

    assert response["execution"]["status"] == "completed"
    assert response["result"]["scf"]["converged"] is False
