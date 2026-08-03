"""Filesystem staging behavior at the typed execution boundary."""

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

import chemtools.execution._common as execution_common
import chemtools.execution.local as local_execution
import chemtools.execution.slurm as slurm_execution
from chemtools.application.execution import ExecutionService
from chemtools.core.execution import StagedFile
from chemtools.core.runner import load_runner_profiles
from chemtools.execution import WorkRootViolation
from chemtools.programs.nwchem.launch import (
    adapt_legacy_nwchem_profile,
    build_nwchem_launch_plan,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "chemtools"
    / "runner_profiles.example.json"
)


def _plan(tmp_path: Path, profile_name: str):
    input_path = tmp_path / "water.nw"
    input_path.write_text(
        "start water\ngeometry\nO 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    profiles = load_runner_profiles(str(PROFILE_PATH))
    adapted = adapt_legacy_nwchem_profile(
        profiles,
        profile_name,
        allowed_work_roots=(tmp_path,),
    )
    return (
        build_nwchem_launch_plan(
            input_path,
            adapted.default_resources,
        ),
        adapted.target,
    )


def _service(tmp_path: Path) -> ExecutionService:
    return ExecutionService(
        enable_execution=True,
        registry_db_path=tmp_path / "registry.db",
    )


def test_staged_file_requires_boolean_required_flag():
    with pytest.raises(TypeError, match="required must be a boolean"):
        StagedFile(
            source=Path("source.dat"),
            destination=Path("destination.dat"),
            required=1,  # type: ignore[arg-type]
        )


def test_slurm_stages_symlink_before_submission(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "slurm_cpu")
    checkpoint = tmp_path / "atomic.h5"
    checkpoint.write_bytes(b"checkpoint")
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=checkpoint,
                destination=Path("seed.h5"),
                mode="symlink",
            ),
        ),
    )

    def fake_run(argv, **kwargs):
        staged = tmp_path / "seed.h5"
        assert staged.is_symlink()
        assert staged.resolve() == checkpoint
        assert staged.read_bytes() == b"checkpoint"
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="Submitted batch job 81\n",
            stderr="",
        )

    monkeypatch.setattr(slurm_execution.subprocess, "run", fake_run)

    launched = _service(tmp_path).launch(staged_plan, target)

    assert launched.record.job_id == "81"
    assert (tmp_path / "seed.h5").is_symlink()


def test_staging_preflights_all_sources_before_copy(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "local")
    available = tmp_path / "available.dat"
    available.write_bytes(b"available")
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=available,
                destination=Path("first.dat"),
            ),
            StagedFile(
                source=Path("missing.dat"),
                destination=Path("second.dat"),
            ),
        ),
    )

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("missing staging source reached process launch")

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        unexpected_popen,
    )

    with pytest.raises(
        FileNotFoundError,
        match="required staged source does not exist",
    ):
        _service(tmp_path).launch(staged_plan, target)

    assert not (tmp_path / "first.dat").exists()
    assert not (tmp_path / "second.dat").exists()


def test_missing_optional_staged_file_is_skipped(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "local")
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=Path("optional.dat"),
                destination=Path("unused.dat"),
                required=False,
            ),
        ),
    )

    class StartedProcess:
        pid = 6262

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        lambda *args, **kwargs: StartedProcess(),
    )

    launched = _service(tmp_path).launch(staged_plan, target)

    assert launched.record.process_id == 6262
    assert not (tmp_path / "unused.dat").exists()


def test_staging_refuses_existing_destination(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "local")
    source = tmp_path / "new.dat"
    destination = tmp_path / "existing.dat"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=source,
                destination=destination,
            ),
        ),
    )

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("existing destination reached process launch")

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        unexpected_popen,
    )

    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite staged destination",
    ):
        _service(tmp_path).launch(staged_plan, target)

    assert destination.read_bytes() == b"old"


def test_staging_rejects_launch_output_destination(tmp_path, monkeypatch):
    plan, target = _plan(tmp_path, "local")
    source = tmp_path / "seed.dat"
    source.write_bytes(b"seed")
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=source,
                destination=Path("water.out"),
            ),
        ),
    )

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("staging collision reached process launch")

    monkeypatch.setattr(
        local_execution.subprocess,
        "Popen",
        unexpected_popen,
    )

    with pytest.raises(
        ValueError,
        match="staged destination conflicts with launch output",
    ):
        _service(tmp_path).launch(staged_plan, target)

    assert source.read_bytes() == b"seed"
    assert not (tmp_path / "water.out").exists()


def test_render_rejects_staged_source_symlink_escape(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    plan, target = _plan(allowed, "local")
    outside_source = outside / "seed.h5"
    outside_source.write_bytes(b"outside")
    escaped_source = allowed / "escaped.h5"
    escaped_source.symlink_to(outside_source)
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=escaped_source,
                destination=Path("seed.h5"),
            ),
        ),
    )

    with pytest.raises(
        WorkRootViolation,
        match="is outside target 'local' roots",
    ):
        ExecutionService().render(staged_plan, target)


def test_render_rejects_staged_destination_symlink_escape(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    plan, target = _plan(allowed, "local")
    source = allowed / "seed.h5"
    source.write_bytes(b"inside")
    escaped_directory = allowed / "escaped"
    escaped_directory.symlink_to(outside, target_is_directory=True)
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(
                source=source,
                destination=Path("escaped/seed.h5"),
            ),
        ),
    )

    with pytest.raises(
        WorkRootViolation,
        match="is outside target 'local' roots",
    ):
        ExecutionService().render(staged_plan, target)


def test_failed_copy_rolls_back_all_staged_destinations(
    tmp_path,
    monkeypatch,
):
    plan, target = _plan(tmp_path, "local")
    first_source = tmp_path / "first-source.dat"
    second_source = tmp_path / "second-source.dat"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    staged_plan = replace(
        plan,
        staged_files=(
            StagedFile(first_source, Path("first.dat")),
            StagedFile(second_source, Path("second.dat")),
        ),
    )
    original_copy = execution_common.shutil.copyfileobj
    calls = 0

    def fail_second_copy(source_handle, destination_handle):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated copy failure")
        original_copy(source_handle, destination_handle)

    monkeypatch.setattr(
        execution_common.shutil,
        "copyfileobj",
        fail_second_copy,
    )

    with pytest.raises(OSError, match="simulated copy failure"):
        _service(tmp_path).launch(staged_plan, target)

    assert calls == 2
    assert not (tmp_path / "first.dat").exists()
    assert not (tmp_path / "second.dat").exists()
