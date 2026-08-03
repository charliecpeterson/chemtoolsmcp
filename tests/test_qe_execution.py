"""Regression coverage for Quantum ESPRESSO typed launch plans."""

import json

from chemtools.application.qe_execution import render_qe_launch
from chemtools.core.runner import load_runner_profiles
from chemtools.execution import LocalExecutor
from chemtools.programs.qe.launch import (
    adapt_legacy_qe_profile,
    build_qe_launch_plan,
)


def test_local_qe_plan_uses_profile_installation_and_artifacts(tmp_path):
    input_path = tmp_path / "silicon.in"
    input_path.write_text("&CONTROL\n/\n", encoding="utf-8")
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "qe_local": {
                "launcher": {"kind": "direct"},
                "programs": {
                    "qe": {"executable_argv": ["/opt/qe/bin/pw.x"]},
                },
                "resources": {"mpi_ranks": 1, "omp_threads": 4},
                "env": {"OMP_NUM_THREADS": "{omp_threads}"},
                "file_rules": {
                    "output_file": "{job_name}.out",
                    "error_file": "{job_name}.err",
                },
            },
        },
    }), encoding="utf-8")

    profiles = load_runner_profiles(str(profile_path))
    adapted = adapt_legacy_qe_profile(
        profiles,
        "qe_local",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_qe_launch_plan(
        input_path,
        adapted.default_resources,
        output_template=adapted.output_template,
        error_template=adapted.error_template,
    )

    rendered = LocalExecutor().render(plan, adapted.target)

    assert rendered.argv == ("/opt/qe/bin/pw.x", "-in", "silicon.in")
    assert rendered.environment == {"OMP_NUM_THREADS": "4"}
    assert rendered.stdout_path == tmp_path / "silicon.out"
    assert rendered.stderr_path == tmp_path / "silicon.err"
    assert [artifact.kind for artifact in plan.expected_artifacts] == [
        "qe.output",
        "qe.error",
    ]

    preview, _ = render_qe_launch(
        input_path=str(input_path),
        profile="qe_local",
        profiles_path=str(profile_path),
        env_overrides={"QE_TRACE": "1"},
    )

    assert preview["environment"] == {
        "OMP_NUM_THREADS": "4",
        "QE_TRACE": "1",
    }
    assert preview["command"] == (
        "/opt/qe/bin/pw.x -in silicon.in > "
        f"{tmp_path / 'silicon.out'} 2> {tmp_path / 'silicon.err'}"
    )
