"""Regression coverage for QMCPACK typed launch plans."""

import json

from chemtools.application.qmcpack_execution import render_qmcpack_launch
from chemtools.execution import LocalExecutor
from chemtools.core.runner import load_runner_profiles
from chemtools.programs.qmcpack.launch import (
    adapt_legacy_qmcpack_profile,
    build_qmcpack_launch_plan,
)


def test_local_qmcpack_plan_uses_profile_installation_and_artifacts(tmp_path):
    input_path = tmp_path / "hydrogen.xml"
    input_path.write_text("<simulation/>", encoding="utf-8")
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({
        "schema_version": "1.0",
        "profiles": {
            "qmcpack_local": {
                "launcher": {"kind": "direct"},
                "programs": {
                    "qmcpack": {"executable_argv": ["/opt/qmcpack/bin/qmcpack"]},
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
    adapted = adapt_legacy_qmcpack_profile(
        profiles,
        "qmcpack_local",
        allowed_work_roots=(tmp_path,),
    )
    plan = build_qmcpack_launch_plan(
        input_path,
        adapted.default_resources,
        output_template=adapted.output_template,
        error_template=adapted.error_template,
        qmcpack_dry_run=True,
    )

    rendered = LocalExecutor().render(plan, adapted.target)

    assert rendered.argv == (
        "/opt/qmcpack/bin/qmcpack",
        "hydrogen.xml",
        "--dryrun",
    )
    assert rendered.environment == {"OMP_NUM_THREADS": "4"}
    assert rendered.stdout_path == tmp_path / "hydrogen.out"
    assert rendered.stderr_path == tmp_path / "hydrogen.err"
    assert [artifact.kind for artifact in plan.expected_artifacts] == [
        "qmcpack.output",
        "qmcpack.error",
    ]

    preview, _ = render_qmcpack_launch(
        input_path=str(input_path),
        profile="qmcpack_local",
        profiles_path=str(profile_path),
        env_overrides={"QMCPACK_TRACE": "1"},
    )

    assert preview["environment"] == {
        "OMP_NUM_THREADS": "4",
        "QMCPACK_TRACE": "1",
    }
