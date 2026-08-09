"""Scheduler resource discovery stays separate from legacy launching."""

from __future__ import annotations

import subprocess

from chemtools.execution import resource_inspection


def test_slurm_partition_inspection_uses_conservative_node_values(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(resource_inspection.shutil, "which", lambda name: name)

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "192000 48 skx,avx2\n"
                "256000 64 spr,avx512\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(resource_inspection.subprocess, "run", fake_run)

    inspected = resource_inspection.query_partition_specs(
        "compute",
        "slurm",
        cache={},
    )

    assert inspected == {
        "node_memory_mb": 192000,
        "cpus_per_node": 48,
        "cpu_arch": "spr",
        "features": ["avx2", "avx512", "skx", "spr"],
    }
    assert calls == [(
        [
            "sinfo",
            "-p",
            "compute",
            "-o",
            "%m %c %f",
            "--noheader",
        ],
        {
            "capture_output": True,
            "text": True,
            "timeout": 10,
        },
    )]


def test_partition_inspection_uses_supplied_cache(monkeypatch):
    cached = {
        "compute": {
            "node_memory_mb": 64000,
            "cpus_per_node": 32,
            "cpu_arch": "cached",
            "features": [],
        }
    }

    def reject_run(*args, **kwargs):
        raise AssertionError("cached partition reached scheduler query")

    monkeypatch.setattr(resource_inspection.subprocess, "run", reject_run)

    assert resource_inspection.query_partition_specs(
        "compute",
        "slurm",
        cache=cached,
    ) is cached["compute"]
