"""Tests for the optional companion scientific-runtime probe."""

import json
import subprocess

import pytest

from chemtools.integrations import science_runtime


def _executable(tmp_path, name="python"):
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _probe_payload(**overrides):
    payload = {
        "schema_version": science_runtime.SCIENCE_RUNTIME_PROBE_SCHEMA,
        "python": {
            "executable": "/opt/chemtools-science/bin/python",
            "implementation": "cpython",
            "version": "3.12.11",
        },
        "packages": {
            name: {"status": "available", "version": version}
            for name, version in {
                "pyscf": "2.11.0",
                "rdkit": "2026.03.4",
                "openbabel": "3.1.1",
                "h5py": "3.15.1",
                "basis_set_exchange": "0.12",
                "orbitron": "0.4.0",
            }.items()
        },
    }
    payload.update(overrides)
    return payload


def test_resolution_requires_an_explicit_configured_path(tmp_path, monkeypatch):
    interpreter = _executable(tmp_path)
    monkeypatch.setenv(
        science_runtime.SCIENCE_RUNTIME_PYTHON_ENV,
        str(interpreter),
    )

    assert science_runtime.resolve_science_runtime_python() == interpreter.resolve()

    monkeypatch.setenv(science_runtime.SCIENCE_RUNTIME_PYTHON_ENV, "python")
    with pytest.raises(science_runtime.ScienceRuntimeUnavailableError):
        science_runtime.resolve_science_runtime_python()


def test_probe_uses_only_the_fixed_command_and_parses_the_contract(
    tmp_path,
    monkeypatch,
):
    interpreter = _executable(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        stdout = "noise from an optional dependency\n"
        stdout += science_runtime._PROBE_SENTINEL + json.dumps(_probe_payload())
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(science_runtime.subprocess, "run", fake_run)

    probe = science_runtime.ScienceRuntimeClient(
        interpreter,
        timeout_seconds=7,
    ).probe()

    assert probe.python["version"] == "3.12.11"
    assert probe.packages["orbitron"] == {
        "status": "available",
        "version": "0.4.0",
    }
    assert calls[0][0][:2] == [str(interpreter.resolve()), "-c"]
    assert calls[0][0][2] == science_runtime._PROBE_SCRIPT
    assert calls[0][1] == {
        "capture_output": True,
        "text": True,
        "timeout": 7,
        "check": False,
    }


def test_probe_rejects_an_unexpected_package_contract(tmp_path, monkeypatch):
    interpreter = _executable(tmp_path)
    payload = _probe_payload()
    del payload["packages"]["orbitron"]

    monkeypatch.setattr(
        science_runtime.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            0,
            science_runtime._PROBE_SENTINEL + json.dumps(payload),
            "",
        ),
    )

    with pytest.raises(science_runtime.ScienceRuntimeProtocolError):
        science_runtime.ScienceRuntimeClient(interpreter).probe()


def test_probe_reports_fixed_command_failure(tmp_path, monkeypatch):
    interpreter = _executable(tmp_path)
    monkeypatch.setattr(
        science_runtime.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            23,
            "",
            "missing shared library",
        ),
    )

    with pytest.raises(science_runtime.ScienceRuntimeCommandError) as error:
        science_runtime.ScienceRuntimeClient(interpreter).probe()

    assert error.value.returncode == 23
    assert error.value.stderr == "missing shared library"


def test_rdkit_preflight_uses_the_fixed_runner_command(tmp_path, monkeypatch):
    interpreter = _executable(tmp_path)
    calls = []
    payload = {
        "schema_version": "chemtools.rdkit-preflight-result/1",
        "status": "valid",
    }

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            science_runtime._RUNNER_RESULT_SENTINEL + json.dumps(payload),
            "",
        )

    monkeypatch.setattr(science_runtime.subprocess, "run", fake_run)

    result = science_runtime.ScienceRuntimeClient(interpreter).rdkit_preflight({
        "schema_version": "chemtools.rdkit-preflight-request/1",
        "format": "smiles",
        "source": "O",
    })

    assert result == payload
    assert calls[0][0] == [
        str(interpreter.resolve()),
        str(science_runtime.science_runner_path()),
        "rdkit-preflight",
    ]
    assert calls[0][1]["input"] == (
        '{"format": "smiles", "schema_version": '
        '"chemtools.rdkit-preflight-request/1", "source": "O"}'
    )


def test_openbabel_conversion_uses_the_fixed_runner_command(tmp_path, monkeypatch):
    interpreter = _executable(tmp_path)
    calls = []
    payload = {
        "schema_version": "chemtools.openbabel-conversion-result/1",
        "status": "completed",
    }

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            science_runtime._RUNNER_RESULT_SENTINEL + json.dumps(payload),
            "",
        )

    monkeypatch.setattr(science_runtime.subprocess, "run", fake_run)

    result = science_runtime.ScienceRuntimeClient(interpreter).openbabel_convert({
        "schema_version": "chemtools.openbabel-conversion-request/1",
        "format": "smiles",
        "source": "O",
        "output_format": "molblock",
    })

    assert result == payload
    assert calls[0][0] == [
        str(interpreter.resolve()),
        str(science_runtime.science_runner_path()),
        "openbabel-convert",
    ]
    assert calls[0][1]["input"] == (
        '{"format": "smiles", "output_format": "molblock", '
        '"schema_version": "chemtools.openbabel-conversion-request/1", '
        '"source": "O"}'
    )


def test_bse_render_uses_the_fixed_runner_command(tmp_path, monkeypatch):
    interpreter = _executable(tmp_path)
    calls = []
    payload = {
        "schema_version": "chemtools.bse-render-result/1",
        "status": "completed",
    }

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            science_runtime._RUNNER_RESULT_SENTINEL + json.dumps(payload),
            "",
        )

    monkeypatch.setattr(science_runtime.subprocess, "run", fake_run)

    result = science_runtime.ScienceRuntimeClient(interpreter).bse_render({
        "schema_version": "chemtools.bse-render-request/1",
        "basis": "def2-SVP",
        "elements": ["O"],
        "program_format": "nwchem",
    })

    assert result == payload
    assert calls[0][0] == [
        str(interpreter.resolve()),
        str(science_runtime.science_runner_path()),
        "bse-render",
    ]


def test_orbitron_periodic_inspection_uses_the_fixed_runner_command(
    tmp_path,
    monkeypatch,
):
    interpreter = _executable(tmp_path)
    calls = []
    payload = {
        "schema_version": "chemtools.orbitron-periodic-electronic-structure-result/1",
        "status": "completed",
    }

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            science_runtime._RUNNER_RESULT_SENTINEL + json.dumps(payload),
            "",
        )

    monkeypatch.setattr(science_runtime.subprocess, "run", fake_run)

    result = science_runtime.ScienceRuntimeClient(
        interpreter
    ).orbitron_periodic_electronic_structure({
        "schema_version": "chemtools.orbitron-periodic-electronic-structure-request/1",
        "path": "/work/vasprun.xml",
    })

    assert result == payload
    assert calls[0][0] == [
        str(interpreter.resolve()),
        str(science_runtime.science_runner_path()),
        "orbitron-periodic-electronic-structure",
    ]
    assert calls[0][1]["input"] == (
        '{"path": "/work/vasprun.xml", "schema_version": '
        '"chemtools.orbitron-periodic-electronic-structure-request/1"}'
    )


def test_orbitron_structure_identity_uses_the_fixed_runner_command(
    tmp_path,
    monkeypatch,
):
    interpreter = _executable(tmp_path)
    calls = []
    payload = {
        "schema_version": "chemtools.orbitron-structure-identity-result/1",
        "status": "completed",
    }

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            science_runtime._RUNNER_RESULT_SENTINEL + json.dumps(payload),
            "",
        )

    monkeypatch.setattr(science_runtime.subprocess, "run", fake_run)

    result = science_runtime.ScienceRuntimeClient(interpreter).orbitron_structure_identity({
        "schema_version": "chemtools.orbitron-structure-identity-request/1",
        "path": "/work/zncl2.xyz",
    })

    assert result == payload
    assert calls[0][0] == [
        str(interpreter.resolve()),
        str(science_runtime.science_runner_path()),
        "orbitron-structure-identity",
    ]
    assert calls[0][1]["input"] == (
        '{"path": "/work/zncl2.xyz", "schema_version": '
        '"chemtools.orbitron-structure-identity-request/1"}'
    )


def test_orbitron_nbo_uses_the_fixed_runner_command(tmp_path, monkeypatch):
    interpreter = _executable(tmp_path)
    calls = []
    payload = {
        "schema_version": "chemtools.orbitron-nbo-result/1",
        "status": "completed",
    }

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            science_runtime._RUNNER_RESULT_SENTINEL + json.dumps(payload),
            "",
        )

    monkeypatch.setattr(science_runtime.subprocess, "run", fake_run)

    result = science_runtime.ScienceRuntimeClient(interpreter).orbitron_nbo({
        "schema_version": "chemtools.orbitron-nbo-request/1",
        "path": "/work/uo2-test.nbo",
    })

    assert result == payload
    assert calls[0][0] == [
        str(interpreter.resolve()),
        str(science_runtime.science_runner_path()),
        "orbitron-nbo",
    ]
    assert calls[0][1]["input"] == (
        '{"path": "/work/uo2-test.nbo", "schema_version": '
        '"chemtools.orbitron-nbo-request/1"}'
    )


def test_runner_result_requires_one_sentinel_line():
    with pytest.raises(science_runtime.ScienceRuntimeProtocolError):
        science_runtime.parse_science_runner_output("{}")
