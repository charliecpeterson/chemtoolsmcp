"""Contract tests for the fixed, read-only Orbitron subprocess boundary."""

import json
import subprocess
from pathlib import Path

import pytest

from chemtools.core.units import HARTREE_TO_EV
from chemtools.integrations import orbitron


def _executable(tmp_path, name):
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _envelope(schema, **fields):
    return {
        "schema": schema,
        "producer": {
            "name": "orbitron",
            "version": "0.4.0",
            "commit": "58aa65b3f280",
        },
        "warnings": [],
        **fields,
    }


def _geometry_envelope(source, **overrides):
    fields = {
        "path": str(source.resolve()),
        "format": "xyz",
        "geometry_role": "input",
        "geometry_source": "the geometry stored in the input file",
        "distance_unit": "angstrom",
        "atoms": 3,
        "bonds": 2,
        "elements": {"H": 2, "O": 1},
        "coordination": {"H:1": 2, "O:2": 1},
        "dangling_atoms": 0,
        "bond_lengths": {
            "count": 2,
            "min": 0.95,
            "max": 0.96,
            "mean": 0.955,
            "std_dev": 0.005,
        },
        "bounding_box": {
            "min": [-0.75, 0.0, -0.47],
            "max": [0.75, 0.0, 0.12],
        },
        "center": [0.0, 0.0, -0.175],
        "span": [1.5, 0.0, 0.59],
        "unit_cell": None,
        **overrides,
    }
    return _envelope("orbitron.analyze.geometry/3", **fields)


def _orbital_entry(label, vector, energy_hartree, occupancy):
    return {
        "label": label,
        "vector": vector,
        "energy_hartree": energy_hartree,
        "energy_ev": energy_hartree * HARTREE_TO_EV,
        "occupancy": occupancy,
        "symmetry": "a",
        "spin": None,
    }


def _orbital_envelope(source, **overrides):
    homo = _orbital_entry("HOMO", 1, -0.4, 2.0)
    lumo = _orbital_entry("LUMO", 2, 0.1, 0.0)
    fields = {
        "path": str(source.resolve()),
        "format": "out",
        "total_orbitals": 2,
        "occupied_count": 1,
        "virtual_count": 1,
        "homo": homo,
        "lumo": lumo,
        "gap_hartree": 0.5,
        "gap_ev": 0.5 * HARTREE_TO_EV,
        "frontier": [homo, lumo],
        "unrestricted": False,
        "occupied_threshold": 0.1,
        "spin_channels": [
            {
                "spin": "restricted",
                "orbital_count": 2,
                "occupied_count": 1,
                "virtual_count": 1,
                "homo": homo,
                "lumo": lumo,
                "gap_hartree": 0.5,
                "gap_ev": 0.5 * HARTREE_TO_EV,
                "frontier": [homo, lumo],
            }
        ],
        **overrides,
    }
    return _envelope("orbitron.analyze.orbitals/2", **fields)


def _atom_charge(atom_index, element, charge):
    return {
        "atom_index": atom_index,
        "element": element,
        "charge": charge,
    }


def _population_envelope(source, **method_overrides):
    charges = [
        _atom_charge(1, "O", -0.8),
        _atom_charge(2, "H", 0.4),
        _atom_charge(3, "H", 0.4),
    ]
    method = {
        "method": "Mulliken",
        "atom_count": 3,
        "total_charge": 0.0,
        "expected_total_charge": None,
        "expected_charge_source": None,
        "charge_residual": None,
        "min_charge": -0.8,
        "max_charge": 0.4,
        "mean_abs_charge": 1.6 / 3,
        "charges": charges,
        "charges_by_atom": {
            str(entry["atom_index"]): entry for entry in charges
        },
        "top_charges": charges,
        "warnings": [],
        **method_overrides,
    }
    return _envelope(
        "orbitron.analyze.populations/2",
        path=str(source.resolve()),
        format="log",
        methods=[method],
    )


def test_resolution_prefers_explicit_path(tmp_path, monkeypatch):
    explicit = _executable(tmp_path, "explicit-orbitron")
    configured = _executable(tmp_path, "configured-orbitron")
    monkeypatch.setenv(orbitron.ORBITRON_CLI_ENV, str(configured))

    assert orbitron.resolve_orbitron_cli(explicit) == explicit.resolve()


def test_resolution_uses_environment_before_path(tmp_path, monkeypatch):
    configured = _executable(tmp_path, "configured-orbitron")
    path_binary = _executable(tmp_path, "path-orbitron")
    monkeypatch.setenv(orbitron.ORBITRON_CLI_ENV, str(configured))
    monkeypatch.setattr(orbitron.shutil, "which", lambda name: str(path_binary))

    assert orbitron.resolve_orbitron_cli() == configured.resolve()


def test_resolution_falls_back_to_path(tmp_path, monkeypatch):
    path_binary = _executable(tmp_path, "path-orbitron")
    monkeypatch.delenv(orbitron.ORBITRON_CLI_ENV, raising=False)
    monkeypatch.setattr(orbitron.shutil, "which", lambda name: str(path_binary))

    assert orbitron.resolve_orbitron_cli() == path_binary.resolve()


def test_info_probes_once_and_uses_fixed_argv(tmp_path, monkeypatch):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "fe.scf.out"
    source.write_text("Quantum ESPRESSO output\n")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "orbitron-cli 0.4.0 (58aa65b3f280)\nBuilt with: rustc\n",
                "",
            )
        payload = _envelope("orbitron.info/2", atoms=2)
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)
    client = orbitron.OrbitronClient(binary, timeout_seconds=12)

    first = client.info(source)
    second = client.info(source)

    assert first.schema == "orbitron.info/2"
    assert first.payload["atoms"] == 2
    assert first.version.commit == "58aa65b3f280"
    assert second.version == first.version
    assert len(calls) == 3
    assert calls[0][0] == [str(binary.resolve()), "--version"]
    assert calls[1][0] == [
        str(binary.resolve()),
        "--quiet",
        "--max-file-size",
        str(orbitron.MAX_ORBITRON_SOURCE_BYTES),
        "info",
        str(source.resolve()),
        "--json",
    ]
    assert calls[1][1] == {
        "capture_output": True,
        "text": True,
        "timeout": 12,
        "check": False,
    }


def test_geometry_analysis_uses_fixed_argv_and_validates_payload(
    tmp_path,
    monkeypatch,
):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "water.xyz"
    source.write_text("3\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_geometry_envelope(source))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    analyzed = orbitron.OrbitronClient(binary).analyze_geometry(source)

    assert analyzed.operation == "analyze_geometry"
    assert analyzed.schema == "orbitron.analyze.geometry/3"
    assert analyzed.payload["elements"] == {"H": 2, "O": 1}
    assert calls[1] == [
        str(binary.resolve()),
        "--quiet",
        "--max-file-size",
        str(orbitron.MAX_ORBITRON_SOURCE_BYTES),
        "analyze",
        "geometry",
        str(source.resolve()),
        "--json",
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"elements": {"H": 2}},
            "element counts do not equal atoms",
        ),
        (
            {"coordination": {"H:1": 2}},
            "coordination counts do not equal atoms",
        ),
        (
            {"bond_lengths": {"count": 1}},
            "bond-length count does not equal bonds",
        ),
        (
            {"span": [1.0, -1.0, 1.0]},
            "span must be non-negative",
        ),
        (
            {"distance_unit": "bohr"},
            "distance_unit must be angstrom",
        ),
        (
            {"geometry_role": "unknown"},
            "geometry_role must be one of",
        ),
        (
            {"geometry_source": ""},
            "geometry_source must be a non-empty string",
        ),
    ),
)
def test_geometry_analysis_rejects_contradictory_payloads(
    tmp_path,
    monkeypatch,
    overrides,
    message,
):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "water.xyz"
    source.write_text("geometry\n")

    def fake_run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_geometry_envelope(source, **overrides))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(orbitron.OrbitronProtocolError, match=message):
        orbitron.OrbitronClient(binary).analyze_geometry(source)


def test_orbital_analysis_uses_fixed_frontier_and_validates_payload(
    tmp_path,
    monkeypatch,
):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "run.out"
    source.write_text("orbitals\n")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_orbital_envelope(source))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    analyzed = orbitron.OrbitronClient(binary).analyze_orbitals(source)

    assert analyzed.operation == "analyze_orbitals"
    assert analyzed.schema == "orbitron.analyze.orbitals/2"
    assert analyzed.payload["gap_hartree"] == 0.5
    assert calls[1] == [
        str(binary.resolve()),
        "--quiet",
        "--max-file-size",
        str(orbitron.MAX_ORBITRON_SOURCE_BYTES),
        "analyze",
        "orbitals",
        str(source.resolve()),
        "--json",
        "--frontier",
        "3",
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"virtual_count": 2},
            "occupied and virtual counts do not equal total_orbitals",
        ),
        (
            {"gap_ev": 99.0},
            "top-level gap_ev does not match the restricted channel",
        ),
        (
            {"frontier": []},
            "frontier does not match the spin-channel frontiers",
        ),
        (
            {"spin_channels": []},
            "spin channels must be",
        ),
    ),
)
def test_orbital_analysis_rejects_contradictory_payloads(
    tmp_path,
    monkeypatch,
    overrides,
    message,
):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "run.out"
    source.write_text("orbitals\n")

    def fake_run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_orbital_envelope(source, **overrides))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(orbitron.OrbitronProtocolError, match=message):
        orbitron.OrbitronClient(binary).analyze_orbitals(source)


def test_population_analysis_uses_fixed_top_count_and_validates_payload(
    tmp_path,
    monkeypatch,
):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "water.log"
    source.write_text("population analysis\n")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_population_envelope(source))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    analyzed = orbitron.OrbitronClient(binary).analyze_populations(source)

    assert analyzed.operation == "analyze_populations"
    assert analyzed.schema == "orbitron.analyze.populations/2"
    assert analyzed.payload["methods"][0]["total_charge"] == 0.0
    assert calls[1] == [
        str(binary.resolve()),
        "--quiet",
        "--max-file-size",
        str(orbitron.MAX_ORBITRON_SOURCE_BYTES),
        "analyze",
        "populations",
        str(source.resolve()),
        "--json",
        "--top",
        "8",
    ]


@pytest.mark.parametrize(
    ("method_overrides", "message"),
    (
        (
            {"atom_count": 4},
            "charges length does not equal atom_count",
        ),
        (
            {"total_charge": 1.0},
            "total_charge does not match its derived value",
        ),
        (
            {"charges_by_atom": {}},
            "charges_by_atom does not match charges",
        ),
        (
            {"top_charges": []},
            "top_charges does not match the fixed top-eight window",
        ),
        (
            {
                "expected_total_charge": 1.0,
                "expected_charge_source": "declared",
                "charge_residual": 0.0,
            },
            "charge_residual does not match its derived value",
        ),
    ),
)
def test_population_analysis_rejects_contradictory_payloads(
    tmp_path,
    monkeypatch,
    method_overrides,
    message,
):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "water.log"
    source.write_text("population analysis\n")

    def fake_run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(
                _population_envelope(source, **method_overrides)
            )
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(orbitron.OrbitronProtocolError, match=message):
        orbitron.OrbitronClient(binary).analyze_populations(source)


def test_inspect_rejects_unsupported_schema(tmp_path, monkeypatch):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "run.out"
    source.write_text("output\n")

    def fake_run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_envelope("orbitron.inspect/3"))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(
        orbitron.OrbitronProtocolError,
        match="unsupported Orbitron inspect schema",
    ):
        orbitron.OrbitronClient(binary).inspect(source)


def test_command_failure_preserves_status_and_stderr(tmp_path, monkeypatch):
    binary = _executable(tmp_path, "orbitron")

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 7, "", "bad input\n")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(orbitron.OrbitronCommandError) as caught:
        orbitron.OrbitronClient(binary).probe()

    assert caught.value.returncode == 7
    assert caught.value.stderr == "bad input\n"
    assert caught.value.argv == (str(binary.resolve()), "--version")


def test_invalid_json_is_a_protocol_error(tmp_path, monkeypatch):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "run.out"
    source.write_text("output\n")

    def fake_run(argv, **kwargs):
        stdout = (
            "orbitron-cli 0.4.0 (58aa65b3f280)\n"
            if argv[1:] == ["--version"]
            else "not JSON"
        )
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(orbitron.OrbitronProtocolError, match="returned invalid JSON"):
        orbitron.OrbitronClient(binary).info(source)


def test_dirty_build_commit_is_valid_provenance(tmp_path, monkeypatch):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "run.out"
    source.write_text("output\n")

    def fake_run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280-dirty)\n"
        else:
            payload = _envelope("orbitron.inspect/2")
            payload["producer"]["commit"] = "58aa65b3f280-dirty"
            stdout = json.dumps(payload)
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    inspected = orbitron.OrbitronClient(binary).inspect(source)

    assert inspected.version.commit == "58aa65b3f280-dirty"
    assert inspected.producer["commit"] == "58aa65b3f280-dirty"


@pytest.mark.parametrize(
    "warning",
    [
        "plain text",
        {"source": "unknown", "code": "bad_source"},
        {"source": "loader", "code": "   "},
        {"source": "cli", "code": "notice", "message": 7},
    ],
)
def test_warning_entries_are_validated(tmp_path, monkeypatch, warning):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "run.out"
    source.write_text("output\n")

    def fake_run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            payload = _envelope("orbitron.inspect/2")
            payload["warnings"] = [warning]
            stdout = json.dumps(payload)
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(orbitron.OrbitronProtocolError, match="Orbitron warning 0"):
        orbitron.OrbitronClient(binary).inspect(source)


def test_source_and_json_limits_are_enforced(tmp_path, monkeypatch):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "run.out"
    source.write_text("output\n")

    def fake_run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_envelope("orbitron.inspect/2"))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    with pytest.raises(orbitron.OrbitronCommandError, match="source exceeds"):
        orbitron.OrbitronClient(binary, max_source_bytes=1).inspect(source)
    with pytest.raises(orbitron.OrbitronProtocolError, match="JSON exceeds"):
        orbitron.OrbitronClient(binary, max_json_bytes=10).inspect(source)


def test_render_uses_a_fixed_ephemeral_png_output(tmp_path, monkeypatch):
    binary = _executable(tmp_path, "orbitron")
    source = tmp_path / "molecule.xyz"
    source.write_text("1\nH\nH 0 0 0\n")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                "orbitron-cli 0.4.0 (58aa65b3f280)\n",
                "",
            )
        output = Path(argv[argv.index("--output") + 1])
        output.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            + (1024).to_bytes(4, "big")
            + (768).to_bytes(4, "big")
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(orbitron.subprocess, "run", fake_run)

    rendered = orbitron.OrbitronClient(binary).render(source)

    assert rendered.source == str(source.resolve())
    assert rendered.width == 1024
    assert rendered.height == 768
    assert rendered.image.startswith(b"\x89PNG\r\n\x1a\n")
    render_call = calls[1]
    assert render_call[:6] == [
        str(binary.resolve()),
        "--quiet",
        "--max-file-size",
        str(orbitron.MAX_ORBITRON_SOURCE_BYTES),
        "render",
        str(source.resolve()),
    ]
    assert render_call[-4:] == ["--width", "1024", "--height", "768"]
    assert "--force" not in render_call
    output = Path(render_call[render_call.index("--output") + 1])
    assert output.name == "render.png"
    assert output.parent.parent == source.parent
    assert not output.parent.exists()


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("timeout_seconds", True),
        ("timeout_seconds", float("nan")),
        ("max_source_bytes", 0),
        ("max_json_bytes", 1.5),
    ],
)
def test_client_limits_reject_invalid_values(tmp_path, keyword, value):
    binary = _executable(tmp_path, "orbitron")

    with pytest.raises(ValueError):
        orbitron.OrbitronClient(binary, **{keyword: value})
