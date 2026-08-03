"""Subprocess contracts for Orbitron's fixed vibration analysis."""

import json
import subprocess

import pytest

from chemtools.integrations import orbitron


def _executable(tmp_path):
    path = tmp_path / "orbitron"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _vibration_envelope(source, **overrides):
    modes = [
        {
            "index": 1,
            "frequency": -100.0,
            "magnitude": 100.0,
            "label": "a",
            "has_displacement": True,
        },
        {
            "index": 2,
            "frequency": 200.0,
            "magnitude": 200.0,
            "label": None,
            "has_displacement": True,
        },
        {
            "index": 3,
            "frequency": 300.0,
            "magnitude": 300.0,
            "label": "b",
            "has_displacement": True,
        },
    ]
    return {
        "schema": "orbitron.analyze.vibrations/4",
        "producer": {
            "name": "orbitron",
            "version": "0.4.0",
            "commit": "58aa65b3f280",
        },
        "warnings": [
            {
                "source": "analysis",
                "code": "analysis_note",
                "message": "1 imaginary mode detected",
            }
        ],
        "path": str(source.resolve()),
        "format": "log",
        "geometry_role": "converged_final",
        "geometry_source": "the final converged optimization geometry",
        "mode_set": "raw",
        "frequency_unit": "cm^-1",
        "frequency_scale_factor": 1.0,
        "has_displacements": True,
        "displacement_mode_count": 3,
        "mode_count": 3,
        "imaginary_count": 1,
        "lowest_frequency": -100.0,
        "highest_frequency": 300.0,
        "mean_frequency": 400.0 / 3,
        "modes": modes,
        "thermochemistry": {
            "temperature_kelvin": 298.15,
            "pressure_atm": 1.0,
            "zero_point_correction_kcal_mol": 13.5,
            "thermal_correction_energy_kcal_mol": 15.2,
            "thermal_correction_enthalpy_kcal_mol": 15.8,
            "thermal_correction_gibbs_kcal_mol": 2.5,
            "total_entropy_cal_mol_k": 45.0,
            "cv_total_cal_mol_k": 6.0,
            "molecular_weight_amu": 18.015,
            "symmetry_number": 2,
        },
        **overrides,
    }


def _fake_run(source, calls, overrides):
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:] == ["--version"]:
            stdout = "orbitron-cli 0.4.0 (58aa65b3f280)\n"
        else:
            stdout = json.dumps(_vibration_envelope(source, **overrides))
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    return run


def test_vibration_analysis_uses_fixed_raw_top_ten_and_validates_payload(
    tmp_path,
    monkeypatch,
):
    binary = _executable(tmp_path)
    source = tmp_path / "water.log"
    source.write_text("frequency analysis\n")
    calls = []
    monkeypatch.setattr(
        orbitron.subprocess,
        "run",
        _fake_run(source, calls, {}),
    )

    analyzed = orbitron.OrbitronClient(binary).analyze_vibrations(source)

    assert analyzed.operation == "analyze_vibrations"
    assert analyzed.schema == "orbitron.analyze.vibrations/4"
    assert analyzed.payload["imaginary_count"] == 1
    assert calls[1] == [
        str(binary.resolve()),
        "--quiet",
        "--max-file-size",
        str(orbitron.MAX_ORBITRON_SOURCE_BYTES),
        "analyze",
        "vibrations",
        str(source.resolve()),
        "--json",
        "--top",
        "10",
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"imaginary_count": 4},
            "imaginary_count exceeds mode_count",
        ),
        (
            {"modes": []},
            "modes length does not match the fixed top-ten window",
        ),
        (
            {
                "modes": [
                    {
                        "index": 1,
                        "frequency": -100.0,
                        "magnitude": 99.0,
                        "label": None,
                        "has_displacement": True,
                    },
                    {
                        "index": 2,
                        "frequency": 200.0,
                        "magnitude": 200.0,
                        "label": None,
                        "has_displacement": True,
                    },
                    {
                        "index": 3,
                        "frequency": 300.0,
                        "magnitude": 300.0,
                        "label": None,
                        "has_displacement": True,
                    },
                ]
            },
            "magnitude does not match its derived value",
        ),
        (
            {"mean_frequency": 0.0},
            "mean_frequency does not match its derived value",
        ),
        (
            {"thermochemistry": {"temperature_kelvin": 0.0}},
            "temperature_kelvin must be positive",
        ),
        (
            {"mode_set": "projected"},
            "mode_set must be raw",
        ),
        (
            {"frequency_scale_factor": 0.98},
            "frequency_scale_factor must be 1.0",
        ),
        (
            {"displacement_mode_count": 2},
            "has_displacements disagrees with displacement_mode_count",
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
def test_vibration_analysis_rejects_contradictory_payloads(
    tmp_path,
    monkeypatch,
    overrides,
    message,
):
    binary = _executable(tmp_path)
    source = tmp_path / "water.log"
    source.write_text("frequency analysis\n")
    monkeypatch.setattr(
        orbitron.subprocess,
        "run",
        _fake_run(source, [], overrides),
    )

    with pytest.raises(orbitron.OrbitronProtocolError, match=message):
        orbitron.OrbitronClient(binary).analyze_vibrations(source)
