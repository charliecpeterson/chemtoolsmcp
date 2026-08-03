"""Regression coverage for bounded HDF5 artifact identification."""

from __future__ import annotations

from pathlib import Path

from chemtools.programs.qe.qmcpack import inspect_pwscf_h5_artifact


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def test_pwscf_h5_artifact_requires_hdf5_signature_at_superblock_offset(tmp_path):
    qe_input = _write(tmp_path, "qe.in", "input")
    qe_output = _write(tmp_path, "qe.out", "output")
    artifact = tmp_path / "orbitals.pwscf.h5"
    artifact.write_bytes(b"not an hdf5 file")

    invalid = inspect_pwscf_h5_artifact(artifact, qe_input, qe_output)

    assert invalid["status"] == "not_ready"
    assert invalid["observed"]["hdf5_signature_offset"] is None

    artifact.write_bytes(b"x" * 512 + _HDF5_SIGNATURE)
    valid_with_user_block = inspect_pwscf_h5_artifact(artifact, qe_input, qe_output)

    assert valid_with_user_block["status"] == "pass"
    assert valid_with_user_block["observed"]["hdf5_signature_offset"] == 512


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path
