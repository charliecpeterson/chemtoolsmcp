"""Pinned source evidence for Orbitron's periodic Python-API summary."""

import hashlib
from pathlib import Path


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "orbitron_periodic" / "vasprun_band_dos.xml"
)


def test_orbitron_periodic_fixture_identity_and_expected_summary():
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == (
        "7da9d64780e54b61fc779d9fae4d8714ef5071cd565b6d88ce0878443cd1f435"
    )
    assert "<i name=\"efermi\"> 1.2 </i>" in FIXTURE_PATH.read_text(
        encoding="utf-8"
    )
    assert FIXTURE_PATH.read_text(encoding="utf-8").count("<r>") == 7
