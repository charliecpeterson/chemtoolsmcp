"""Pinned source evidence for Orbitron's structure-identity Python API."""

import hashlib
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "orbitron_identity" / "zncl2.xyz"


def test_orbitron_identity_fixture_is_pinned_zinc_chloride_input():
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == (
        "bdd9c6c2bf1e578bebd137cb33d02bdd3a3cdd032af6e03dc89957fc063ed8e8"
    )
    assert FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[1] == "Zinc chloride salt"
