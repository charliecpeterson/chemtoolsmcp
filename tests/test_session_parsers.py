"""Unit tests for parser / verdict logic added during the 2026 dogfood sweep.

Inline fixtures (no external corpus) so a clean clone can run them. These pin
the behavior most likely to regress silently: relativistic correlation energies,
the open-shell COSCI resolution table, the active-space verdict thresholds, and
the multiplicity-scan recommender.
"""
from chemtools.programs.dirac.parse.relccsd import parse_relccsd
from chemtools.programs.dirac.parse.output import parse_cosci_energies
from chemtools.programs.molcas.strategy.active_space import (
    _classify_orbitals,
    _verdict_from_quality,
)
from chemtools.programs.nwchem.strategy.input_advisors import recommend_multiplicity_scan


RELCCSD_OUT = """
@ SCF energy :                              -112.822297403801542
@ MP2 correlation energy :                    -0.297454376874968
@ CCSD correlation energy :                   -0.305554660559528
@ 5th order triples (T) correction :           0.001710981618592
@ Total CCSD(T) energy :                    -113.140209506332639
"""


def test_relccsd_energies_and_totals():
    r = parse_relccsd(RELCCSD_OUT)
    assert r["available"]
    assert abs(r["mp2_correlation_hartree"] - (-0.297454376874968)) < 1e-12
    assert abs(r["ccsd_t_total_hartree"] - (-113.140209506332639)) < 1e-9
    # Totals are derived (SCF + correlation), not parsed — guard the arithmetic.
    assert abs(r["mp2_total_hartree"] - (-112.822297403801542 - 0.297454376874968)) < 1e-9


def test_relccsd_absent_returns_unavailable():
    assert parse_relccsd("no correlation here\n")["available"] is False


COSCI_RESOLVE_OUT = """
    ******************** Resolution of open-shell states ********************
 Level  eigenvalue (eV)  Eigenvalue (cm-1)    0g|  2g|
    1        0.000000000          0.000000     1|   0|
    2        1.285366965      10367.183733     0|   1|
"""


def test_cosci_resolve_states():
    r = parse_cosci_energies(COSCI_RESOLVE_OUT)
    assert r["n_states"] == 2
    assert abs(r["states"][1]["energy_cm1"] - 10367.183733) < 1e-3


def _verdict(occupations):
    c = _classify_orbitals(occupations)
    per_root = [{"n_active": len(occupations), "n_truly_active": c["counts"]["truly_active"]}]
    return _verdict_from_quality(per_root)


def test_active_space_verdict_healthy_closed_shell_cas():
    # Textbook pi CAS (bonding ~1.9 / antibonding ~0.1) must read healthy, not
    # poor — the threshold-tuning fix this regression-guards.
    assert _verdict([1.99, 1.95, 1.91, 0.09, 0.05]) == "healthy"


def test_active_space_verdict_poor_when_all_inert():
    assert _verdict([2.0, 2.0, 0.0, 0.0]) == "poor"


def test_multiplicity_scan_warranted_for_open_shell_metal():
    r = recommend_multiplicity_scan(["Fe", "O"], charge=0, current_multiplicity=3)
    assert r["scan_warranted"]
    assert r["recommended_multiplicities"] == [1, 3, 5, 7]


def test_multiplicity_scan_not_warranted_for_closed_shell():
    r = recommend_multiplicity_scan(["O", "H", "H"], charge=0, current_multiplicity=1)
    assert not r["scan_warranted"]
