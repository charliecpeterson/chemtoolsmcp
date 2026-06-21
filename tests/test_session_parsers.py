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
from chemtools.programs.grasp.parse.sum_file import parse_sum as parse_grasp_sum
from chemtools.programs.grasp.parse.hfs import parse_hfs
from chemtools.programs.grasp.parse.ris import parse_ris
from chemtools.programs.grasp.parse.transition import parse_transition


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


GRASP_CSUM = """
 There are 90 electrons in the cloud
  in 9 relativistic CSFs
  based on 27 relativistic subshells.

The atomic number is  90.0000000000;

Speed of light =  1.370359991390D+02 atomic units.

 To H (Dirac Coulomb) is added
  H (Transverse) --- factor multiplying the photon frequency:  1.00000000D-06;
  H (Vacuum Polarisation);
  the total will be diagonalised.
 Diagonal contributions from H (Self Energy) will be estimated
  from a screened hydrogenic approximation.
"""

GRASP_SUM_DHF = """
 There are 90 electrons in the cloud
  in 9 relativistic CSFs
  based on 27 relativistic subshells.

The atomic number is  90.0000000000;

Speed of light =  1.370359991390D+02 atomic units.
"""


def test_grasp_csum_reports_rci_corrections():
    c = parse_grasp_sum(GRASP_CSUM)["rci_corrections"]
    assert c["is_rci"] and c["transverse_breit"] and c["vacuum_polarisation"]
    assert c["self_energy"] and not c["normal_mass_shift"]
    assert c["photon_frequency_factor"] == 1e-06


def test_grasp_dhf_sum_has_no_rci_corrections():
    assert "rci_corrections" not in parse_grasp_sum(GRASP_SUM_DHF)


# rhfs_lsj .chlsj — the GRASP manual's Li 1s(2).2p_2P example (real Li-7 moments,
# published A/B/g_J values; pins the parser against ground truth).
GRASP_CHLSJ = """Nuclear spin 1.500000000000000D+00 au
Nuclear magnetic dipole moment 3.256426800000000D+00 n.m.
Nuclear electric quadrupole moment -4.000000000000000D-02 barns
Energy State J P A(MHz) B(MHz) gJ
-7.4042610 1s(2).2p_2P 1/2 - 4.482D+01 -0.000D+00 6.666573D-01
-7.4042597 1s(2).2p_2P 3/2 - -3.538D+00 -1.773D-01 1.333325D+00
"""


def test_grasp_hfs_lsj_parses_published_li_values():
    r = parse_hfs(GRASP_CHLSJ)
    assert r["nuclear_spin"] == 1.5
    assert r["n_levels"] == 2
    p12, p32 = r["levels"]
    assert p12["j_str"] == "1/2" and abs(p12["a_mhz"] - 44.82) < 1e-2
    assert abs(p32["a_mhz"] - (-3.538)) < 1e-3 and abs(p32["b_mhz"] - (-0.1773)) < 1e-4


def test_grasp_hfs_raw_h_table():
    # rhfs .h row: Level J Parity A B g_J delta_g_J total_g_J
    raw = ("Nuclear spin                         5.000000000000000D-01 au\n"
           " Interaction constants:\n"
           "   1        1 +      3.7774375523D+03   -2.1917712011D+02    "
           "4.2281763569D-01   -1.3363807600D-03    4.2148125493D-01\n")
    r = parse_hfs(raw)
    assert r["nuclear_spin"] == 0.5 and r["n_levels"] == 1
    assert abs(r["levels"][0]["a_mhz"] - 3777.4375523) < 1e-4
    assert "total_g_j" in r["levels"][0]


# ris4 .i — Th 6d^2 ground level (real container output).
GRASP_RIS_I = """ Level  J Parity  Energy
   1        0 +        -0.2651014327D+05  (a.u.)

 Level  J Parity  Normal mass shift parameter

                             <K^1>             <K^2+K^3>         <K^1+K^2+K^3>
   1        0 +         0.4711339546D+05   -0.2316071896D+05    0.2395267650D+05  (a.u.)
                        0.1700549720D+09   -0.8359820762D+08    0.8645676441D+08  (GHz u)

 Level  J Parity  Specific mass shift parameter

                             <K^1>             <K^2+K^3>         <K^1+K^2+K^3>
   1        0 +        -0.1126069117D+05    0.3466679116D+04   -0.7794012053D+04  (a.u.)
                       -0.4064526666D+08    0.1251291728D+08   -0.2813234939D+08  (GHz u)

 Electron density in atomic units

 Level  J Parity        DENS (a.u.)

   1        0 +         0.5194244460D+07
"""


def test_grasp_ris_parses_mass_shift_and_density():
    r = parse_ris(GRASP_RIS_I)
    assert r["n_levels"] == 1
    assert abs(r["normal_mass_shift"][0]["k1"] - 47113.39546) < 1e-3
    assert abs(r["specific_mass_shift"][0]["k1_k2_k3"] - (-7794.012053)) < 1e-3
    assert abs(r["electron_density"][0]["density_au"] - 5194244.46) < 1e-1


# rtransition .t.lsj — the Li 2s->2p resonance line (real container output).
GRASP_T_LSJ = """ Transition between files:
 Li2s
 Li2p


   1   -7.43353309  1s(2).2s_2S
   1   -7.36586156  1s(2).2p_2P
   14852.18 CM-1      6733.02 ANGS(VAC)      6732.32 ANGS(AIR)
 E1  S =  1.13141D+01   GF =  5.10428D-01   AKI =  3.75515D+07   dT =  0.03441
          1.17173D+01         5.28618D-01          3.88897D+07
"""


def test_grasp_transition_parses_e1_line():
    r = parse_transition(GRASP_T_LSJ)
    assert r["n_transitions"] == 1
    t = r["transitions"][0]
    assert t["type"] == "E1" and t["lower"]["label"] == "1s(2).2s_2S"
    assert abs(t["wavelength_vac_ang"] - 6733.02) < 1e-2
    assert abs(t["length_gauge"]["gf"] - 0.510428) < 1e-5
    assert abs(t["length_gauge"]["a_ki_per_s"] - 3.75515e7) < 1e2
    assert abs(t["velocity_gauge"]["gf"] - 0.528618) < 1e-5
