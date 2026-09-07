"""Unit tests for parser / verdict logic added during the 2026 dogfood sweep.

Committed fixtures require no external corpus, so a clean clone can run them.
These pin behavior most likely to regress silently: relativistic correlation
energies, the open-shell COSCI resolution table, active-space verdict
thresholds, multiplicity scans, and GRASP convergence classification.
"""
from pathlib import Path

import pytest

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
from chemtools.programs.grasp.parse.rmcdhf_log import parse_rmcdhf_log
from chemtools.programs.grasp.parse.rci_log import parse_rci_log
from chemtools.programs.grasp.parse.lsjlbl import parse_lsjlbl
from chemtools.programs.grasp import GRASP
from chemtools.programs.grasp.strategy.diagnose import suggest_grasp_recovery


KNOWLEDGE_FIXTURES = (
    Path(__file__).parent / "fixtures" / "knowledge" / "silent_success"
)


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


GRASP_MULTI_BLOCK_CSUM = """
Eigenenergies:
Level  J Parity       Hartrees              Kaysers                eV
  1   5/2 -   -8.85246198044431D+03 -1.94289082987637D+09 -2.40887760159539D+05
Weights of major contributors to ASF:
Level J Parity      CSF contributions
  1   5/2 -      1.00000
Self Energy Corrections:
Eigenenergies:
Level  J Parity       Hartrees              Kaysers                eV
  1   5/2 -   -8.84677361312425D+03 -1.94164237755570D+09 -2.40732971800581D+05
Eigenenergies:
Level  J Parity       Hartrees              Kaysers                eV
  1   7/2 -   -8.85245246910603D+03 -1.94288874237891D+09 -2.40887501342841D+05
Weights of major contributors to ASF:
Level J Parity      CSF contributions
  1   7/2 -      1.00000
"""


def test_grasp_csum_parser_keeps_every_asf_block_and_excludes_qed_tables():
    parsed = parse_grasp_sum(GRASP_MULTI_BLOCK_CSUM)

    assert [
        (level["j_str"], level["parity"], level["energy_hartree"])
        for level in parsed["eigenenergies"]
    ] == [
        ("5/2", "-", -8852.46198044431),
        ("7/2", "-", -8852.45246910603),
    ]


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


def test_grasp_summary_reports_comparison_metadata():
    parsed = parse_grasp_sum(
        """ There are 4 electrons in the cloud
 in 26 relativistic CSFs
 based on 16 relativistic subshells.
The atomic number is 4.0000000000;
 the nucleus is stationary;
 Fermi nucleus:
 c = 3.906285761065D-05 Bohr radii,
 a = 9.890591372451D-06 Bohr radii;
 there are 89 tabulation points in the nucleus.
Speed of light = 1.370359991390D+02 atomic units.
Radial grid: R(I) = RNT*(exp((I-1)*H)-1), I = 1, ..., N;
 RNT = 5.000000000000D-07 Bohr radii;
 H = 5.000000000000D-02 Bohr radii;
 N = 590;
 R(N) = 3.082779741723D+06 Bohr radii.
 OL calculation.
 Level 1 will be optimised.
"""
    )

    assert parsed["n_electrons"] == 4
    assert parsed["n_csfs"] == 26
    assert parsed["n_subshells"] == 16
    assert parsed["nucleus"] == {
        "stationary": True,
        "mass_electron_units": 0.0,
        "model": "fermi",
        "fermi_c_bohr": 3.906285761065e-05,
        "fermi_a_bohr": 9.890591372451e-06,
        "tabulation_points": 89,
    }
    assert parsed["radial_grid"] == {
        "RNT": 5.0e-7,
        "H": 0.05,
        "N": 590,
        "rmax_bohr": 3.082779741723e6,
    }
    assert parsed["optimization_mode"] == "OL"
    assert parsed["ol_level_optimized"] == 1


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


RMCDHF_FAILED = """ Iteration number   1
 Average energy = -2.6510D+04 Hartrees
 Method 1 unable to solve for  7s  orbital
 Failure; equation for orbital  7s  could not be solved using method 1
 Method 2 unable to solve for  7s  orbital
 Failure; equation for orbital  7s  could not be solved using method 2
ERROR STOP
Error termination. Backtrace:
"""

RMCDHF_OK = """ Iteration number   1
 Average energy = -2.6510D+04 Hartrees
 Iteration number   2
 Average energy = -2.6510D+04 Hartrees
 RMCDHF: Execution complete.
"""

RMCDHF_BE_LEVELS = """ Iteration number  16
 Average energy =  -1.2603378003D+01 Hartrees
 Level  1    Energy = -1.462198600448D+01    Weight =  1.00000D+00
 Iteration number  17
 Average energy =  -1.2602945004D+01 Hartrees
 Level  1    Energy = -1.462198600430D+01    Weight =  1.00000D+00
 RMCDHF: Execution complete.
"""

RMCDHF_GROWING_ALTERNATION = """ There are 616 relativistic CSFs
 There are/is 24 relativistic subshells
 Iteration number   1
  6p    4.0363833D-01  3  3.088D+02  1.00D-02 -3.92D-02 0.800   371   421  0  4
 Level  1    Energy = -1.406770368000D+04    Weight =  1.00000D+00
 Iteration number   2
  6p    3.9840810D-01  3  3.075D+02  1.20D-02  2.01D-02 0.800   371   421  0  4
 Level  1    Energy = -1.406770368400D+04    Weight =  1.00000D+00
 Iteration number   3
  6p    3.9741981D-01  3  3.107D+02  2.00D-02 -4.03D-02 0.800   371   421  0  3
 Level  1    Energy = -1.406770368700D+04    Weight =  1.00000D+00
 Iteration number   4
  6p    3.9806664D-01  3  3.012D+02  3.40D-02  3.91D-02 0.800   371   421  0  3
 Level  1    Energy = -1.406770368751D+04    Weight =  1.00000D+00
 RMCDHF: Execution complete.
"""

RCI_BE_LOG = """ RCI
 Block            1 ,  ncf =           26
 There are/is           16  relativistic subshells;
 There are           26  relativistic CSFs... load complete;
 Computing       53106  Breit integrals of type 1
 Computing       26494  Breit integrals of type 2
 INTERP: Accuracy of interpolation (3.8D-03) is below input criterion.
 RCI: Execution complete.
"""


def test_rmcdhf_log_flags_orbital_solver_crash():
    r = parse_rmcdhf_log(RMCDHF_FAILED)
    assert r["converged"] is False
    assert r["orbital_solver_failed"] and r["failed_orbitals"] == ["7s"]
    assert r["error_stop"] is True


def test_rmcdhf_log_clean_run_not_flagged():
    r = parse_rmcdhf_log(RMCDHF_OK)
    assert r["converged"] is True
    assert not r["orbital_solver_failed"] and not r["error_stop"]


def test_rmcdhf_log_cycle_limit_overrides_execution_complete():
    parsed = parse_rmcdhf_log(
        RMCDHF_OK + " Maximum iterations in SCF Exceeded.\n"
    )

    assert parsed["converged"] is False
    assert parsed["explicitly_not_converged"] is True


def test_rmcdhf_log_flags_mpi_signal_termination():
    parsed = parse_rmcdhf_log(
        RMCDHF_OK
        + "BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES\n"
    )

    assert parsed["converged"] is False
    assert parsed["error_stop"] is True


def test_rmcdhf_log_uses_asf_level_energy_not_average_energy_record():
    parsed = parse_rmcdhf_log(RMCDHF_BE_LEVELS)

    assert parsed["final_energy_hartree"] == -14.62198600430
    assert parsed["final_energy_source"] == "rmcdhf_level_record"
    assert parsed["energy_change"] == pytest.approx(1.8e-10)
    assert parsed["iterations"][-1]["average_energy_records_hartree"] == [-12.602945004]


def test_rmcdhf_log_preserves_orbital_trace_after_energy_stopped_run():
    parsed = parse_rmcdhf_log(RMCDHF_GROWING_ALTERNATION)

    assert parsed["converged"] is True
    assert parsed["convergence_evidence"] == (
        "execution_complete_without_reported_stopping_test"
    )
    assert parsed["max_final_orbital_self_consistency"] == 0.034
    assert parsed["final_orbitals"] == [{
        "label": "6p",
        "energy_au": 0.39806664,
        "method": 3,
        "p0": 301.2,
        "self_consistency": 0.034,
        "norm_minus_one": 0.0391,
        "damping_factor": 0.8,
        "join_point": 371,
        "max_tabulation_point": 421,
        "inversion_count": 0,
        "node_count": 3,
    }]
    assert parsed["alternating_norm_orbitals"] == ["6p"]
    assert parsed["growing_alternating_orbitals"] == ["6p"]
    assert parsed["node_count_changes"] == [{
        "label": "6p",
        "node_counts": [4, 3],
    }]


def test_grasp_parser_marks_growing_orbital_alternation_for_review(tmp_path):
    path = tmp_path / "rmcdhf_mem.stdout"
    path.write_text(RMCDHF_GROWING_ALTERNATION, encoding="utf-8")

    parsed = GRASP.parser.parse_output(str(path))

    assert parsed["derived"][
        "grasp:max_final_orbital_self_consistency"
    ] == 0.034
    assert parsed["derived"]["grasp:growing_alternating_orbitals"] == ["6p"]
    assert parsed["diagnosis"]["verdict"]["label"] == (
        "completed_with_unstable_orbital_trace"
    )
    assert parsed["diagnosis"]["next_actions"][0]["action"] == (
        "stage_grasp_orbital_recovery"
    )


def test_grasp_recovery_stages_growing_orbital_alternation():
    recovery = suggest_grasp_recovery(
        error_text=RMCDHF_GROWING_ALTERNATION
    )

    assert recovery["failure_class"] == "rmcdhf_orbital_oscillation"
    assert recovery["orbitals"] == ["6p"]
    assert recovery["next_actions"][0] == (
        "Do not propagate rwfn.out from this stage."
    )


def test_rci_log_reports_execution_metadata_but_no_energy():
    parsed = parse_rci_log(RCI_BE_LOG)

    assert parsed["completed"] is True
    assert parsed["n_csfs"] == 26
    assert parsed["n_subshells"] == 16
    assert parsed["breit_integrals"] == {"1": 53106, "2": 26494}
    assert parsed["qed_interpolation_warning_count"] == 1
    assert parsed["final_energy_hartree"] is None
    assert parsed["energy_source_required"] == "rci_summary_csum"


def test_zero_exit_rmcdhf_requires_positive_convergence_evidence():
    cases = (
        (
            "grasp_rmcdhf_converged.log",
            0,
            True,
            False,
            "success",
            None,
        ),
        (
            "grasp_rmcdhf_cycle_limit.log",
            0,
            False,
            True,
            "failed",
            "max_iter_exhausted",
        ),
    )

    for (
        filename,
        process_exit_code,
        converged,
        cycle_limit_reached,
        task_outcome,
        recovery_class,
    ) in cases:
        path = KNOWLEDGE_FIXTURES / filename
        parsed = parse_rmcdhf_log(str(path))
        tasks = GRASP.parser.task_index(str(path))

        assert process_exit_code == 0
        assert parsed["converged"] is converged
        assert parsed["explicitly_not_converged"] is cycle_limit_reached
        assert [task["outcome"] for task in tasks] == [task_outcome]
        assert tasks[0]["line_range"] == (
            1,
            len(path.read_text(encoding="utf-8").splitlines()),
        )
        if recovery_class is not None:
            recovery = suggest_grasp_recovery(
                error_text=path.read_text(encoding="utf-8")
            )
            assert recovery["failure_class"] == recovery_class


def test_grasp_recovery_classifies_orbital_solver_failure():
    rec = suggest_grasp_recovery(error_text=RMCDHF_FAILED)
    assert rec["failure_class"] == "rmcdhf_orbital_not_solved"
    assert "bootstrap" in rec["fix_recipe"].lower()


# lsj.lbl with a correlation expansion: the leading CSF is small, but the level
# is a near-pure LS term once components are summed by their total term (_3H).
GRASP_LSJ_CORR = """ Pos   J   Parity      Energy Total      Comp. of ASF
  1    6     +        -13541.502299495      99.924%
        -0.39583658    0.15668659   4f(11)4I1.5f_3H
         0.31257928    0.09770581   4f(12)_3H1
         0.25638451    0.06573302   4f(11)4G1.5f_3H
        -0.11028494    0.01216277   4f(10)3L1.5f(2)1I1_3H
         0.10000000    0.02000000   4f(11)2H2.5f_1I
"""


def test_grasp_lsjlbl_term_composition_aggregates_by_total_term():
    lv = parse_lsjlbl(GRASP_LSJ_CORR)["levels"][0]
    # leading CSF is only ~16%, but summed-by-term shows a near-pure 3H level
    assert lv["dominant_weight"] < 0.20
    assert lv["dominant_term"] == "3H"
    assert lv["term_composition"]["3H"] > 0.30
    assert lv["term_composition"]["1I"] == 0.02


def test_grasp_transition_parses_e1_line():
    r = parse_transition(GRASP_T_LSJ)
    assert r["n_transitions"] == 1
    t = r["transitions"][0]
    assert t["type"] == "E1" and t["lower"]["label"] == "1s(2).2s_2S"
    assert abs(t["wavelength_vac_ang"] - 6733.02) < 1e-2
    assert abs(t["length_gauge"]["gf"] - 0.510428) < 1e-5
    assert abs(t["length_gauge"]["a_ki_per_s"] - 3.75515e7) < 1e2
    assert abs(t["velocity_gauge"]["gf"] - 0.528618) < 1e-5


def test_grasp_backend_routes_specialized_property_outputs(tmp_path):
    cases = (
        (
            "li.hlsj",
            GRASP_CHLSJ,
            "hfs",
            "grasp:n_hfs_levels",
            2,
        ),
        (
            "th.i",
            GRASP_RIS_I,
            "isotope_shift",
            "grasp:n_isotope_shift_levels",
            1,
        ),
        (
            "li.t.lsj",
            GRASP_T_LSJ,
            "transition",
            "grasp:n_transitions",
            1,
        ),
    )

    for filename, contents, file_kind, evidence_key, expected_count in cases:
        path = tmp_path / filename
        path.write_text(contents, encoding="utf-8")

        assert GRASP.detector.detect(contents)
        parsed = GRASP.parser.parse_output(str(path))
        assert parsed["derived"]["grasp:file_kind"] == file_kind
        assert parsed["derived"][evidence_key] == expected_count
        assert parsed["tasks"][0]["kind"] == "property"
        assert parsed["tasks"][0]["outcome"] == "success"


def test_grasp_backend_distinguishes_rci_from_rmcdhf_summary(tmp_path):
    path = tmp_path / "th.csum"
    path.write_text(GRASP_CSUM, encoding="utf-8")

    parsed = GRASP.parser.parse_output(str(path))

    assert parsed["derived"]["grasp:file_kind"] == "rci_summary"
    assert parsed["derived"]["grasp:rci_corrections"]["transverse_breit"] is True
    assert parsed["tasks"][0]["name"] == "rci summary"
    assert parsed["tasks"][0]["method"] == "RCI"


def test_molcas_recovery_refuses_unrecognized_input(tmp_path):
    from chemtools.programs.molcas.strategy.recovery import apply_recovery

    input_path = tmp_path / "job.nw"
    target_path = tmp_path / "job_recovered.nw"
    input_path.write_text(
        "geometry\nH 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )

    recovered = apply_recovery(
        str(input_path),
        recovery={
            "failure_class": "memory_exceeded",
            "current_memory_mb": 4000,
        },
        write_to=str(target_path),
    )

    assert recovered["error"] == "input_format_mismatch"
    assert recovered["changes_applied"] == []
    assert not target_path.exists()
