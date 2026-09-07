"""Regression tests for model-ladder and fixed-orbital RCI inspection."""

from pathlib import Path

from chemtools.programs.grasp.strategy.case_inspection import (
    inspect_grasp_case_directory,
)


def _summary(
    energy: float,
    n_csfs: int,
    n_subshells: int,
    hamiltonian: str | None = None,
) -> str:
    corrections = ""
    if hamiltonian == "dc":
        corrections = "H (Dirac Coulomb) will be diagonalised by itself."
    elif hamiltonian == "breit0":
        corrections = """
To H (Dirac Coulomb) is added H (Transverse)
factor multiplying the photon frequency: 0.000000000000D+00
"""
    return f"""
There are 4 electrons in the cloud
 in {n_csfs} relativistic CSFs
 based on {n_subshells} relativistic subshells.
The atomic number is 4.0000000000;
 the nucleus is stationary;
 Fermi nucleus:
 c = 3.906285761065D-05 Bohr radii,
 a = 9.890591372451D-06 Bohr radii;
 there are 89 tabulation points in the nucleus.
Speed of light = 1.370359991390D+02 atomic units.
{corrections}
Radial grid: R(I) = RNT*(exp((I-1)*H)-1), I = 1, ..., N;
 RNT = 5.000000000000D-07 Bohr radii;
 H = 5.000000000000D-02 Bohr radii;
 N = 590;
 R(N) = 3.082779741723D+06 Bohr radii.
OL calculation. Level 1 will be optimised
Subshell radial wavefunction summary:
  1s 4.7000000000D+00 1.000D+00 1.00 1.000D-07 1.000D-12 1.000D-10 350
  2s 3.0000000000D-01 1.000D+00 1.00 1.000D-07 1.000D-12 1.000D-10 350
Eigenenergies
 Level J Parity Hartrees Kaysers eV
 1 0 + {energy:.12E} -1.000000000D+06 -3.000000000D+02
Weights of major contributors to ASF:
"""


def _write_mcdhf_stage(
    root: Path,
    name: str,
    energy: float,
    n_csfs: int,
    peel: str,
    active_orbitals: str,
    rank: int,
) -> Path:
    stage = root / name
    stage.mkdir(parents=True)
    n_subshells = 1 + len(peel.split())
    csf_text = f"Core subshells:\n  1s\nPeel subshells:\n  {peel}\nCSF(s):\n"
    (stage / f"{name}.sum").write_text(
        _summary(energy, n_csfs, n_subshells),
        encoding="utf-8",
    )
    (stage / f"{name}.c").write_text(csf_text, encoding="utf-8")
    (stage / f"{name}.w").write_bytes(b"radial wavefunction")
    (stage / "isodata").write_text("fixed nucleus\n", encoding="utf-8")
    (stage / "rcsfgenerate.in").write_text(
        f"*\n1\n2s(2,*)\n\n{active_orbitals}\n0,0\n{rank}\nn\n",
        encoding="utf-8",
    )
    (stage / "rmcdhf.in").write_text("y\n1\n*\n*\n100\n", encoding="utf-8")
    (stage / "rmcdhf_mem.stdout").write_text(
        f"Iteration number 1\nLevel 1 Energy = {energy:.12E}\nRMCDHF: Execution complete\n",
        encoding="utf-8",
    )
    return stage


def _write_rci_pair(
    root: Path,
    stage: Path,
    dc_energy: float,
    breit_energy: float,
) -> None:
    name = stage.name
    n_csfs = 1 if name == "dhf" else 3
    n_subshells = 2 if name == "dhf" else 4
    for variant, energy, hamiltonian in (
        ("dc", dc_energy, "dc"),
        ("breit0", breit_energy, "breit0"),
    ):
        target = root / name / variant
        target.mkdir(parents=True)
        (target / "state.csum").write_text(
            _summary(energy, n_csfs, n_subshells, hamiltonian),
            encoding="utf-8",
        )
        (target / "state.c").write_bytes((stage / f"{name}.c").read_bytes())
        (target / "state.w").write_bytes((stage / f"{name}.w").read_bytes())
        (target / "isodata").write_bytes((stage / "isodata").read_bytes())
        (target / "rci.in").write_text("input\n", encoding="utf-8")
        (target / "rci.stdout").write_text(
            "RCI\nBlock 1, ncf = 1\nRCI: Execution complete.\n",
            encoding="utf-8",
        )


def _case(tmp_path: Path) -> tuple[Path, Path]:
    mcdhf_root = tmp_path / "mcdhf"
    rci_root = tmp_path / "rci"
    dhf = _write_mcdhf_stage(mcdhf_root, "dhf", -14.0, 1, "2s", "2s", 0)
    cas = _write_mcdhf_stage(
        mcdhf_root,
        "cas",
        -14.1,
        3,
        "2s 2p- 2p",
        "2s,2p",
        0,
    )
    _write_rci_pair(rci_root, dhf, -14.0, -13.9993)
    _write_rci_pair(rci_root, cas, -14.1, -14.0993)
    return mcdhf_root, rci_root


def test_inspector_reconstructs_ladder_and_validates_rci_pair(tmp_path: Path):
    mcdhf_root, rci_root = _case(tmp_path)

    report = inspect_grasp_case_directory(
        str(mcdhf_root),
        str(rci_root),
        ["dhf", "cas"],
    )

    assert report["assessment"]["verdict"] == "complete"
    assert report["comparison_table"] == [
        {
            "stage": "dhf",
            "n_csfs": 1,
            "n_subshells": 2,
            "core_subshells": ["1s"],
            "active_electrons": 2,
            "active_spinors": 2,
            "raw_determinant_dimension": 1,
            "substitution_model": "reference_only",
            "mcdhf_energy_hartree": -14.0,
            "mcdhf_iterations": 1,
            "mcdhf_converged": True,
            "mcdhf_convergence_evidence": (
                "execution_complete_without_reported_stopping_test"
            ),
            "mcdhf_max_final_orbital_self_consistency": None,
            "mcdhf_unstable_orbitals": [],
            "rci_dirac_coulomb_energy_hartree": -14.0,
            "rci_breit0_energy_hartree": -13.9993,
            "breit0_correction_microhartree": 700.000000000145,
            "rci_dc_minus_mcdhf_microhartree": 0.0,
        },
        {
            "stage": "cas",
            "n_csfs": 3,
            "n_subshells": 4,
            "core_subshells": ["1s"],
            "active_electrons": 2,
            "active_spinors": 8,
            "raw_determinant_dimension": 28,
            "substitution_model": "reference_only",
            "mcdhf_energy_hartree": -14.1,
            "mcdhf_iterations": 1,
            "mcdhf_converged": True,
            "mcdhf_convergence_evidence": (
                "execution_complete_without_reported_stopping_test"
            ),
            "mcdhf_max_final_orbital_self_consistency": None,
            "mcdhf_unstable_orbitals": [],
            "rci_dirac_coulomb_energy_hartree": -14.1,
            "rci_breit0_energy_hartree": -14.0993,
            "breit0_correction_microhartree": 700.000000000145,
            "rci_dc_minus_mcdhf_microhartree": 0.0,
        },
    ]
    assert report["increments"]["mcdhf"][0]["delta_microhartree"] == -99999.99999999965
    assert report["issues"] == []


def test_inspector_rejects_rci_pair_with_changed_orbitals(tmp_path: Path):
    mcdhf_root, rci_root = _case(tmp_path)
    (rci_root / "cas" / "breit0" / "state.w").write_bytes(b"different orbitals")

    report = inspect_grasp_case_directory(
        str(mcdhf_root),
        str(rci_root),
        ["dhf", "cas"],
    )

    assert report["assessment"]["verdict"] == "invalid"
    assert report["assessment"]["ready_for_numerical_comparison"] is False
    assert [issue["code"] for issue in report["issues"]] == ["rci_pair_model_mismatch"]


def test_inspector_reads_rmcdhf_cycle_limit_from_stderr(tmp_path: Path):
    mcdhf_root, _ = _case(tmp_path)
    stage = mcdhf_root / "dhf"
    (stage / "rmcdhf_mem.stderr").write_text(
        "Maximum iterations in SCF Exceeded.\n",
        encoding="utf-8",
    )

    report = inspect_grasp_case_directory(
        str(mcdhf_root),
        stages=["dhf"],
    )

    assert report["assessment"]["verdict"] == "invalid"
    assert report["stages"][0]["execution"]["converged"] is False
    assert report["stages"][0]["execution"][
        "explicitly_not_converged"
    ] is True
    assert report["issues"][0]["code"] == "rmcdhf_not_converged"
    assert "rmcdhf_stderr" in report["stages"][0]["artifacts"]


def test_inspector_flags_growing_orbital_alternation(tmp_path: Path):
    mcdhf_root, _ = _case(tmp_path)
    stage = mcdhf_root / "dhf"
    rows = (
        (1, 1.00e-2, -3.92e-2),
        (2, 1.20e-2, 2.01e-2),
        (3, 2.00e-2, -4.03e-2),
        (4, 3.40e-2, 3.91e-2),
    )
    trace = ""
    for iteration, residual, norm in rows:
        trace += (
            f"Iteration number {iteration}\n"
            "  6p    3.9806664D-01  3  3.012D+02 "
            f"{residual:.2E} {norm:.2E} 0.800   371   421  0  4\n"
            "Level 1 Energy = -1.400000000000E+01\n"
        )
    trace += "RMCDHF: Execution complete.\n"
    (stage / "rmcdhf_mem.stdout").write_text(trace, encoding="utf-8")

    report = inspect_grasp_case_directory(
        str(mcdhf_root),
        stages=["dhf"],
    )

    assert report["assessment"]["verdict"] == "partial"
    assert report["assessment"]["ready_for_numerical_comparison"] is False
    assert report["comparison_table"][0]["mcdhf_unstable_orbitals"] == [
        "6p"
    ]
    assert report["issues"][0]["code"] == (
        "rmcdhf_unstable_orbital_trace"
    )
