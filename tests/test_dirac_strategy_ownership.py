"""DIRAC scientific summaries belong to strategy modules, not MCP handlers."""

from chemtools.mcp.tools import dirac
from chemtools.programs.dirac.parse.output import filter_spinor_spectrum
from chemtools.programs.dirac.strategy import open_shell, triage


def test_open_shell_occupation_analysis_reports_checkpoint_parity(monkeypatch):
    monkeypatch.setattr(
        open_shell,
        "parse_inp",
        lambda path: {"has_open_shell": True, "has_closed_shell": False},
    )
    monkeypatch.setattr(open_shell, "H5PY_AVAILABLE", True)
    monkeypatch.setattr(
        open_shell,
        "read_orbital_summary",
        lambda path: [
            {
                "shell_class": "open",
                "fermion_symmetry": 2,
                "irrep": "E1u",
                "positive_energy_index": 7,
                "energy_hartree": -0.25,
                "occupation": 0.5,
            },
            {
                "shell_class": "closed",
                "fermion_symmetry": 1,
                "irrep": "E1g",
                "positive_energy_index": 3,
                "energy_hartree": -0.8,
                "occupation": 1.0,
            },
        ],
    )

    analyzed = open_shell.analyze_open_shell_occupations(
        "atom.inp",
        "atom.h5",
    )

    assert analyzed == {
        "verdict": "consistent",
        "input_has_open_shell": True,
        "h5_has_fractional_occupation": True,
        "open_shell_n_orbitals": 1,
        "open_shell_total_occupation_kramers": 0.5,
        "open_shell_by_fermion_symmetry": {
            2: [{
                "irrep": "E1u",
                "positive_energy_index": 7,
                "energy_hartree": -0.25,
                "occupation": 0.5,
            }],
        },
    }


def test_open_shell_occupation_analysis_keeps_optional_dependency_boundary(
    monkeypatch,
):
    monkeypatch.setattr(
        open_shell,
        "parse_inp",
        lambda path: {"has_open_shell": True, "has_closed_shell": False},
    )
    monkeypatch.setattr(open_shell, "H5PY_AVAILABLE", False)

    analyzed = open_shell.analyze_open_shell_occupations(
        "atom.inp",
        "atom.h5",
    )

    assert analyzed["verdict"] == "h5py_missing"
    assert analyzed["input_summary"] == {
        "has_open_shell": True,
        "has_closed_shell": False,
    }


def test_single_run_summary_combines_text_and_checkpoint_evidence(monkeypatch):
    monkeypatch.setattr(
        triage,
        "parse_output",
        lambda path: {
            "program_version": "25.0",
            "tasks_detected": ["SCF", "RELCCSD"],
            "total_energy_hartree": -100.0,
            "scf_converged": True,
            "scf_n_iterations": 12,
            "symmetry": {"point_group": "D2h"},
            "open_shell_setup": {"n_open": 1},
            "homo_lumo_per_symmetry": [{}, {}],
            "excitations": {"available": False},
            "relccsd": {
                "available": True,
                "mp2_total_hartree": -100.1,
                "ccsd_total_hartree": -100.2,
                "ccsd_t_total_hartree": -100.25,
                "mp2_correlation_hartree": -0.1,
                "ccsd_correlation_hartree": -0.2,
            },
            "cosci": {"n_states": 0},
        },
    )
    monkeypatch.setattr(triage, "H5PY_AVAILABLE", True)
    monkeypatch.setattr(
        triage,
        "read_metadata",
        lambda path: {
            "version": "25.0",
            "scf_energy_hartree": -100.0000005,
            "n_fermion_symmetries": 2,
            "n_mo_per_fsym": [10, 10],
            "n_pos_energy_per_fsym": [8, 8],
        },
    )
    monkeypatch.setattr(
        triage,
        "read_orbital_summary",
        lambda path: [
            {"shell_class": "closed"},
            {"shell_class": "open"},
        ],
    )

    summary = triage.summarize_dirac_run("run.out", "run.h5")

    assert summary["verdict"] == "scf_converged"
    assert summary["correlation"]["ccsd_t_total_hartree"] == -100.25
    assert summary["h5_status"] == "loaded"
    assert summary["shell_class_counts"] == {"closed": 1, "open": 1}
    assert summary["text_vs_h5_energy_consistent"] is True


def test_dirac_handlers_only_translate_strategy_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dirac,
        "_analyze_open_shell_occupations",
        lambda input_file, h5_file: calls.append(
            ("open_shell", input_file, h5_file)
        ) or {"kind": "open_shell"},
    )
    monkeypatch.setattr(
        dirac,
        "_summarize_dirac_run",
        lambda output_file, h5_file: calls.append(
            ("summary", output_file, h5_file)
        ) or {"kind": "summary"},
    )

    assert dirac._handle_analyze_dirac_open_shell({
        "input_file": "atom.inp",
        "h5_file": "atom.h5",
    }) == {"kind": "open_shell"}
    assert dirac._handle_summarize_dirac_run({
        "output_file": "run.out",
        "h5_file": "run.h5",
    }) == {"kind": "summary"}
    assert calls == [
        ("open_shell", "atom.inp", "atom.h5"),
        ("summary", "run.out", "run.h5"),
    ]


def test_spinor_filter_owns_occupation_and_energy_selection():
    spectrum = [
        {"energy_hartree": -0.5, "occupation": 1.0},
        {"energy_hartree": -0.2, "occupation": 0.5},
        {"energy_hartree": 0.1, "occupation": 0.0},
        {"energy_hartree": None, "occupation": 1.0},
    ]

    assert filter_spinor_spectrum(
        spectrum,
        occupied_only=True,
        energy_range=[-0.6, -0.1],
    ) == [{"energy_hartree": -0.5, "occupation": 1.0}]
