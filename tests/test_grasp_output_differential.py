"""Calibration cases for the persistent GRASP differential checker."""

from scripts.check_grasp_output_consistency import compare_case


def _summary(energy: float) -> str:
    energy_text = f"{energy:.14E}".replace("E", "D")
    return f""" There are 4 electrons in the cloud
 in 26 relativistic CSFs
 based on 16 relativistic subshells.
The atomic number is 4.0000000000;
 the nucleus is stationary;
 Fermi nucleus:
 c = 3.906285761065D-05 Bohr radii,
 a = 9.890591372451D-06 Bohr radii;
 there are 89 tabulation points in the nucleus.
Eigenenergies:
Level J Parity Hartrees Kaysers eV
 1 0 + {energy_text} -3.209154988193D+06 -3.978845055127D+02
Weights of major contributors to ASF:
Level J Parity CSF contributions
 1 0 + 1.00000
"""


def _rmcdhf_log(energy: float) -> str:
    energy_text = f"{energy:.14E}".replace("E", "D")
    return f""" RMCDHF
 Iteration number 1
 Average energy = -1.2602945004D+01 Hartrees
 Level 1 Energy = {energy_text} Weight = 1.00000D+00
 RMCDHF: Execution complete.
"""


def test_differential_checker_agrees_on_rmcdhf_level_energy(tmp_path):
    log = tmp_path / "rmcdhf.stdout"
    summary = tmp_path / "state.sum"
    log.write_text(_rmcdhf_log(-14.62198600430), encoding="utf-8")
    summary.write_text(_summary(-14.62198600430), encoding="utf-8")

    checked = compare_case(
        "agree",
        log,
        summary,
        energy_tolerance=1.0e-10,
    )

    assert checked["outcome"] == "agree"
    assert checked["energy_difference_hartree"] == 0.0


def test_differential_checker_detects_known_wrong_energy(tmp_path):
    log = tmp_path / "rmcdhf.stdout"
    summary = tmp_path / "state.sum"
    log.write_text(_rmcdhf_log(-12.602945004), encoding="utf-8")
    summary.write_text(_summary(-14.62198600430), encoding="utf-8")

    checked = compare_case(
        "known-defect",
        log,
        summary,
        energy_tolerance=1.0e-10,
    )

    assert checked["outcome"] == "disagree"
    assert checked["energy_difference_hartree"] > 2.0


def test_differential_checker_records_rci_stdout_refusal(tmp_path):
    log = tmp_path / "rci.stdout"
    summary = tmp_path / "state.csum"
    log.write_text(
        " RCI\n Block 1, ncf = 26\n RCI: Execution complete.\n",
        encoding="utf-8",
    )
    summary.write_text(_summary(-14.6212846667336), encoding="utf-8")

    checked = compare_case(
        "rci-log",
        log,
        summary,
        energy_tolerance=1.0e-10,
    )

    assert checked["outcome"] == "tool-refused"
    assert checked["parser_energy_hartree"] is None
    assert checked["reference_energy_hartree"] == -14.6212846667336
