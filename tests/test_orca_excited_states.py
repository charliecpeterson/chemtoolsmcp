"""Root-resolved parser contracts for ORCA multireference and excited states.

The values are pinned from the ORCA 6.1.1 reference runs in the external corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemtools.programs.orca import ORCA


HEADER = """\
                 *****************
                 * O   R   C   A *
                 *****************
                 Program Version 6.1.1  -  RELEASE   -
"""


def _parse(tmp_path: Path, name: str, body: str):
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return ORCA.parser.parse_output(str(path))


def test_parse_casscf_nevpt2_state_and_total_energies(tmp_path):
    parsed = _parse(tmp_path, "n2_nevpt2.out", """\
|  1> ! def2-SVP NoFrozenCore TightSCF
Number of active electrons          ...    6
Number of active orbitals           ...    6
A PT2 calculation will be performed (PT2 = SC-NEVPT2)
Final CASSCF energy       : -108.707127700 Eh   -2958.0713 eV
CAS-SCF STATES FOR BLOCK  0 MULT= 1 NROOTS= 1
ROOT   0:  E=    -108.7071276998 Eh
NEVPT2 Results
MULT 1, ROOT 0
Total Energy Correction : dE = -0.16407720165333
Reference  Energy       : E0 = -108.70712769981699
Total Energy (E0+dE)    : E  = -108.87120490147032
FINAL SINGLE POINT ENERGY      -108.871204901470
****ORCA TERMINATED NORMALLY****
""")

    assert parsed["tasks"][0]["method"] == "SC-NEVPT2"
    assert parsed["tasks"][0]["outcome"] == "success"
    assert parsed["derived"]["orca:casscf"] == {
        "active_electrons": 6,
        "active_orbitals": 6,
        "state_average_energy_hartree": -108.7071277,
        "state_average_energy_line": 9,
        "roots": [{
            "block": 0,
            "multiplicity": 1,
            "root": 0,
            "energy_hartree": -108.7071276998,
            "excitation_energy_ev": 0.0,
            "wavenumber_cm1": 0.0,
            "line": 11,
        }],
    }
    assert parsed["derived"]["orca:multireference_pt2"] == {
        "method": "SC-NEVPT2",
        "states": [{
            "multiplicity": 1,
            "root": 0,
            "correction_energy_hartree": -0.16407720165333,
            "reference_energy_hartree": -108.70712769981699,
            "total_energy_hartree": -108.87120490147032,
        }],
    }


def test_parse_mrci_transition_energies_and_reference_weights(tmp_path):
    parsed = _parse(tmp_path, "formaldehyde_mrci.out", """\
|  1> ! HF def2-SVP TightSCF
* SCF CONVERGED AFTER 10 CYCLES *
STATE   0:  Energy=   -114.113096218 Eh RefWeight=  0.9124  0.00 eV
STATE   0:  Energy=   -113.964600459 Eh RefWeight=  0.8883  0.00 eV
STATE   0:  Energy=   -113.979014972 Eh RefWeight=  0.9002  0.00 eV
TRANSITION ENERGIES
The lowest energy is   -114.113096218 Eh
State Mult Irrep Root Block    mEh          eV      1/cm
  0    1    A1    0    0      0.000      0.000       0.0
  1    3    A2    0    2    134.081      3.649   29427.4
  2    1    A2    0    1    148.496      4.041   32591.1
FINAL SINGLE POINT ENERGY      -114.113096217709
****ORCA TERMINATED NORMALLY****
""")

    mrci = parsed["derived"]["orca:mrci"]
    assert parsed["tasks"][0]["method"] == "MRCI"
    assert mrci["lowest_energy_hartree"] == -114.113096218
    assert [state["energy_hartree"] for state in mrci["states"]] == [
        -114.113096218,
        -113.979014972,
        -113.964600459,
    ]
    assert [state["reference_weight"] for state in mrci["states"]] == [
        0.9124,
        0.9002,
        0.8883,
    ]


def test_parse_tddft_singlet_and_triplet_roots(tmp_path):
    parsed = _parse(tmp_path, "formaldehyde_tddft.out", """\
|  1> ! PBE0 def2-TZVP RIJCOSX TightSCF
* SCF CONVERGED AFTER 11 CYCLES *
TD-DFT/TDA EXCITED STATES (SINGLETS)
STATE  1:  E=   0.144305 au      3.927 eV    31671.2 cm**-1 <S**2> =   0.000000 Mult 1
TD-DFT/TDA EXCITED STATES (TRIPLETS)
STATE  7:  E=   0.117037 au      3.185 eV    25686.6 cm**-1 <S**2> =   2.000000 Mult 3
FINAL SINGLE POINT ENERGY      -114.273247626762
****ORCA TERMINATED NORMALLY****
""")

    assert parsed["tasks"][0]["method"] == "TD-DFT/PBE0"
    assert parsed["derived"]["orca:tddft"]["states"] == [
        {
            "state": 1,
            "energy_hartree": 0.144305,
            "energy_ev": 3.927,
            "wavenumber_cm1": 31671.2,
            "expectation_s2": 0.0,
            "multiplicity": 1,
            "line": 8,
        },
        {
            "state": 7,
            "energy_hartree": 0.117037,
            "energy_ev": 3.185,
            "wavenumber_cm1": 25686.6,
            "expectation_s2": 2.0,
            "multiplicity": 3,
            "line": 10,
        },
    ]


def test_parse_eom_ccsd_rhs_roots_without_lhs_duplicates(tmp_path):
    parsed = _parse(tmp_path, "formaldehyde_eom_ccsd.out", """\
|  1> ! RHF EOM-CCSD cc-pVDZ TightSCF
* SCF CONVERGED AFTER 13 CYCLES *
--- The Coupled-Cluster iterations have converged ---
E(TOT)                                     ...   -114.208715775
--- The EOM iterations have converged ---
EOM-CCSD RESULTS (RHS)
IROOT=  1:  0.147826 au     4.023 eV   32444.1 cm**-1
Percentage singles character=     92.30
Ground State  LHS
IROOT=  1:  0.147826 au     4.023 eV   32444.1 cm**-1
FINAL SINGLE POINT ENERGY      -114.060889708424
****ORCA TERMINATED NORMALLY****
""")

    eom = parsed["derived"]["orca:eom_ccsd"]
    assert parsed["tasks"][0]["method"] == "EOM-CCSD"
    assert eom["ground_state_energy_hartree"] == -114.208715775
    assert eom["roots"] == [{
        "root": 1,
        "energy_hartree": 0.147826,
        "energy_ev": 4.023,
        "wavenumber_cm1": 32444.1,
        "line": 11,
        "singles_character_percent": 92.3,
    }]


def test_parse_excited_casscf_caspt2_diagnostics(tmp_path):
    parsed = _parse(tmp_path, "n2_caspt2.out", """\
|  1> ! def2-SVP NoFrozenCore TightSCF
Number of active electrons          ...    6
Number of active orbitals           ...    6
A PT2 calculation will be performed (PT2 = CASPT2)
--- Failed to constrain active orbitals due to rotations:
Final CASSCF energy       : -108.707074810 Eh   -2958.0699 eV
CAS-SCF STATES FOR BLOCK  0 MULT= 1 NROOTS= 2
ROOT   0:  E=    -108.9756766188 Eh
ROOT   1:  E=    -108.5872648913 Eh 10.569 eV  85246.5 cm**-1
MULT 1, ROOT 0
smallest energy denominator ITUV =      0.803090021
CASPT2 calculation converged in 9 iterations
MULT 1, ROOT 1
smallest energy denominator ITUV =      0.184577601
CASPT2 calculation converged in 10 iterations
CASPT2 Results
MULT 1, ROOT 0
Total Energy Correction : dE = -0.18349551285289
Reference  Energy       : E0 = -108.97567661884827
Reference Weight        : W0 = 0.95272017103735
Total Energy (E0+dE)    : E  = -109.15917213170115
MULT 1, ROOT 1
Total Energy Correction : dE = -0.20410465661105
Reference  Energy       : E0 = -108.58726489132805
Reference Weight        : W0 = 0.94265924463732
Total Energy (E0+dE)    : E  = -108.79136954793910
FINAL SINGLE POINT ENERGY      -108.975270839820
****ORCA TERMINATED NORMALLY****
""")

    states = parsed["derived"]["orca:multireference_pt2"]["states"]
    assert parsed["tasks"][0]["method"] == "CASPT2"
    assert parsed["tasks"][0]["outcome"] == "success"
    assert [state["convergence_iterations"] for state in states] == [9, 10]
    assert [state["minimum_denominator_hartree"] for state in states] == [
        0.803090021,
        0.184577601,
    ]
    assert [state["reference_weight"] for state in states] == pytest.approx([
        0.95272017103735,
        0.94265924463732,
    ])
    assert parsed["diagnostics"] == [{
        "kind": "warning",
        "message": "Failed to constrain active orbitals due to rotations:",
        "line": 9,
        "file": str(tmp_path / "n2_caspt2.out"),
    }]
