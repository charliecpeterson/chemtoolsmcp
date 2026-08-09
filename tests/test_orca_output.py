"""Contracts pinned from the initial ORCA 6.1.1 serial corpus."""

from __future__ import annotations

from pathlib import Path

from chemtools.application.run_inspection import inspect_run
from chemtools.core.program import ProgramCapability
from chemtools.programs.orca import ORCA
from chemtools.programs.orca.output import parse_orca_output_text


HEADER = """\
                 *****************
                 * O   R   C   A *
                 *****************
                 Program Version 6.1.1  -  RELEASE   -
"""


H2_OUTPUT = HEADER + """\
Your calculation utilizes the basis: def2-SVP
NAME = h2.inp
|  1> ! HF DEF2-SVP
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  H      0.000000    0.000000    0.000000
  H      0.000000    0.000000    0.740000

 Hartree-Fock type      HFTyp           .... RHF
 Total Charge           Charge          ....    0
 Multiplicity           Mult            ....    1
*           SCF CONVERGED AFTER   7 CYCLES          *
FINAL SINGLE POINT ENERGY        -1.128893619388
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 221 msec
"""


WATER_OUTPUT = HEADER + """\
Your calculation utilizes the basis: def2-mTZVPP
NAME = water.inp
|  1> ! R2SCAN-3C Opt Freq TightSCF
WARNING: SCAN, rSCAN, or r2SCAN functional requested.
                       * Geometry Optimization Run *
*           SCF CONVERGED AFTER  11 CYCLES          *
FINAL SINGLE POINT ENERGY       -76.418926067517
***        THE OPTIMIZATION HAS CONVERGED     ***
***               (AFTER    4 CYCLES)               ***
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  O      0.000000    0.000000   -0.002861
  H      0.758146    0.000000    0.588930
  H     -0.758146    0.000000    0.588930

FINAL SINGLE POINT ENERGY       -76.418938720848
VIBRATIONAL FREQUENCIES
-----------------------
     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:       0.00 cm**-1
     6:    1653.27 cm**-1
     7:    3813.56 cm**-1
     8:    3932.69 cm**-1
NORMAL MODES
THERMOCHEMISTRY AT 298.15K
Temperature         ...   298.15 K
Pressure            ...     1.00 atm
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 3 seconds 399 msec
"""


O2_OUTPUT = HEADER + """\
Your calculation utilizes the basis: def2-SVP
NAME = o2.inp
|  1> ! PBE0 DEF2-SVP TightSCF
Warning: RI is on but no J-basis has been assigned. Assigning Def2/J (nothing to worry about!)
WARNING: your system is open-shell and RHF/RKS was chosen
 Hartree-Fock type      HFTyp           .... UHF
 Total Charge           Charge          ....    0
 Multiplicity           Mult            ....    3
*           SCF CONVERGED AFTER   9 CYCLES          *
Expectation value of <S**2>     :     2.007223
Ideal value S*(S+1) for S=1.0   :     2.000000
Sum of atomic spin populations:    2.0000000
FINAL SINGLE POINT ENERGY      -150.051687658399
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 591 msec
"""


URANYL_OUTPUT = HEADER + """\
Your calculation utilizes the basis: ZORA-def2-TZVP
Your calculation utilizes the basis: SARC-ZORA-TZVP
Your calculation utilizes the auxiliary basis: SARC/J
Number of basis functions                   ...    223
Relativistic Method            ... ZORA(MP)
 Hartree-Fock type      HFTyp           .... RHF
 Total Charge           Charge          ....    2
 Multiplicity           Mult            ....    1
 Number of Electrons    NEL             ....  106
*           SCF CONVERGED AFTER  23 CYCLES          *
FINAL SINGLE POINT ENERGY    -29564.019120058209
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 1 minutes 8 seconds 812 msec
"""


SCF_FAILURE_OUTPUT = HEADER + """\
Your calculation utilizes the basis: def2-TZVP
NAME = cucl4_interrupted.inp
|  1> ! PBE0 DEF2-TZVP TightSCF NoAutoStart
INITIAL GUESS: MODEL POTENTIAL
 Hartree-Fock type      HFTyp           .... UHF
 Total Charge           Charge          ....   -2
 Multiplicity           Mult            ....    2
*        SCF NOT CONVERGED AFTER   2 CYCLES         *
ORCA finished by error termination in LEANSCF
"""


DLPNO_OUTPUT = HEADER + """\
Your calculation utilizes the basis: cc-pVTZ
Your calculation utilizes the auxiliary basis: cc-pVTZ/C
Number of basis functions                   ...     64
 Number of Electrons    NEL             ....   10
NAME = water_dlpno_ccsdt.inp
|  1> ! DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/C TightPNO TightSCF NoAutoStart
INITIAL GUESS: MODEL POTENTIAL
 Hartree-Fock type      HFTyp           .... RHF
 Total Charge           Charge          ....    0
 Multiplicity           Mult            ....    1
*           SCF CONVERGED AFTER  11 CYCLES          *
--- The Coupled-Cluster iterations have converged ---
T1 diagnostic                              ...      0.006583174
Triples Correction (T)                     ...     -0.007429935
Final correlation energy                   ...     -0.275316684
E(CCSD)                                    ...    -76.324684273
E(CCSD(T))                                 ...    -76.332114208
FINAL SINGLE POINT ENERGY       -76.332114208318
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 5 seconds 276 msec
"""


RIJCOSX_OUTPUT = HEADER + """\
Your calculation utilizes the basis: def2-TZVP
Your calculation utilizes the auxiliary basis: def2/J
NAME = water_pbe0_rijcosx.inp
|  1> ! PBE0 def2-TZVP def2/J RIJCOSX TightSCF NoAutoStart
 RI-approximation to the Coulomb term is turned on
   RIJ-COSX (HFX calculated with COS-X)).... on
 Hartree-Fock type      HFTyp           .... RHF
 Total Charge           Charge          ....    0
 Multiplicity           Mult            ....    1
*           SCF CONVERGED AFTER   9 CYCLES          *
FINAL SINGLE POINT ENERGY       -76.377445212252
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 740 msec
"""


QMMM_OUTPUT = HEADER + """\
NAME = water_pentamer_qmmm.inp
|  1> ! QMMM PBEh-3c TightSCF NoAutoStart
Multiscale model                       ... QM/MM
Coupling Scheme                        ... additive
Embedding Scheme                       ... electrostatic
Point charges in QM calc. from MM atoms... 12
Size of QMMM System                    ... 15
Size of MM Subsystem                   ... 12
Size of QM Subsystem                   ... 3
Number of link atoms                   ... 0
FINAL SINGLE POINT ENERGY (MM)        0.019670168498
*           SCF CONVERGED AFTER  11 CYCLES          *
FINAL SINGLE POINT ENERGY       -76.219940276758
FINAL SINGLE POINT ENERGY (QM/MM)      -76.200270108260
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 492 msec
"""


CRYSTAL_QMMM_OUTPUT = HEADER + """\
NAME = nacl_ionic_crystal_qmmm.inp
|  1> ! IONIC-CRYSTAL-QMMM PBE0 def2-SVP def2/J RIJCOSX TightSCF
Multiscale model                       ... Ionic-Crystal-QMMM
Coupling Scheme                        ... additive
Embedding Scheme                       ... electrostatic
Number of ECP layers                   ... 1
Point charges in QM calc. from MM atoms... 2620
Size of QMMM System                    ... 2645
Size of MM Subsystem                   ... 2620
Size of QM Subsystem (excl HF/ECP)     ... 7
Number of ECP layer atoms              ... 18
*           SCF CONVERGED AFTER  24 CYCLES          *
Maximum charge difference  0.630 (Threshold:  0.010)
*           SCF CONVERGED AFTER  17 CYCLES          *
Maximum charge difference  0.007 (Threshold:  0.010)
FINAL SINGLE POINT ENERGY     -2926.983813378450
FINAL SINGLE POINT ENERGY (QM/MM)    -2926.983813378450
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 2 minutes 9 seconds 127 msec
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_orca_backend_declares_only_observed_parser_capabilities():
    assert ORCA.capabilities == frozenset({
        ProgramCapability.OUTPUT_PARSE,
        ProgramCapability.OUTPUT_TASK_INDEX,
        ProgramCapability.OUTPUT_GEOMETRY,
        ProgramCapability.OUTPUT_FREQUENCIES,
        ProgramCapability.INPUT_PARSE,
    })


def test_orca_detector_requires_banner_and_version():
    assert ORCA.detector.detect(H2_OUTPUT) is True
    assert ORCA.detector.detect_version(H2_OUTPUT) == "6.1.1"
    assert ORCA.detector.detect("ORCA 6.1.1 input example") is False


def test_parse_h2_serial_completion(tmp_path):
    path = _write(tmp_path, "h2.out", H2_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["tasks"] == [{
        "index": 0,
        "kind": "energy",
        "name": "ORCA Single Point Energy",
        "method": "HF",
        "basis": "def2-SVP",
        "energy_hartree": -1.128893619388,
        "line_range": (1, 18),
        "outcome": "success",
        "has_usable_data": True,
        "selection_priority": 1,
    }]
    assert parsed["derived"] == {
        "n_tasks": 1,
        "scf_converged": True,
        "orca:normal_termination": True,
        "orca:normal_termination_line": 18,
        "orca:input_file": "h2.inp",
        "orca:simple_keywords": ["HF", "DEF2-SVP"],
        "orca:basis_sets": [{"name": "def2-SVP", "line": 5}],
        "orca:auxiliary_basis_sets": [],
        "orca:number_of_basis_functions": None,
        "orca:number_of_electrons": None,
        "orca:relativistic_method": None,
        "orca:initial_guess": None,
        "orca:scf_cycles": [{"cycles": 7, "line": 16}],
        "orca:scf_failures": [],
        "orca:error_termination": None,
        "orca:warning_count": 0,
        "orca:runtime_seconds": 0.221,
        "orca:wavefunction_type": "RHF",
        "orca:charge": 0,
        "orca:multiplicity": 1,
        "final_energy_hartree": -1.128893619388,
        "primary_energy_hartree": -1.128893619388,
        "orca:final_energy_line": 17,
    }
    assert parsed["diagnosis"]["verdict"]["label"] == "completed"
    assert ORCA.parser.get_geometry(str(path)) == [
        {"element": "H", "x": 0.0, "y": 0.0, "z": 0.0},
        {"element": "H", "x": 0.0, "y": 0.0, "z": 0.74},
    ]


def test_parse_optimization_frequency_and_informational_warning(tmp_path):
    path = _write(tmp_path, "water.out", WATER_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert [task["kind"] for task in parsed["tasks"]] == [
        "optimize",
        "frequency",
    ]
    assert [task["outcome"] for task in parsed["tasks"]] == [
        "success",
        "success",
    ]
    assert parsed["primary_task_index"] == 0
    assert parsed["derived"]["final_energy_hartree"] == -76.418938720848
    assert parsed["derived"]["orca:optimization"] == {
        "started_line": 9,
        "converged_line": 12,
        "cycles": 4,
    }
    assert parsed["derived"]["orca:frequencies_cm1"] == [
        1653.27,
        3813.56,
        3932.69,
    ]
    assert parsed["derived"]["n_imaginary_modes"] == 0
    assert parsed["derived"]["orca:thermochemistry"] == {
        "temperature_kelvin": 298.15,
        "pressure_atm": 1.0,
        "line": 33,
    }
    assert parsed["diagnostics"] == [{
        "kind": "warning",
        "message": "SCAN, rSCAN, or r2SCAN functional requested.",
        "line": 8,
        "file": str(path),
    }]
    assert ORCA.parser.get_frequency(str(path))["frequencies_cm1"] == [
        1653.27,
        3813.56,
        3932.69,
    ]


def test_parse_open_shell_spin_evidence_without_failing_on_warnings(tmp_path):
    path = _write(tmp_path, "o2.out", O2_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["tasks"][0]["outcome"] == "success"
    assert parsed["derived"]["orca:wavefunction_type"] == "UHF"
    assert parsed["derived"]["orca:multiplicity"] == 3
    assert parsed["derived"]["orca:spin"] == {
        "expectation_s2": 2.007223,
        "expectation_line": 14,
        "spin_s": 1.0,
        "ideal_s2": 2.0,
        "mulliken_spin_population_sum": 2.0,
        "mulliken_spin_population_sum_line": 16,
    }
    assert parsed["derived"]["orca:warning_count"] == 2
    assert parsed["diagnosis"]["verdict"]["label"] == "completed"


def test_parse_relativistic_and_element_specific_basis_evidence(tmp_path):
    path = _write(tmp_path, "uranyl.out", URANYL_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["derived"]["orca:basis_sets"] == [
        {"name": "ZORA-def2-TZVP", "line": 5},
        {"name": "SARC-ZORA-TZVP", "line": 6},
    ]
    assert parsed["derived"]["orca:auxiliary_basis_sets"] == [
        {"name": "SARC/J", "line": 7}
    ]
    assert parsed["derived"]["orca:number_of_basis_functions"] == 223
    assert parsed["derived"]["orca:number_of_electrons"] == 106
    assert parsed["derived"]["orca:relativistic_method"] == "ZORA(MP)"


def test_parse_dlpno_ccsd_t_energy_components(tmp_path):
    path = _write(tmp_path, "water_dlpno_ccsdt.out", DLPNO_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["tasks"][0]["method"] == "DLPNO-CCSD(T)"
    assert parsed["tasks"][0]["energy_hartree"] == -76.332114208318
    assert parsed["derived"]["orca:coupled_cluster"] == {
        "converged_line": 16,
        "t1_diagnostic": 0.006583174,
        "t1_diagnostic_line": 17,
        "triples_correction_hartree": -0.007429935,
        "triples_correction_line": 18,
        "correlation_energy_hartree": -0.275316684,
        "correlation_energy_line": 19,
        "ccsd_energy_hartree": -76.324684273,
        "ccsd_energy_line": 20,
        "ccsd_t_energy_hartree": -76.332114208,
        "ccsd_t_energy_line": 21,
    }


def test_parse_effective_rijcosx_output_marker(tmp_path):
    path = _write(tmp_path, "water_pbe0_rijcosx.out", RIJCOSX_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["tasks"][0]["method"] == "PBE0"
    assert parsed["derived"]["orca:auxiliary_basis_sets"] == [
        {"name": "def2/J", "line": 6}
    ]
    assert parsed["derived"]["orca:ri_approximation"] == {
        "name": "RIJCOSX",
        "line": 10,
    }


def test_parse_additive_qmmm_subsystems_and_total_energy(tmp_path):
    path = _write(tmp_path, "water_pentamer_qmmm.out", QMMM_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["tasks"][0]["energy_hartree"] == -76.200270108260
    assert parsed["derived"]["final_energy_hartree"] == -76.200270108260
    assert parsed["derived"]["orca:final_energy_line"] == 18
    assert parsed["derived"]["orca:multiscale"] == {
        "model": "QM/MM",
        "coupling_scheme": "additive",
        "embedding_scheme": "electrostatic",
        "point_charge_count": 12,
        "system_size_atoms": 15,
        "mm_atoms": 12,
        "qm_atoms": 3,
        "link_atoms": 0,
        "ecp_layers": None,
        "ecp_atoms": None,
        "charge_convergence": [],
        "mm_energy_hartree": 0.019670168498,
        "mm_energy_line": 15,
        "qmmm_energy_hartree": -76.200270108260,
        "qmmm_energy_line": 18,
    }


def test_parse_ionic_crystal_qmmm_charge_convergence(tmp_path):
    path = _write(tmp_path, "nacl_crystal_qmmm.out", CRYSTAL_QMMM_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))
    multiscale = parsed["derived"]["orca:multiscale"]

    assert parsed["tasks"][0]["outcome"] == "success"
    assert parsed["tasks"][0]["energy_hartree"] == -2926.983813378450
    assert multiscale["model"] == "Ionic-Crystal-QMMM"
    assert multiscale["point_charge_count"] == 2620
    assert multiscale["system_size_atoms"] == 2645
    assert multiscale["qm_atoms"] == 7
    assert multiscale["ecp_layers"] == 1
    assert multiscale["ecp_atoms"] == 18
    assert multiscale["charge_convergence"] == [
        {"maximum_difference": 0.63, "threshold": 0.01, "line": 17},
        {"maximum_difference": 0.007, "threshold": 0.01, "line": 19},
    ]


def test_missing_normal_termination_is_incomplete(tmp_path):
    path = _write(
        tmp_path,
        "partial.out",
        H2_OUTPUT.replace(
            "                             ****ORCA TERMINATED NORMALLY****\n"
            "TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 221 msec\n",
            "",
        ),
    )

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["tasks"][0]["outcome"] == "incomplete"
    assert parsed["diagnosis"]["verdict"] == {
        "label": "incomplete",
        "confidence": 0.9,
        "reasons": ["The output has no ORCA normal-termination marker."],
    }


def test_explicit_scf_error_termination_is_failed(tmp_path):
    path = _write(tmp_path, "cucl4_interrupted.out", SCF_FAILURE_OUTPUT)

    parsed = ORCA.parser.parse_output(str(path))

    assert parsed["tasks"][0]["outcome"] == "failed"
    assert parsed["derived"]["scf_converged"] is False
    assert parsed["derived"]["orca:initial_guess"] == "MODEL POTENTIAL"
    assert parsed["derived"]["orca:scf_failures"] == [
        {"cycles": 2, "line": 12}
    ]
    assert parsed["derived"]["orca:error_termination"] == {
        "module": "LEANSCF",
        "line": 13,
    }
    assert parsed["diagnosis"]["verdict"] == {
        "label": "failed",
        "confidence": 0.99,
        "reasons": ["ORCA reported error termination in LEANSCF."],
    }


def test_parse_inline_xyz_input(tmp_path):
    path = _write(
        tmp_path,
        "o2.inp",
        "! PBE0 DEF2-SVP TightSCF\n* XYZ 0 3\nO 0 0 0\nO 0 0 1.21\n*\n",
    )

    parsed = ORCA.parser.parse_input(str(path))

    assert parsed["simple_keywords"] == ["PBE0", "DEF2-SVP", "TightSCF"]
    assert parsed["charge"] == 0
    assert parsed["multiplicity"] == 3
    assert parsed["atoms"] == [
        {"element": "O", "x": 0.0, "y": 0.0, "z": 0.0, "line": 3},
        {"element": "O", "x": 0.0, "y": 0.0, "z": 1.21, "line": 4},
    ]


def test_parse_external_pdb_coordinate_reference(tmp_path):
    path = _write(
        tmp_path,
        "nacl.inp",
        "! IONIC-CRYSTAL-QMMM PBE0 def2-SVP\n"
        "%qmmm\n"
        "  QMAtoms {1 6 8 11 12 14 62} end\n"
        "end\n"
        "* pdbfile -5 1 nacl.pdb\n",
    )

    parsed = ORCA.parser.parse_input(str(path))

    assert parsed["block_names"] == ["qmmm"]
    assert parsed["charge"] == -5
    assert parsed["multiplicity"] == 1
    assert parsed["coordinate_format"] == "pdb"
    assert parsed["coordinate_file"] == "nacl.pdb"
    assert parsed["atoms"] == []


def test_parse_element_specific_basis_assignment(tmp_path):
    path = _write(
        tmp_path,
        "uranyl.inp",
        "! PBE0 ZORA ZORA-def2-TZVP SARC/J\n"
        "%basis\n"
        '  NewGTO U "SARC-ZORA-TZVP" end\n'
        "end\n"
        "* XYZ 2 1\n"
        "O 0 0 -1.76\n"
        "U 0 0 0\n"
        "O 0 0 1.76\n"
        "*\n",
    )

    parsed = ORCA.parser.parse_input(str(path))

    assert parsed["block_names"] == ["basis"]
    assert parsed["element_basis_sets"] == [{
        "element": "U",
        "basis": "SARC-ZORA-TZVP",
        "line": 3,
    }]


def test_parse_moread_restart_input(tmp_path):
    path = _write(
        tmp_path,
        "cucl4_restarted.inp",
        "! PBE0 DEF2-TZVP TightSCF MORead NoAutoStart\n"
        '%moinp "cucl4_interrupted.gbw"\n'
        "* XYZ -2 2\n"
        "Cu 0 0 0\n"
        "Cl 2.25 0 0\n"
        "Cl -2.25 0 0\n"
        "Cl 0 2.25 0\n"
        "Cl 0 -2.25 0\n"
        "*\n",
    )

    parsed = ORCA.parser.parse_input(str(path))

    assert parsed["block_names"] == ["moinp"]
    assert parsed["moinp"] == "cucl4_interrupted.gbw"


def test_parse_moread_basis_projection_input(tmp_path):
    path = _write(
        tmp_path,
        "fe_macrocycle_bp86_def2svp.inp",
        "! BP86 DEF2-SVP RI DEF2/J TightSCF MORead NoAutoStart\n"
        '%moinp "fe_macrocycle_bp86_sv.gbw"\n'
        "%scf\n"
        "  GuessMode CMatrix\n"
        "  MaxIter 100\n"
        "end\n"
        "* XYZ 1 6\n"
        "Fe 0 0 0\n"
        "*\n",
    )

    parsed = ORCA.parser.parse_input(str(path))

    assert parsed["block_names"] == ["moinp", "scf"]
    assert parsed["moinp"] == "fe_macrocycle_bp86_sv.gbw"
    assert parsed["guess_mode"] == "CMatrix"


def test_guided_inspection_uses_orca_completion_evidence(tmp_path):
    path = _write(tmp_path, "h2.out", H2_OUTPUT)

    inspected = inspect_run(ORCA, path, resolved_by="content")

    assert inspected["program"] == {
        "name": "orca",
        "version": "6.1.1",
        "resolved_by": "content",
    }
    assert inspected["assessment"] == {
        "source": "parser_diagnosis",
        "verdict": {
            "label": "completed",
            "confidence": 0.99,
            "reasons": [
                "ORCA printed normal termination and every recognized "
                "operation completed."
            ],
        },
    }
    assert inspected["evidence"]["tasks"][0]["energy_hartree"] == (
        -1.128893619388
    )


def test_native_parser_rejects_non_orca_text():
    try:
        parse_orca_output_text("NWChem 7.2.3\n")
    except ValueError as error:
        assert str(error) == (
            "ORCA output does not contain an ORCA release banner."
        )
    else:
        raise AssertionError("non-ORCA text was accepted")
