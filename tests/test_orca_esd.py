"""Parser contracts for ORCA_ESD spectra and radiative-rate evidence.

The values and output spelling come from the ORCA 6.1.1 formaldehyde runs.
"""

from __future__ import annotations

from pathlib import Path

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


def test_parse_fluorescence_spectrum_and_rate(tmp_path):
    parsed = _parse(tmp_path, "formaldehyde_fluorescence.out", """\
|  1> ! PBE0 def2-SVP def2/J RIJCOSX TightSCF ESD(FLUOR)
Your calculation utilizes the basis: def2-SVP
* SCF CONVERGED AFTER 11 CYCLES *
FINAL SINGLE POINT ENERGY      -114.284008969306
Requested calculation:                         ...fluorescence
Lineshape function:                            ...Lorenztian
Excited state PES:                             ...vertical gradient
Use operator derivatives:                      ...yes
Homogeneous linewidth is:                      50.00 cm-1
Temperature used:                              298.15 K
Adiabatic energy difference:                   30844.79 cm-1
0-0 energy difference:                         30844.79 cm-1
The calculated fluorescence rate constant is   1.823670e+05 s-1
with 0.00% from FC and 100.00% from HT
The fluorescence spectrum was saved in         formaldehyde_fluorescence.spectrum
****ORCA ESD FINISHED WITHOUT ERROR****
****ORCA TERMINATED NORMALLY****
""")

    assert parsed["tasks"] == [{
        "index": 0,
        "kind": "property",
        "name": "ORCA_ESD fluorescence",
        "method": "PBE0",
        "basis": "def2-SVP",
        "energy_hartree": -114.284008969306,
        "line_range": (9, 21),
        "outcome": "success",
        "has_usable_data": True,
        "selection_priority": 2,
    }]
    assert parsed["derived"]["orca:esd"] == {
        "process": "fluorescence",
        "started_line": 9,
        "finished_line": 20,
        "line_shape": "Lorenztian",
        "excited_state_pes": "vertical gradient",
        "operator_derivatives": True,
        "homogeneous_linewidth_cm1": 50.0,
        "inhomogeneous_linewidth_cm1": None,
        "temperature_kelvin": 298.15,
        "adiabatic_energy_cm1": 30844.79,
        "zero_zero_energy_cm1": 30844.79,
        "laser_energy_cm1": None,
        "rate_constant_s1": 182367.0,
        "rate_process": "fluorescence",
        "franck_condon_percent": 0.0,
        "herzberg_teller_percent": 100.0,
        "spectrum_file": "formaldehyde_fluorescence.spectrum",
        "spectrum_line": 19,
    }
    assert parsed["diagnosis"]["verdict"]["label"] == "completed"


def test_parse_resonance_raman_laser_energy(tmp_path):
    parsed = _parse(tmp_path, "formaldehyde_rr.out", """\
|  1> ! PBE0 def2-SVP def2/J RIJCOSX TightSCF ESD(RR)
Your calculation utilizes the basis: def2-SVP
* SCF CONVERGED AFTER 11 CYCLES *
FINAL SINGLE POINT ENERGY      -114.284008969306
Requested calculation:                         ...resonant Raman
Lineshape function:                            ...Voigt
Excited state PES:                             ...vertical gradient
Use operator derivatives:                      ...yes
Homogeneous linewidth is:                      50.00 cm-1
Inhomogeneous linewidth is:                    250.00 cm-1
Temperature used:                              0.00 K
Adiabatic energy difference:                   30844.79 cm-1
0-0 energy difference:                         30844.79 cm-1
The laser energy is:                           30844.79 cm-1
The resonant Raman spectrum was saved in       formaldehyde_rr.spectrum.30845
****ORCA ESD FINISHED WITHOUT ERROR****
****ORCA TERMINATED NORMALLY****
""")

    esd = parsed["derived"]["orca:esd"]
    assert parsed["tasks"][0]["name"] == "ORCA_ESD resonance_raman"
    assert esd["laser_energy_cm1"] == 30844.79
    assert esd["inhomogeneous_linewidth_cm1"] == 250.0
    assert esd["spectrum_file"] == "formaldehyde_rr.spectrum.30845"


def test_parse_phosphorescence_interfering_rate_components(tmp_path):
    parsed = _parse(tmp_path, "formaldehyde_phosphorescence.out", """\
|  1> ! PBE0 def2-SVP def2/J RIJCOSX TightSCF ESD(PHOSP) RI-SOMF(1X)
Your calculation utilizes the basis: def2-SVP
* SCF CONVERGED AFTER 11 CYCLES *
FINAL SINGLE POINT ENERGY      -114.284008969306
Requested calculation:                         ...phosphorescence
Homogeneous linewidth is:                      10.00 cm-1
Adiabatic energy difference:                   23336.22 cm-1
0-0 energy difference:                         22615.33 cm-1
The calculated phosphorescence rate constant is 4.810585e+01 s-1
with 304.69% from FC and -204.69% from HT
The phosphorescence spectrum was saved in      formaldehyde_phosphorescence.spectrum
****ORCA ESD FINISHED WITHOUT ERROR****
****ORCA TERMINATED NORMALLY****
""")

    esd = parsed["derived"]["orca:esd"]
    assert esd["rate_process"] == "phosphorescence"
    assert esd["rate_constant_s1"] == 48.10585
    assert esd["franck_condon_percent"] == 304.69
    assert esd["herzberg_teller_percent"] == -204.69


def test_esd_without_module_completion_is_failed(tmp_path):
    parsed = _parse(tmp_path, "interrupted.out", """\
|  1> ! PBE0 def2-SVP TightSCF ESD(ABS)
* SCF CONVERGED AFTER 11 CYCLES *
FINAL SINGLE POINT ENERGY      -114.284008969306
Requested calculation:                         ...absorption
****ORCA TERMINATED NORMALLY****
""")

    assert parsed["tasks"][0]["outcome"] == "failed"
    assert parsed["tasks"][0]["has_usable_data"] is False
    assert parsed["diagnosis"]["verdict"]["label"] == "failed"
