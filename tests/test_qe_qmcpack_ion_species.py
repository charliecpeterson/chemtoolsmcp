"""Regression coverage for QE and QMCPACK ion-species evidence."""

from __future__ import annotations

from pathlib import Path

from chemtools.mcp.tools.qe import _handle_inspect_qe_qmcpack_conversion_ion_species


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def test_ion_species_compares_element_counts_without_coordinates(tmp_path):
    qe_input = _write(tmp_path, "qe.in", """\
&CONTROL
 calculation = 'scf', disk_io = 'medium'
/
&SYSTEM
 ibrav = 0, nat = 1, ntyp = 1, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Ce 140.116 Ce.UPF
CELL_PARAMETERS bohr
24 0 0
0 24 0
0 0 24
ATOMIC_POSITIONS bohr
Ce 0 0 0
K_POINTS crystal
1
0 0 0 1
""")
    qe_output = _write(tmp_path, "qe.out", """\
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""")
    _write(tmp_path, "wavefunction.xml", """\
<qmcsystem><sposet_collection href="Ce.pwscf.h5"/></qmcsystem>
""")
    ions = _write(tmp_path, "ions.xml", """\
<qmcsystem>
  <particleset name="ion0" size="1"><group name="Ce"/></particleset>
  <particleset name="e"><group name="u" size="1"/></particleset>
</qmcsystem>
""")
    qmcpack_input = _write(tmp_path, "qmcpack.xml", """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="ions.xml"/>
  <hamiltonian name="h0" target="e"/>
</simulation>
""")
    pwscf_h5 = tmp_path / "Ce.pwscf.h5"
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)

    inspected = _handle_inspect_qe_qmcpack_conversion_ion_species({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"
    assert inspected["checks"][-1]["observed"]["qmcpack_element_counts"] == {"Ce": 1}

    ions.write_text(
        ions.read_text(encoding="utf-8").replace('name="Ce"', 'name="La"'),
        encoding="utf-8",
    )
    not_ready = _handle_inspect_qe_qmcpack_conversion_ion_species({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert not_ready["checks"][-1]["status"] == "not_ready"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path
