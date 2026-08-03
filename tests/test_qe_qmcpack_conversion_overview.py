"""End-to-end coverage for the aggregate QE-to-QMCPACK conversion tool."""

from __future__ import annotations

import json
from pathlib import Path

from chemtools.mcp.dispatch import handle_request
from chemtools.mcp.tools.qe import (
    _handle_inspect_qe_qmcpack_conversion,
    _handle_inspect_qe_qmcpack_conversion_execution,
    _handle_inspect_qe_qmcpack_conversion_projectors,
)
from chemtools.programs.qe.qmcpack import inspect_qmcpack_hdf5_deck_metadata


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_conversion_execution_requires_completed_converter_evidence(tmp_path):
    qe_input = _write(tmp_path, "qe.in", """\
&CONTROL
 calculation = 'scf', disk_io = 'medium', pseudo_dir = '.', prefix = 'H', outdir = '.'
/
&SYSTEM
 ibrav = 0, nat = 1, ntyp = 1, ecutwfc = 50, tot_charge = 0,
 nspin = 2, tot_magnetization = 1
/
&ELECTRONS
/
ATOMIC_SPECIES
H 1.008 H.UPF
CELL_PARAMETERS bohr
20 0 0
0 20 0
0 0 20
ATOMIC_POSITIONS bohr
H 0 0 0
K_POINTS crystal
1
0 0 0 1
""")
    _write(tmp_path, "H.UPF", """\
<UPF version="2.0.1">
  <PP_HEADER element="H" pseudo_type="NC" z_valence="1.0"/>
</UPF>
""")
    qe_output = _write(tmp_path, "qe.out", """\
Program PWSCF v.7.5 starts on  2Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""")
    converter_input = _write(tmp_path, "pw2qmcpack.in", """\
&inputpp
 prefix = 'H', outdir = '.', write_psir = .false.
/
""")
    pwscf_h5 = tmp_path / "H.pwscf.h5"
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)
    converter_output = _write(tmp_path, "pw2qmcpack.out", """\
Program pw2qmcpack v.7.5 starts on  2Aug2026
esh5 create H.pwscf.h5
JOB DONE.
""")
    _write(tmp_path, "wavefunction.xml", """\
<qmcsystem><sposet_collection href="H.pwscf.h5"/></qmcsystem>
""")
    _write(tmp_path, "ions.xml", """\
<qmcsystem>
  <simulationcell>
    <parameter name="lattice" units="bohr">20 0 0 0 20 0 0 0 20</parameter>
    <parameter name="bconds">p p p</parameter>
  </simulationcell>
  <particleset name="ion0" size="1">
    <group name="H"><parameter name="valence">1</parameter><attrib name="position">0 0 0</attrib></group>
  </particleset>
  <particleset name="e"><group name="u" size="1"/><group name="d" size="0"/></particleset>
</qmcsystem>
""")
    _write(tmp_path, "H.semilocal.xml", """\
<pseudo version="0.5">
  <header symbol="H" atomic-number="1" zval="1"/>
  <grid type="linear" units="bohr" ri="0" rf="10" npts="3"/>
  <semilocal units="hartree" format="r*V" l-local="0">
    <vps l="s"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="3"/><data>-1 -1 -1</data></radfunc></vps>
  </semilocal>
</pseudo>
""")
    qmcpack_input = _write(tmp_path, "qmcpack.xml", """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="ions.xml"/>
  <hamiltonian name="h0" target="e">
    <pairpot name="PseudoPot" type="pseudo">
      <pseudo elementType="H" href="H.semilocal.xml"/>
    </pairpot>
  </hamiltonian>
</simulation>
""")
    arguments = {
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pw2qmcpack_input": str(converter_input),
        "pw2qmcpack_output": str(converter_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    }

    inspected = _handle_inspect_qe_qmcpack_conversion_execution(arguments)

    assert inspected["schema_version"] == "chemtools.qe-qmcpack-conversion-execution/1"
    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"
    assert inspected["pw2qmcpack_inspection"]["assessment"]["verdict"] == {
        "label": "converter_completed",
        "confidence": 0.98,
        "reasons": ["pw2qmcpack reported creating H.pwscf.h5, then printed JOB DONE."],
    }

    response, should_exit = handle_request({
        "jsonrpc": "2.0",
        "id": "qe.qmcpack_conversion_execution",
        "method": "tools/call",
        "params": {
            "name": "inspect_qe_qmcpack_conversion_execution",
            "arguments": arguments,
        },
    })

    assert should_exit is False
    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["schema_version"] == "chemtools.qe-qmcpack-conversion-execution/1"
    assert payload["readiness"] == "ready"

    qe_input.write_text(
        qe_input.read_text(encoding="utf-8").replace("prefix = 'H'", "prefix = 'other'"),
        encoding="utf-8",
    )
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)

    wrong_source = _handle_inspect_qe_qmcpack_conversion_execution(arguments)

    assert wrong_source["readiness"] == "not_ready"
    assert wrong_source["checks"][-2] == {
        "name": "qe_pw2qmcpack_control_paths",
        "status": "not_ready",
        "observed": {
            "qe": {"prefix": "other", "outdir": "."},
            "pw2qmcpack": {"prefix": "H", "outdir": "."},
            "normalized_outdirs": {"qe": ".", "pw2qmcpack": "."},
        },
        "message": "QE and pw2qmcpack use different prefix or outdir values.",
    }

    qe_input.write_text(
        qe_input.read_text(encoding="utf-8").replace("prefix = 'other'", "prefix = 'H'"),
        encoding="utf-8",
    )
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)

    converter_input.write_text(
        converter_input.read_text(encoding="utf-8").replace(
            "write_psir = .false.", "write_psir = .true."
        ),
        encoding="utf-8",
    )
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)

    unsupported_converter = _handle_inspect_qe_qmcpack_conversion_execution(
        arguments
    )

    assert unsupported_converter["readiness"] == "review_required"
    assert unsupported_converter["checks"][-3] == {
        "name": "pw2qmcpack_input_scope",
        "status": "review_required",
        "observed": {
            "issues": [{
                "level": "warning",
                "message": (
                    "The demonstrated converter form uses write_psir=.false.; "
                    "review a different setting before conversion."
                ),
                "line": 1,
                "suggested_fix": None,
            }],
        },
        "message": "The converter input uses settings outside Chemtools' supported form.",
    }

    converter_input.write_text(
        converter_input.read_text(encoding="utf-8").replace(
            "write_psir = .true.", "write_psir = .false."
        ),
        encoding="utf-8",
    )
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)

    converter_output.write_text("""\
Program pw2qmcpack v.7.5 starts on  2Aug2026
esh5 create H.pwscf.h5
""", encoding="utf-8")

    incomplete = _handle_inspect_qe_qmcpack_conversion_execution(arguments)

    assert incomplete["readiness"] == "not_ready"
    assert incomplete["checks"][-1]["status"] == "not_ready"
    assert incomplete["checks"][-1]["message"] == (
        "pw2qmcpack did not report completed converter output."
    )


def test_conversion_overview_combines_supported_evidence(tmp_path):
    qe_input = _write(tmp_path, "qe.in", """\
&CONTROL
 calculation = 'scf', disk_io = 'medium', pseudo_dir = '.'
/
&SYSTEM
 ibrav = 0, nat = 1, ntyp = 1, ecutwfc = 50, tot_charge = 3,
 nspin = 2, tot_magnetization = 1
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
    _write(tmp_path, "Ce.UPF", """\
<UPF version="2.0.1">
  <PP_HEADER element="Ce" pseudo_type="NC" z_valence="12.0"/>
</UPF>
""")
    qe_output = _write(tmp_path, "qe.out", """\
Program PWSCF v.7.5 starts on  2Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""")
    _write(tmp_path, "wavefunction.xml", """\
<qmcsystem><sposet_collection href="Ce.pwscf.h5"/></qmcsystem>
""")
    _write(tmp_path, "ions.xml", """\
<qmcsystem>
  <simulationcell>
    <parameter name="lattice" units="bohr">24 0 0 0 24 0 0 0 24</parameter>
    <parameter name="bconds">p p p</parameter>
  </simulationcell>
  <particleset name="ion0" size="1">
    <group name="Ce"><parameter name="valence">12</parameter><attrib name="position">0 0 0</attrib></group>
  </particleset>
  <particleset name="e">
    <group name="u" size="5"/>
    <group name="d" size="4"/>
  </particleset>
</qmcsystem>
""")
    _write(tmp_path, "Ce.semilocal.xml", """\
<pseudo version="0.5">
  <header symbol="Ce" atomic-number="58" zval="12"/>
  <grid type="linear" units="bohr" ri="0" rf="10" npts="3"/>
  <semilocal units="hartree" format="r*V" l-local="1">
    <vps l="s"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="3"/><data>-12 -12 -12</data></radfunc></vps>
    <vps l="p"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="3"/><data>-12 -12 -12</data></radfunc></vps>
  </semilocal>
</pseudo>
""")
    qmcpack_input = _write(tmp_path, "qmcpack.xml", """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="ions.xml"/>
  <hamiltonian name="h0" target="e">
    <pairpot name="PseudoPot" type="pseudo">
      <pseudo elementType="Ce" href="Ce.semilocal.xml"/>
    </pairpot>
  </hamiltonian>
</simulation>
""")
    pwscf_h5 = tmp_path / "Ce.pwscf.h5"
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)

    inspected = _handle_inspect_qe_qmcpack_conversion({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["schema_version"] == "chemtools.qe-qmcpack-conversion/1"
    assert inspected["readiness"] == "ready"
    assert len(inspected["checks"]) == 18
    assert {check["status"] for check in inspected["checks"]} == {
        "not_applicable",
        "pass",
    }
    (tmp_path / "Ce.UPF").write_text("""\
<UPF version="2.0.1">
  <PP_HEADER element="Ce" pseudo_type="NC" z_valence="12.0" number_of_proj="2"/>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="3">0</PP_BETA.1>
    <PP_BETA.2 index="2" angular_momentum="3">0</PP_BETA.2>
  </PP_NONLOCAL>
</UPF>
""", encoding="utf-8")
    qmcpack_input.write_text(
        qmcpack_input.read_text(encoding="utf-8").replace(
            "</simulation>",
            "  <include href=\"dmc.xml\"/>\n</simulation>",
        ),
        encoding="utf-8",
    )
    _write(tmp_path, "dmc.xml", """\
<simulation>
  <project id="Ce"/>
  <qmc method="vmc"/>
  <qmc method="dmc"/>
</simulation>
""")
    projector_review = _handle_inspect_qe_qmcpack_conversion_projectors({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert projector_review["readiness"] == "review_required"
    assert projector_review["checks"][-1] == {
        "name": "qe_qmcpack_projector_evidence",
        "status": "review_required",
        "observed": {
            "qmcpack_dmc_blocks": [{
                "source_path": str((tmp_path / "dmc.xml").resolve()),
                "qmc_block_index": 1,
            }],
            "qe_pseudopotentials": [{
                "species_label": "Ce",
                "path": str((tmp_path / "Ce.UPF").resolve()),
                "status": "parsed",
                "pseudo_type": "NC",
                "projector_channel_evidence": {
                    "status": "complete",
                    "declared_total": 2,
                    "observed_total": 2,
                    "invalid_angular_momentum_count": 0,
                    "counts_by_angular_momentum": {"3": 2},
                    "declared_total_matches_observed": True,
                },
            }],
            "include_review_status": "reviewed",
            "incomplete_qe_pseudopotentials": [],
            "non_semilocal_qe_pseudopotentials": [],
            "multi_projector_channels": [{
                "species_label": "Ce",
                "path": str((tmp_path / "Ce.UPF").resolve()),
                "counts_by_angular_momentum": {"3": 2},
            }],
        },
        "message": (
            "At least one QE UPF has multiple projectors in an angular channel; "
            "confirm that QMCPACK DMC uses a separately generated semilocal "
            "potential rather than a reconstructed projector form."
        ),
    }

    qmcpack_input.write_text(
        qmcpack_input.read_text(encoding="utf-8").replace(
            '  <include href="dmc.xml"/>\n',
            '  <qmc method="dmc"/>\n',
        ),
        encoding="utf-8",
    )
    primary_dmc_review = _handle_inspect_qe_qmcpack_conversion_projectors({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert primary_dmc_review["checks"][-1]["observed"][
        "qmcpack_dmc_blocks"
    ] == [{
        "source_path": str(qmcpack_input.resolve()),
        "qmc_block_index": 0,
    }]

    (tmp_path / "Ce.UPF").write_text("""\
<UPF version="2.0.1">
  <PP_HEADER element="Ce" pseudo_type="PAW" z_valence="12.0" number_of_proj="1"/>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="3">0</PP_BETA.1>
  </PP_NONLOCAL>
</UPF>
""", encoding="utf-8")
    non_semilocal_review = _handle_inspect_qe_qmcpack_conversion_projectors({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert non_semilocal_review["checks"][-1]["observed"][
        "non_semilocal_qe_pseudopotentials"
    ] == [{
        "species_label": "Ce",
        "path": str((tmp_path / "Ce.UPF").resolve()),
        "pseudo_type": "PAW",
    }]

    (tmp_path / "Ce.UPF").write_text("""\
<UPF version="2.0.1">
  <PP_HEADER element="Ce" pseudo_type="SL" z_valence="12.0" number_of_proj="2"/>
  <PP_SEMILOCAL>
    <PP_VNL.1 L="0">0</PP_VNL.1>
  </PP_SEMILOCAL>
</UPF>
""", encoding="utf-8")
    semilocal_source = _handle_inspect_qe_qmcpack_conversion_projectors({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert semilocal_source["checks"][-1]["status"] == "pass"
    assert semilocal_source["checks"][-1]["observed"][
        "incomplete_qe_pseudopotentials"
    ] == []


def test_recognized_pwscf_metadata_must_match_the_qmcpack_deck():
    parsed_qmcpack = {
        "hamiltonians": [{"target": "e"}],
        "particle_sets": [
            {
                "name": "ion0",
                "size": "1",
                "groups": [{"name": "H", "size": None, "parameters": {}}],
            },
            {
                "name": "e",
                "size": None,
                "groups": [
                    {"name": "u", "size": "1", "parameters": {}},
                    {"name": "d", "size": "0", "parameters": {}},
                ],
            },
        ],
    }
    include_review = {"status": "not_applicable", "entries": []}
    hdf5_inspection = {
        "status": "recognized",
        "artifact_kind": "pwscf_wavefunction",
        "source": {"path": "/work/H.pwscf.h5"},
        "wavefunction": {
            "atoms": {
                "count": 1,
                "species": [{"name": "H", "atom_count": 1}],
            },
            "electrons": {"spin_populations": [1, 0], "spin_count": 2},
        },
    }

    matched = inspect_qmcpack_hdf5_deck_metadata(
        hdf5_inspection,
        parsed_qmcpack,
        include_review,
    )

    assert matched["status"] == "pass"
    assert matched["observed"]["hdf5"]["species_counts"] == {"H": 1}

    hdf5_inspection["wavefunction"]["electrons"]["spin_populations"] = [0, 1]
    mismatched = inspect_qmcpack_hdf5_deck_metadata(
        hdf5_inspection,
        parsed_qmcpack,
        include_review,
    )

    assert mismatched["status"] == "not_ready"
    assert mismatched["message"] == (
        "pw2qmcpack HDF5 metadata does not match the QMCPACK deck for spin_populations."
    )
