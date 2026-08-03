"""Regression coverage for the QE input checks before pw2qmcpack conversion."""

from __future__ import annotations

import os
from pathlib import Path

from chemtools.mcp.tools.qe import (
    _handle_check_qe_qmcpack_conversion_ready,
    _handle_inspect_qe_qmcpack_conversion_artifacts,
    _handle_inspect_qe_qmcpack_conversion_atoms,
    _handle_inspect_qe_qmcpack_conversion_charge,
    _handle_inspect_qe_qmcpack_conversion_deck,
    _handle_inspect_qe_qmcpack_conversion_electrons,
    _handle_inspect_qe_qmcpack_conversion_geometry,
    _handle_inspect_qe_qmcpack_conversion_pseudopotentials,
    _handle_inspect_qe_qmcpack_conversion_spin,
    _handle_inspect_qe_qmcpack_conversion_species,
    _handle_inspect_qe_qmcpack_conversion_valence,
    _handle_draft_ph_x_input,
    _handle_draft_pw2qmcpack_input,
    _handle_plan_qe_qmcpack_conversion,
)


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _write(tmp_path: Path, text: str, *, name: str = "qe.in") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_hdf5(tmp_path: Path, *, name: str = "Ce.pwscf.h5") -> Path:
    path = tmp_path / name
    path.write_bytes(_HDF5_SIGNATURE)
    return path


def _input(*, disk_io: str = "medium", k_points: str = """\
K_POINTS crystal
1
0.0 0.0 0.0 1.0
""", isolated: str = "") -> str:
    return f"""\
&CONTROL
 calculation = 'scf', disk_io = '{disk_io}'
/
&SYSTEM
 ibrav = 0, nat = 1, ntyp = 1, ecutwfc = 50 {isolated}
/
&ELECTRONS
/
ATOMIC_SPECIES
Ce 140.116 Ce.UPF
CELL_PARAMETERS bohr
24.0 0.0 0.0
0.0 24.0 0.0
0.0 0.0 24.0
ATOMIC_POSITIONS bohr
Ce 0.0 0.0 0.0
{k_points}"""


def test_qe_qmcpack_conversion_readiness_accepts_documented_input(tmp_path):
    result = _handle_check_qe_qmcpack_conversion_ready({
        "qe_input": str(_write(tmp_path, _input())),
    })

    assert result["schema_version"] == "chemtools.qe-qmcpack-conversion-readiness/1"
    assert result["readiness"] == "ready"
    assert [check["status"] for check in result["checks"]] == [
        "pass", "pass", "pass", "pass",
    ]


def test_qe_qmcpack_conversion_readiness_accepts_fblock_template():
    result = _handle_check_qe_qmcpack_conversion_ready({
        "qe_input": "notes/fblock/examples/qe/Ce-ion-in-box-for-qmcpack.in",
    })

    assert result["readiness"] == "ready"


def test_qe_qmcpack_conversion_plan_preserves_declared_handoff(tmp_path):
    qe_input = _write(tmp_path, _input())
    planned_h5 = tmp_path / "artifacts" / "Ce.pwscf.h5"

    plan = _handle_plan_qe_qmcpack_conversion({
        "qe_input": str(qe_input),
        "pwscf_h5": str(planned_h5),
    })

    assert plan["schema_version"] == "chemtools.qe-qmcpack-conversion-plan/1"
    assert plan["readiness"] == "ready"
    assert plan["pwscf_h5"] == str(planned_h5.resolve())
    assert plan["steps"] == [{
        "id": "qe_scf",
        "program": "qe",
        "executable": "pw.x",
        "input": str(qe_input.resolve()),
        "required_evidence": "completed_converged_scf_output",
    }, {
        "id": "pw2qmcpack_conversion",
        "program": "qe",
        "executable": "pw2qmcpack.x",
        "requires": {
            "qe_input": str(qe_input.resolve()),
            "scf_wavefunctions": "retained_by_qe",
        },
        "produces": {"pwscf_h5": str(planned_h5.resolve())},
        "command_line": None,
    }, {
        "id": "qmcpack_deck_validation",
        "program": "qmcpack",
        "requires": {"pwscf_h5": str(planned_h5.resolve())},
        "tool": "inspect_qe_qmcpack_conversion_deck",
    }]


def test_qe_qmcpack_conversion_plan_retains_preflight_failure(tmp_path):
    plan = _handle_plan_qe_qmcpack_conversion({
        "qe_input": str(_write(tmp_path, _input(disk_io="low"))),
        "pwscf_h5": str(tmp_path / "Ce.pwscf.h5"),
    })

    assert plan["readiness"] == "not_ready"
    assert plan["preflight"][1]["status"] == "not_ready"


def test_draft_pw2qmcpack_input_uses_explicit_qe_control_paths(tmp_path):
    qe_input = _write(tmp_path, _input().replace(
        "calculation = 'scf', disk_io = 'medium'",
        "calculation = 'scf', disk_io = 'medium', prefix = 'Ce', outdir = './qmcprep'",
    ))

    drafted = _handle_draft_pw2qmcpack_input({"qe_input": str(qe_input)})

    assert drafted == {
        "schema_version": "chemtools.pw2qmcpack-input-draft/1",
        "qe_input": str(qe_input.resolve()),
        "status": "ready",
        "input_text": (
            "&inputpp\n"
            "  prefix = 'Ce'\n"
            "  outdir = './qmcprep'\n"
            "  write_psir = .false.\n"
            "/\n"
        ),
        "checks": [{
            "name": "prefix",
            "status": "pass",
            "observed": "Ce",
            "source_line": 2,
            "message": "QE &CONTROL prefix is an explicit renderable path.",
        }, {
            "name": "outdir",
            "status": "pass",
            "observed": "./qmcprep",
            "source_line": 2,
            "message": "QE &CONTROL outdir is an explicit renderable path.",
        }],
        "scope_limit": (
            "This drafts only the supported inputpp form. It does not infer QE "
            "defaults, select converter options, launch pw2qmcpack, or inspect "
            "the resulting HDF5 file."
        ),
    }


def test_draft_pw2qmcpack_input_refuses_missing_qe_control_paths(tmp_path):
    drafted = _handle_draft_pw2qmcpack_input({
        "qe_input": str(_write(tmp_path, _input())),
    })

    assert drafted["status"] == "review_required"
    assert drafted["input_text"] is None
    assert [check["status"] for check in drafted["checks"]] == [
        "review_required", "review_required",
    ]


def test_draft_ph_x_input_copies_qe_paths_and_preserves_gamma_advisory(tmp_path):
    qe_input = _write(tmp_path, _input().replace(
        "calculation = 'scf', disk_io = 'medium'",
        "calculation = 'scf', disk_io = 'medium', prefix = 'Ce', outdir = './qmcprep'",
    ))

    drafted = _handle_draft_ph_x_input({
        "qe_input": str(qe_input),
        "title": "Ce single-q phonon",
        "q_point": [0, 0, 0],
    })

    assert drafted["status"] == "ready"
    assert drafted["input_text"] == (
        "Ce single-q phonon\n"
        "&INPUTPH\n"
        "  prefix = 'Ce'\n"
        "  outdir = './qmcprep'\n"
        "/\n"
        "0 0 0\n"
    )
    assert drafted["expected_artifacts"] == {"dynamical_matrix": "matdyn"}
    assert drafted["advisories"] == [{
        "name": "gamma_nonanalytic_terms",
        "status": "review_required",
        "message": (
            "At q=0, this draft leaves epsil at its QE default. Decide whether "
            "the non-analytic LO-TO term is needed before running ph.x."
        ),
    }]


def test_draft_ph_x_input_refuses_invalid_q_vector_and_missing_paths(tmp_path):
    drafted = _handle_draft_ph_x_input({
        "qe_input": str(_write(tmp_path, _input())),
        "title": "phonon",
        "q_point": [0, 0],
    })

    assert drafted["status"] == "review_required"
    assert drafted["input_text"] is None
    assert drafted["expected_artifacts"] == {}
    assert drafted["advisories"] == []
    assert [check["status"] for check in drafted["checks"]] == [
        "review_required", "review_required", "pass", "review_required",
    ]


def test_qe_qmcpack_conversion_readiness_blocks_known_incompatible_inputs(tmp_path):
    result = _handle_check_qe_qmcpack_conversion_ready({
        "qe_input": str(_write(tmp_path, _input(
            disk_io="low",
            k_points="K_POINTS gamma\n",
            isolated=", assume_isolated = 'm-t'",
        ))),
    })

    assert result["readiness"] == "not_ready"
    assert [check["status"] for check in result["checks"]] == [
        "pass", "not_ready", "not_ready", "not_ready",
    ]


def test_qe_qmcpack_conversion_readiness_requires_review_for_unsupported_case(tmp_path):
    result = _handle_check_qe_qmcpack_conversion_ready({
        "qe_input": str(_write(tmp_path, _input(
            disk_io="nowf",
            k_points="K_POINTS automatic\n1 1 1 0 0 0\n",
            isolated=", assume_isolated = 'esm'",
        ))),
    })

    assert result["readiness"] == "review_required"
    assert [check["status"] for check in result["checks"]] == [
        "pass", "review_required", "review_required", "review_required",
    ]


def test_qe_qmcpack_conversion_artifacts_require_completed_output_and_current_h5(tmp_path):
    qe_input = _write(tmp_path, _input())
    qe_output = tmp_path / "qe.out"
    qe_output.write_text("""\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", encoding="utf-8")
    pwscf_h5 = _write_hdf5(tmp_path)

    inspected = _handle_inspect_qe_qmcpack_conversion_artifacts({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
    })

    assert inspected["readiness"] == "ready"
    assert [check["status"] for check in inspected["checks"]] == ["pass"] * 6

    os.utime(pwscf_h5, ns=(1, 1))
    stale = _handle_inspect_qe_qmcpack_conversion_artifacts({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
    })
    assert stale["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_deck_requires_matching_hdf5_reference(tmp_path):
    qe_input = _write(tmp_path, _input())
    qe_output = tmp_path / "qe.out"
    qe_output.write_text("""\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", encoding="utf-8")
    pwscf_h5 = tmp_path / "Ce.pwscf.h5"
    _write(tmp_path, """\
<qmcsystem>
  <sposet_collection href="Ce.pwscf.h5"/>
</qmcsystem>
""", name="wavefunction.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
</simulation>
""", name="qmcpack.xml")
    pwscf_h5.write_bytes(_HDF5_SIGNATURE)

    inspected = _handle_inspect_qe_qmcpack_conversion_deck({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert [check["status"] for check in inspected["checks"]] == ["pass"] * 7
    assert inspected["checks"][-1]["observed"]["matching_references"] == [{
        "href": "Ce.pwscf.h5",
        "path": str(pwscf_h5.resolve()),
        "reference_kinds": ["sposet_collection"],
        "source_path": str((tmp_path / "wavefunction.xml").resolve()),
        "status": "present",
        "freshness": "not_older_than_input",
        "size_bytes": 8,
        "modified_ns": pwscf_h5.stat().st_mtime_ns,
        "hdf5_signature_offset": 0,
    }]

    missing_reference = _write(
        tmp_path,
        "<simulation/>",
        name="missing.xml",
    )
    not_ready = _handle_inspect_qe_qmcpack_conversion_deck({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(missing_reference),
    })
    assert not_ready["checks"][-1]["status"] == "not_ready"

    sidecar_only = _write(
        tmp_path,
        "<simulation><override_variational_parameters href=\"Ce.pwscf.h5\"/></simulation>",
        name="sidecar-only.xml",
    )
    wrong_kind = _handle_inspect_qe_qmcpack_conversion_deck({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(sidecar_only),
    })
    assert wrong_kind["checks"][-1]["status"] == "not_ready"
    assert wrong_kind["checks"][-1]["observed"][
        "non_orbital_matching_references"
    ][0]["reference_kinds"] == ["override_variational_parameters"]


def test_qe_qmcpack_conversion_pseudopotentials_require_semilocal_card(tmp_path):
    qe_input = _write(tmp_path, _input())
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    pwscf_h5 = _write_hdf5(tmp_path)
    _write(tmp_path, """\
<qmcsystem>
  <sposet_collection href="Ce.pwscf.h5"/>
</qmcsystem>
""", name="wavefunction.xml")
    pseudopotential = _write(tmp_path, """\
<pseudo version="0.5">
  <header symbol="Ce" atomic-number="58" zval="12"/>
  <grid type="linear" units="bohr" ri="0" rf="10" npts="3"/>
  <semilocal units="hartree" format="r*V" l-local="1">
    <vps l="s"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="3"/><data>-12 -12 -12</data></radfunc></vps>
    <vps l="p"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="3"/><data>-12 -12 -12</data></radfunc></vps>
  </semilocal>
</pseudo>
""", name="Ce.semilocal.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <hamiltonian name="h0" target="e">
    <pairpot name="PseudoPot" type="pseudo">
      <pseudo elementType="Ce" href="Ce.semilocal.xml"/>
    </pairpot>
  </hamiltonian>
</simulation>
""", name="qmcpack.xml")

    inspected = _handle_inspect_qe_qmcpack_conversion_pseudopotentials({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    pseudopotential_check = inspected["checks"][-1]
    assert pseudopotential_check["status"] == "pass"
    assert pseudopotential_check["observed"]["inspections"][0]["path"] == str(
        pseudopotential.resolve(),
    )

    pseudopotential.write_text("<pseudo/>", encoding="utf-8")
    not_ready = _handle_inspect_qe_qmcpack_conversion_pseudopotentials({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })
    assert not_ready["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_species_follow_nested_pseudopotential_references(
    tmp_path,
):
    qe_input = _write(tmp_path, _input())
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    pwscf_h5 = _write_hdf5(tmp_path)
    _write(tmp_path, """\
<qmcsystem>
  <sposet_collection href="Ce.pwscf.h5"/>
</qmcsystem>
""", name="wavefunction.xml")
    potential = _write(tmp_path, """\
<qmcsystem>
  <hamiltonian name="h0" target="e">
    <pairpot name="PseudoPot" type="pseudo">
      <pseudo elementType="Ce" href="Ce.semilocal.xml"/>
    </pairpot>
  </hamiltonian>
</qmcsystem>
""", name="potential.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="potential.xml"/>
</simulation>
""", name="qmcpack.xml")

    inspected = _handle_inspect_qe_qmcpack_conversion_species({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"
    assert inspected["checks"][-1]["observed"]["qmcpack_elements"] == ["Ce"]

    potential.write_text(potential.read_text(encoding="utf-8").replace(
        'elementType="Ce"',
        'elementType="La"',
    ), encoding="utf-8")
    not_ready = _handle_inspect_qe_qmcpack_conversion_species({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert not_ready["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_valence_compares_upf_and_xml_headers(tmp_path):
    qe_input = _write(
        tmp_path,
        _input().replace("disk_io = 'medium'", "disk_io = 'medium', pseudo_dir = '.'"),
    )
    _write(tmp_path, """\
<UPF version="2.0.1">
  <PP_HEADER element="Ce" pseudo_type="NC" z_valence="12.0"/>
</UPF>
""", name="Ce.UPF")
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    pwscf_h5 = _write_hdf5(tmp_path)
    _write(tmp_path, """\
<qmcsystem>
  <sposet_collection href="Ce.pwscf.h5"/>
</qmcsystem>
""", name="wavefunction.xml")
    pseudopotential = _write(tmp_path, """\
<pseudo version="0.5">
  <header symbol="Ce" atomic-number="58" zval="12"/>
  <grid type="linear" units="bohr" ri="0" rf="10" npts="3"/>
  <semilocal units="hartree" format="r*V" l-local="1">
    <vps l="s"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="3"/><data>-12 -12 -12</data></radfunc></vps>
    <vps l="p"><radfunc><grid type="linear" units="bohr" ri="0" rf="10" npts="3"/><data>-12 -12 -12</data></radfunc></vps>
  </semilocal>
</pseudo>
""", name="Ce.semilocal.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <hamiltonian name="h0" target="e">
    <pairpot name="PseudoPot" type="pseudo">
      <pseudo elementType="Ce" href="Ce.semilocal.xml"/>
    </pairpot>
  </hamiltonian>
</simulation>
""", name="qmcpack.xml")

    inspected = _handle_inspect_qe_qmcpack_conversion_valence({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"
    assert inspected["checks"][-1]["observed"]["qe_valence_by_element"] == {
        "Ce": 12.0,
    }

    pseudopotential.write_text(
        pseudopotential.read_text(encoding="utf-8").replace('zval="12"', 'zval="11"'),
        encoding="utf-8",
    )
    not_ready = _handle_inspect_qe_qmcpack_conversion_valence({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert not_ready["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_electrons_match_runtime_and_upf_evidence(tmp_path):
    qe_input = _write(
        tmp_path,
        _input().replace("disk_io = 'medium'", "disk_io = 'medium', pseudo_dir = '.'"),
    )
    _write(tmp_path, """\
<UPF version="2.0.1">
  <PP_HEADER element="Ce" pseudo_type="NC" z_valence="12.0"/>
</UPF>
""", name="Ce.UPF")
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
number of electrons = 12.00
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    pwscf_h5 = _write_hdf5(tmp_path)
    _write(tmp_path, """\
<qmcsystem>
  <sposet_collection href="Ce.pwscf.h5"/>
</qmcsystem>
""", name="wavefunction.xml")
    _write(tmp_path, """\
<qmcsystem>
  <particleset name="e">
    <group name="u" size="6"/>
    <group name="d" size="6"/>
  </particleset>
</qmcsystem>
""", name="particles.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="particles.xml"/>
  <hamiltonian name="h0" target="e"/>
</simulation>
""", name="qmcpack.xml")

    inspected = _handle_inspect_qe_qmcpack_conversion_electrons({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    electron_check = inspected["checks"][-1]
    assert electron_check["status"] == "pass"
    assert electron_check["observed"]["qmcpack"]["electron_count"] == 12

    _write(tmp_path, """\
<qmcsystem>
  <particleset name="e">
    <group name="u" size="6"/>
    <group name="d" size="5"/>
  </particleset>
</qmcsystem>
""", name="particles.xml")
    not_ready = _handle_inspect_qe_qmcpack_conversion_electrons({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })
    assert not_ready["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_atoms_match_non_electron_particle_sets(tmp_path):
    qe_input = _write(tmp_path, _input())
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    pwscf_h5 = _write_hdf5(tmp_path)
    _write(tmp_path, """\
<qmcsystem>
  <sposet_collection href="Ce.pwscf.h5"/>
</qmcsystem>
""", name="wavefunction.xml")
    ions = _write(tmp_path, """\
<qmcsystem>
  <particleset name="ion0" size="1">
    <group name="Ce"/>
  </particleset>
  <particleset name="e">
    <group name="u" size="6"/>
    <group name="d" size="6"/>
  </particleset>
</qmcsystem>
""", name="particles.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="particles.xml"/>
  <hamiltonian name="h0" target="e"/>
</simulation>
""", name="qmcpack.xml")

    inspected = _handle_inspect_qe_qmcpack_conversion_atoms({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"

    ions.write_text("""\
<qmcsystem>
  <particleset name="ion0" size="2">
    <group name="Ce"/>
  </particleset>
  <particleset name="e">
    <group name="u" size="6"/>
    <group name="d" size="6"/>
  </particleset>
</qmcsystem>
""", encoding="utf-8")
    not_ready = _handle_inspect_qe_qmcpack_conversion_atoms({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })
    assert not_ready["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_geometry_compares_periodic_cell_and_ions(tmp_path):
    qe_input = _write(
        tmp_path,
        _input().replace("Ce 0.0 0.0 0.0", "Ce 12.0 12.0 12.0"),
    )
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    _write(tmp_path, """\
<qmcsystem>
  <sposet_collection href="Ce.pwscf.h5"/>
</qmcsystem>
""", name="wavefunction.xml")
    ions = _write(tmp_path, """\
<qmcsystem>
  <simulationcell>
    <parameter name="lattice" units="bohr">24 0 0 0 24 0 0 0 24</parameter>
    <parameter name="bconds">p p p</parameter>
  </simulationcell>
  <particleset name="ion0" size="1">
    <group name="Ce"><attrib name="position">12 12 12</attrib></group>
  </particleset>
</qmcsystem>
""", name="ions.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="ions.xml"/>
  <hamiltonian name="h0" target="e"/>
</simulation>
""", name="qmcpack.xml")
    pwscf_h5 = _write_hdf5(tmp_path)

    inspected = _handle_inspect_qe_qmcpack_conversion_geometry({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"

    ions.write_text(
        ions.read_text(encoding="utf-8").replace("12 12 12", "11 12 12"),
        encoding="utf-8",
    )
    not_ready = _handle_inspect_qe_qmcpack_conversion_geometry({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert not_ready["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_geometry_rejects_singular_periodic_cells(tmp_path):
    qe_input = _write(
        tmp_path,
        _input()
        .replace("0.0 24.0 0.0", "0.0 0.0 0.0")
        .replace("Ce 0.0 0.0 0.0", "Ce 12.0 0.0 12.0"),
    )
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    _write(tmp_path, """\
<qmcsystem><sposet_collection href="Ce.pwscf.h5"/></qmcsystem>
""", name="wavefunction.xml")
    _write(tmp_path, """\
<qmcsystem>
  <simulationcell>
    <parameter name="lattice" units="bohr">24 0 0 0 0 0 0 0 24</parameter>
    <parameter name="bconds">p p p</parameter>
  </simulationcell>
  <particleset name="ion0" size="1">
    <group name="Ce"><attrib name="position">12 0 12</attrib></group>
  </particleset>
</qmcsystem>
""", name="ions.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="ions.xml"/>
  <hamiltonian name="h0" target="e"/>
</simulation>
""", name="qmcpack.xml")
    pwscf_h5 = _write_hdf5(tmp_path)

    inspected = _handle_inspect_qe_qmcpack_conversion_geometry({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    geometry_check = inspected["checks"][-1]
    assert inspected["readiness"] == "not_ready"
    assert geometry_check["status"] == "not_ready"
    assert geometry_check["observed"]["qe_cell_volume_angstrom3"] is None
    assert geometry_check["observed"]["qmcpack_cell_volume_angstrom3"] is None


def test_qe_qmcpack_conversion_spin_compares_fixed_collinear_moment(tmp_path):
    qe_input = _write(
        tmp_path,
        _input().replace("ecutwfc = 50", "ecutwfc = 50, nspin = 2, tot_magnetization = 2"),
    )
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    _write(tmp_path, """\
<qmcsystem><sposet_collection href="Ce.pwscf.h5"/></qmcsystem>
""", name="wavefunction.xml")
    particles = _write(tmp_path, """\
<qmcsystem>
  <particleset name="e">
    <group name="u" size="7"/>
    <group name="d" size="5"/>
  </particleset>
</qmcsystem>
""", name="particles.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="particles.xml"/>
  <hamiltonian name="h0" target="e"/>
</simulation>
""", name="qmcpack.xml")
    pwscf_h5 = _write_hdf5(tmp_path)

    inspected = _handle_inspect_qe_qmcpack_conversion_spin({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"
    assert inspected["checks"][-1]["observed"]["qmcpack_spin_imbalance"] == 2

    particles.write_text(
        particles.read_text(encoding="utf-8").replace('name="d" size="5"', 'name="d" size="4"'),
        encoding="utf-8",
    )
    not_ready = _handle_inspect_qe_qmcpack_conversion_spin({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert not_ready["checks"][-1]["status"] == "not_ready"


def test_qe_qmcpack_conversion_charge_compares_explicit_valence_evidence(tmp_path):
    qe_input = _write(
        tmp_path,
        _input().replace("disk_io = 'medium'", "disk_io = 'medium', pseudo_dir = '.'").replace(
            "ecutwfc = 50", "ecutwfc = 50, tot_charge = 3",
        ),
    )
    _write(tmp_path, """\
<UPF version="2.0.1">
  <PP_HEADER element="Ce" pseudo_type="NC" z_valence="12.0"/>
</UPF>
""", name="Ce.UPF")
    qe_output = _write(tmp_path, """\
Program PWSCF v.7.5 starts on  1Aug2026
Self-consistent Calculation
convergence has been achieved in 8 iterations
JOB DONE.
""", name="qe.out")
    _write(tmp_path, """\
<qmcsystem><sposet_collection href="Ce.pwscf.h5"/></qmcsystem>
""", name="wavefunction.xml")
    particles = _write(tmp_path, """\
<qmcsystem>
  <particleset name="ion0" size="1">
    <group name="Ce"><parameter name="valence">12</parameter></group>
  </particleset>
  <particleset name="e">
    <group name="u" size="5"/>
    <group name="d" size="4"/>
  </particleset>
</qmcsystem>
""", name="particles.xml")
    qmcpack_input = _write(tmp_path, """\
<simulation>
  <include href="wavefunction.xml"/>
  <include href="particles.xml"/>
  <hamiltonian name="h0" target="e"/>
</simulation>
""", name="qmcpack.xml")
    pwscf_h5 = _write_hdf5(tmp_path)

    inspected = _handle_inspect_qe_qmcpack_conversion_charge({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert inspected["readiness"] == "ready"
    assert inspected["checks"][-1]["status"] == "pass"
    assert inspected["checks"][-1]["observed"]["qmcpack_net_charge"] == 3.0

    particles.write_text(
        particles.read_text(encoding="utf-8").replace('name="d" size="4"', 'name="d" size="5"'),
        encoding="utf-8",
    )
    not_ready = _handle_inspect_qe_qmcpack_conversion_charge({
        "qe_input": str(qe_input),
        "qe_output": str(qe_output),
        "pwscf_h5": str(pwscf_h5),
        "qmcpack_input": str(qmcpack_input),
    })

    assert not_ready["checks"][-1]["status"] == "not_ready"
