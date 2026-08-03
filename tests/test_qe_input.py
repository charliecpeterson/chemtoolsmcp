"""Regression tests for the initial Quantum ESPRESSO pw.x input review."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from chemtools.application.input_review import (
    detect_input_backend,
    review_input,
)
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend
from chemtools.programs.qe import QE
from chemtools.programs.qe.input import (
    lint_pw_input,
    parse_pw_input,
    unsupported_qe_program,
)
from chemtools.programs.qe.phonon import lint_ph_x_input, parse_ph_x_text
from chemtools.programs.qe.pw2qmcpack import (
    is_pw2qmcpack_input,
    lint_pw2qmcpack_input,
    parse_pw2qmcpack_text,
)
from chemtools.programs.qe.pseudopotentials import parse_upf_header


SCF_INPUT = """\
 &control
   calculation = 'scf', pseudo_dir = '.'
 /
 &system
   ibrav = 2, nat = 2, ntyp = 1,
   ecutwfc = 1.8d1,
 /
 &electrons
   conv_thr = 1.0d-8
 /
ATOMIC_SPECIES
 Si 28.086 Si.pbe.UPF
ATOMIC_POSITIONS (alat)
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS {automatic}
 4 4 4 1 1 1
"""


VC_RELAX_INPUT = """\
&CONTROL
 calculation = 'vc-relax',
/
&SYSTEM
 ibrav = 0, nat = 2, ntyp = 2,
 ecutwfc = 70,
 occupations = 'smearing', smearing = 'mp', degauss = 0.02,
 nspin = 2, starting_magnetization(1) = 0.8,
/
&ELECTRONS
/
&IONS
 ion_dynamics = 'bfgs'
/
&CELL
 cell_dynamics = 'bfgs'
/
CELL_PARAMETERS angstrom
 4.0 0.0 0.0
 0.0 4.0 0.0
 0.0 0.0 4.0
ATOMIC_SPECIES
 Fe 55.845 Fe.pbe.UPF
 O 15.999 O.pbe.UPF
ATOMIC_POSITIONS angstrom
 Fe 0.0 0.0 0.0
 O  2.0 2.0 2.0
K_POINTS automatic
 6 6 6 0 0 0
"""


PHONON_INPUT = """\
Si single-q phonon
&INPUTPH
 prefix = 'si', outdir = './tmp'
/
0.25 0.0 0.0
"""


PW2QMCPACK_INPUT = """\
&inputpp
 prefix = 'si', outdir = './tmp', write_psir = .false.
/
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_upf(
    tmp_path: Path,
    *,
    filename: str = "Si.pbe.UPF",
    element: str = "Si",
    pseudo_type: str = "PAW",
    z_valence: float = 4.0,
    has_spin_orbit: bool = False,
    wfc_cutoff: float = 16.0,
    rho_cutoff: float = 70.0,
) -> Path:
    return _write(
        tmp_path,
        filename,
        (
            '<UPF version="2.0.1">\n'
            f'<PP_HEADER element="{element}" pseudo_type="{pseudo_type}" '
            'relativistic="scalar" functional="PBE" '
            f'z_valence="{z_valence}" is_ultrasoft="true" is_paw="true" '
            f'has_so="{str(has_spin_orbit).lower()}" core_correction="true" '
            f'wfc_cutoff="{wfc_cutoff}" rho_cutoff="{rho_cutoff}"/>\n'
            '</UPF>\n'
        ),
    )


def test_parse_pw_input_normalizes_observed_fortran_and_card_forms(tmp_path):
    parsed = parse_pw_input(_write(tmp_path, "scf.in", SCF_INPUT))

    assert parsed["format"] == "qe-pw-input/1"
    assert parsed["calculation"] == "scf"
    assert parsed["system"] == {
        "ibrav": 2,
        "nat": 2,
        "ntyp": 1,
        "ecutwfc_ry": 18.0,
        "ecutrho_ry": None,
        "occupations": None,
        "smearing": None,
        "degauss_ry": None,
        "nspin": 1,
        "starting_magnetization": {},
    }
    assert parsed["atomic_positions"]["units"] == "alat"
    assert parsed["k_points"] == {
        "option": "automatic",
        "grid": [4, 4, 4],
        "shift": [1, 1, 1],
    }


def test_parse_ph_x_text_captures_single_q_input():
    assert parse_ph_x_text(PHONON_INPUT) == {
        "format": "qe-ph-input/1",
        "title": "Si single-q phonon",
        "inputph_line": 2,
        "inputph_closed": True,
        "namelist": {"prefix": "si", "outdir": "./tmp"},
        "q_point": [0.25, 0.0, 0.0],
        "q_point_line": 5,
    }
    assert lint_ph_x_input(PHONON_INPUT) == []


def test_review_input_recognizes_supported_single_q_phonon_input(tmp_path):
    path = _write(tmp_path, "phonon.in", PHONON_INPUT)
    backends = tuple(load_backend(spec) for spec in BUILTIN_BACKENDS)

    backend, resolved_by = detect_input_backend(backends, path)
    reviewed = review_input(backend, path, resolved_by=resolved_by)

    assert backend is QE
    assert resolved_by == "content"
    assert reviewed["assessment"]["verdict"]["label"] == "checks_passed"
    assert reviewed["evidence"]["parser"]["result"]["format"] == "qe-ph-input/1"
    assert reviewed["evidence"]["lint"]["issues"] == []


def test_review_input_recognizes_bounded_pw2qmcpack_input(tmp_path):
    path = _write(tmp_path, "p2q.in", PW2QMCPACK_INPUT)
    backends = tuple(load_backend(spec) for spec in BUILTIN_BACKENDS)

    assert is_pw2qmcpack_input(PW2QMCPACK_INPUT) is True
    assert parse_pw2qmcpack_text(PW2QMCPACK_INPUT) == {
        "format": "qe-pw2qmcpack-input/1",
        "inputpp_line": 1,
        "inputpp_closed": True,
        "namelist": {
            "prefix": "si",
            "outdir": "./tmp",
            "write_psir": False,
        },
    }
    assert lint_pw2qmcpack_input(PW2QMCPACK_INPUT) == []

    backend, resolved_by = detect_input_backend(backends, path)
    reviewed = review_input(backend, path, resolved_by=resolved_by)

    assert backend is QE
    assert resolved_by == "content"
    assert reviewed["assessment"]["verdict"]["label"] == "checks_passed"
    assert reviewed["evidence"]["parser"]["result"]["format"] == "qe-pw2qmcpack-input/1"
    assert reviewed["evidence"]["lint"]["issues"] == []


def test_pw2qmcpack_review_leaves_other_inputpp_forms_unsupported():
    expanded = PW2QMCPACK_INPUT.replace(
        "write_psir = .false.",
        "write_psir = .false., filplot = 'rho.dat'",
    )

    assert is_pw2qmcpack_input(expanded) is False
    assert QE.inputs.lint_input(expanded) == lint_pw_input(expanded)


def test_pw2qmcpack_review_explains_missing_explicit_handoff_paths():
    incomplete = "&inputpp\nwrite_psir = .false.\n/\n"

    assert is_pw2qmcpack_input(incomplete) is True
    assert QE.inputs.lint_input(incomplete) == [{
        "level": "warning",
        "message": (
            "&INPUTPP prefix is not explicit, so Chemtools cannot confirm "
            "the preceding QE calculation."
        ),
        "line": 1,
        "suggested_fix": None,
    }, {
        "level": "warning",
        "message": (
            "&INPUTPP outdir is not explicit, so Chemtools cannot confirm "
            "the preceding QE calculation."
        ),
        "line": 1,
        "suggested_fix": None,
    }]


def test_lint_ph_x_input_flags_gamma_default_and_invalid_epsil_scope():
    gamma = PHONON_INPUT.replace("0.25 0.0 0.0", "0.0 0.0 0.0")
    assert lint_ph_x_input(gamma) == [{
        "level": "warning",
        "message": (
            "At q=0, decide whether epsil is needed for the non-analytic "
            "LO-TO term before running ph.x."
        ),
        "line": 5,
        "suggested_fix": None,
    }]

    finite_q_epsil = PHONON_INPUT.replace(
        " prefix = 'si', outdir = './tmp'",
        " prefix = 'si', outdir = './tmp', epsil = .true.",
    )
    assert lint_ph_x_input(finite_q_epsil) == [{
        "level": "error",
        "message": "epsil=.true. is only supported at q=0 in the documented ph.x scope.",
        "line": 5,
        "suggested_fix": None,
    }]


def test_parse_pw_input_evaluates_documented_coordinate_expressions(tmp_path):
    expression_input = SCF_INPUT.replace(
        "Si 0.25 0.25 0.25",
        "Si 1/3 1/2*3^(-1/2) 1.d-1",
    )

    parsed = parse_pw_input(_write(tmp_path, "expressions.in", expression_input))

    coordinates = parsed["atomic_positions"]["atoms"][1]["coordinates"]
    assert coordinates[0] == 1.0 / 3.0
    assert math.isclose(
        coordinates[1],
        1.0 / (2.0 * math.sqrt(3.0)),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert coordinates[2] == 0.1
    assert lint_pw_input(expression_input) == []


def test_parse_pw_input_rejects_unsupported_coordinate_expressions(tmp_path):
    invalid_input = SCF_INPUT.replace(
        "Si 0.25 0.25 0.25",
        "Si __import__('os') 0.0 2^2048",
    )

    parsed = parse_pw_input(_write(tmp_path, "invalid-expression.in", invalid_input))

    assert len(parsed["atomic_positions"]["atoms"]) == 1
    assert [issue["message"] for issue in lint_pw_input(invalid_input)] == [
        "Every ATOMIC_POSITIONS row must contain a label and three numeric coordinates.",
        "nat=2 but ATOMIC_POSITIONS contains 1 valid row(s).",
    ]


def test_lint_pw_input_accepts_crystal_sg_syntax_without_numeric_coordinates():
    crystal_sg_input = """\\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, nat = 2, ntyp = 1, space_group = 225, ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
CELL_PARAMETERS angstrom
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
ATOMIC_POSITIONS crystal_sg
Si 8c 0.25
K_POINTS gamma
"""

    issues = lint_pw_input(crystal_sg_input)

    assert issues == [{
        "level": "info",
        "message": (
            "ATOMIC_POSITIONS crystal_sg uses symmetry expansion; "
            "coordinate and atom-count checks are not available."
        ),
        "line": 16,
        "suggested_fix": None,
    }]


@pytest.mark.parametrize("space_group", ("", "space_group = 0, "))
def test_lint_pw_input_requires_space_group_for_crystal_sg(space_group: str):
    crystal_sg_input = f"""\\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, nat = 2, ntyp = 1, {space_group}ecutwfc = 50
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.UPF
CELL_PARAMETERS angstrom
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
ATOMIC_POSITIONS crystal_sg
Si 8c 0.25
K_POINTS gamma
"""

    issues = lint_pw_input(crystal_sg_input)

    assert [issue["message"] for issue in issues] == [
        "ATOMIC_POSITIONS crystal_sg requires a positive space_group number."
    ]


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    (
        (
            "CELL_PARAMETERS angstrom",
            "CELL_PARAMETERS fractional",
            (
                "CELL_PARAMETERS option 'fractional' is not recognized; "
                "use alat, bohr, or angstrom."
            ),
        ),
        (
            "ATOMIC_POSITIONS angstrom",
            "ATOMIC_POSITIONS cartesian",
            (
                "ATOMIC_POSITIONS option 'cartesian' is not recognized; "
                "use alat, bohr, angstrom, crystal, or crystal_sg."
            ),
        ),
    ),
)
def test_lint_pw_input_rejects_unsupported_coordinate_card_options(
    original: str,
    replacement: str,
    message: str,
):
    issues = lint_pw_input(VC_RELAX_INPUT.replace(original, replacement))

    assert [issue["message"] for issue in issues] == [message]


@pytest.mark.parametrize("nonfinite", ("nan", "1e999"))
def test_lint_pw_input_rejects_nonfinite_real_values(nonfinite: str):
    issues = lint_pw_input(SCF_INPUT.replace("1.8d1", nonfinite))

    assert [issue["message"] for issue in issues] == [
        "&SYSTEM requires a positive ecutwfc value in Ry."
    ]


@pytest.mark.parametrize(
    "constraints",
    ("1 0", "1 0 2"),
)
def test_lint_pw_input_rejects_invalid_position_constraints(constraints: str):
    issues = lint_pw_input(VC_RELAX_INPUT.replace(
        "Fe 0.0 0.0 0.0",
        f"Fe 0.0 0.0 0.0 {constraints}",
    ))

    assert [issue["message"] for issue in issues] == [
        "ATOMIC_POSITIONS constraints must be three values, each equal to 0 or 1."
    ]


def test_parse_pw_input_preserves_spin_and_relaxation_namelists(tmp_path):
    parsed = parse_pw_input(
        _write(tmp_path, "feo.vc-relax.in", VC_RELAX_INPUT)
    )

    assert parsed["calculation"] == "vc-relax"
    assert parsed["system"]["starting_magnetization"] == {"1": 0.8}
    assert parsed["namelists"]["ions"] == {"ion_dynamics": "bfgs"}
    assert parsed["namelists"]["cell"] == {"cell_dynamics": "bfgs"}
    assert parsed["cell_parameters"] == {
        "units": "angstrom",
        "vectors": [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
    }


def test_lint_pw_input_accepts_supported_scf_and_vc_relax_examples():
    assert lint_pw_input(SCF_INPUT) == []
    assert lint_pw_input(VC_RELAX_INPUT) == []


def test_lint_pw_input_reports_structural_cross_reference_errors():
    broken = """\
&CONTROL
 calculation = 'scf'
/
&SYSTEM
 ibrav = 0, nat = 2, ntyp = 2, ecutwfc = 0,
 occupations = 'smearing'
/
&ELECTRONS
/
ATOMIC_SPECIES
 Fe 55.845 Fe.UPF
ATOMIC_POSITIONS angstrom
 X 0.0 0.0 0.0
K_POINTS automatic
 4 4 0 0 0 2
"""

    issues = lint_pw_input(broken)
    assert [issue["message"] for issue in issues] == [
        "&SYSTEM requires a positive ecutwfc value in Ry.",
        "ntyp=2 but ATOMIC_SPECIES contains 1 valid row(s).",
        "nat=2 but ATOMIC_POSITIONS contains 1 valid row(s).",
        "ATOMIC_POSITIONS uses labels absent from ATOMIC_SPECIES: ['X'].",
        "ibrav=0 requires a CELL_PARAMETERS card.",
        "The three automatic k-point grid dimensions must be positive.",
        "The three automatic k-point shifts must each be 0 or 1.",
        "occupations='smearing' requires the smearing method.",
        "occupations='smearing' requires a positive degauss value in Ry.",
    ]
    assert all(issue["level"] == "error" for issue in issues)


def test_lint_pw_input_marks_other_pw_calculations_as_limited_review():
    bands = SCF_INPUT.replace("'scf'", "'bands'", 1)

    assert lint_pw_input(bands) == [{
        "level": "warning",
        "message": (
            "calculation='bands' parses, but the current Chemtools review "
            "covers only scf, relax, and vc-relax semantics."
        ),
        "line": 2,
        "suggested_fix": None,
    }]


@pytest.mark.parametrize(("namelist", "program"), [
    ("BANDS", "bands.x"),
    ("DOS", "dos.x"),
    ("INPUTPP", "pp.x"),
    ("PROJWFC", "projwfc.x"),
])
def test_lint_pw_input_identifies_unsupported_qe_programs(
    namelist: str,
    program: str,
):
    text = f"&{namelist}\n/\n"

    assert unsupported_qe_program(text) == program
    assert lint_pw_input(text) == [{
        "level": "error",
        "message": (
            f"This is a Quantum ESPRESSO {program} input; the current Chemtools "
            "QE reviewer supports pw.x inputs only."
        ),
        "line": None,
        "suggested_fix": None,
    }]
    assert QE.inputs.lint_input(text) == lint_pw_input(text)


@pytest.mark.parametrize(("namelist", "program"), [
    ("BANDS", "bands.x"),
    ("DOS", "dos.x"),
    ("INPUTPP", "pp.x"),
    ("PROJWFC", "projwfc.x"),
])
def test_guided_review_detects_unsupported_qe_companion_inputs(
    tmp_path: Path,
    namelist: str,
    program: str,
):
    path = _write(tmp_path, "companion.in", f"&{namelist}\n/\n")
    backends = tuple(load_backend(spec) for spec in BUILTIN_BACKENDS)

    backend, resolved_by = detect_input_backend(backends, path)
    reviewed = review_input(backend, path, resolved_by=resolved_by)

    assert backend is QE
    assert resolved_by == "content"
    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "error",
        "message": (
            f"This is a Quantum ESPRESSO {program} input; the current "
            "Chemtools QE reviewer supports pw.x inputs only."
        ),
        "line": None,
        "suggested_fix": None,
    }]


def test_qe_backend_drives_generic_guided_review(tmp_path):
    path = _write(tmp_path, "scf.in", SCF_INPUT)
    _write_upf(tmp_path)
    backends = tuple(load_backend(spec) for spec in BUILTIN_BACKENDS)

    backend, resolved_by = detect_input_backend(backends, path)
    reviewed = review_input(backend, path, resolved_by=resolved_by)

    assert backend is QE
    assert resolved_by == "content"
    assert reviewed["program"] == {"name": "qe", "resolved_by": "content"}
    assert reviewed["assessment"]["verdict"]["label"] == "checks_passed"
    assert reviewed["evidence"]["parser"]["result"]["calculation"] == "scf"
    pseudo_review = reviewed["evidence"]["parser"]["result"][
        "pseudopotential_review"
    ]
    assert pseudo_review["status"] == "parsed"
    assert pseudo_review["resolution"]["basis"] == "input_directory_assumption"
    assert pseudo_review["cutoff_review"] == {
        "ecutwfc_ry": 18.0,
        "effective_ecutrho_ry": 72.0,
        "ecutrho_source": "qe_default_4x",
        "suggested_ecutwfc_ry": 16.0,
        "suggested_ecutrho_ry": 70.0,
        "suggested_ecutwfc_source": {
            "value_ry": 16.0,
            "element": "Si",
            "path": str((tmp_path / "Si.pbe.UPF").resolve()),
        },
        "suggested_ecutrho_source": {
            "value_ry": 70.0,
            "element": "Si",
            "path": str((tmp_path / "Si.pbe.UPF").resolve()),
        },
        "wavefunction_status": "meets_suggestion",
        "density_status": "meets_suggestion",
        "convergence_established": False,
    }
    assert reviewed["evidence"]["parser"]["result"][
        "charge_spin_review"
    ]["electron_accounting"] == {
        "status": "complete",
        "basis": "UPF PP_HEADER z_valence summed over ATOMIC_POSITIONS",
        "valence_electrons_before_charge": 8.0,
        "tot_charge": 0.0,
        "electron_count": 8.0,
        "missing_species": [],
    }
    k_point_review = reviewed["evidence"]["parser"]["result"][
        "k_point_review"
    ]
    assert k_point_review["sampling"] == {
        "mode": "mesh",
        "option": "automatic",
        "mesh": [4, 4, 4],
        "shift": [1, 1, 1],
        "requested_full_grid_points": 64,
    }
    assert k_point_review["convergence_plan"]["candidate_meshes"] == [
        {
            "stage": "current",
            "mesh": [4, 4, 4],
            "shift": [1, 1, 1],
            "requested_full_grid_points": 64,
        },
        {
            "stage": "refine_1",
            "mesh": [6, 6, 6],
            "shift": [1, 1, 1],
            "requested_full_grid_points": 216,
        },
        {
            "stage": "refine_2",
            "mesh": [8, 8, 8],
            "shift": [1, 1, 1],
            "requested_full_grid_points": 512,
        },
    ]
    assert k_point_review["convergence_plan"]["convergence_established"] is False
    assert reviewed["evidence"]["lint"]["issues"] == []
    assert reviewed["uncertainty"] == []


def test_parse_upf_header_returns_trusted_metadata(tmp_path):
    path = _write_upf(
        tmp_path,
        element="Fe",
        pseudo_type="USPP",
        wfc_cutoff=71.0,
        rho_cutoff=496.0,
    )

    assert parse_upf_header(path) == {
        "schema_version": "chemtools.qe-upf-header/1",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "upf_version": "2.0.1",
        "element": "Fe",
        "pseudo_type": "USPP",
        "relativistic": "scalar",
        "functional": "PBE",
        "z_valence": 4.0,
        "local_channel": None,
        "projector_count": None,
        "projector_channel_evidence": {
            "status": "not_available",
            "declared_total": None,
            "observed_total": 0,
            "invalid_angular_momentum_count": 0,
            "counts_by_angular_momentum": {},
            "declared_total_matches_observed": None,
        },
        "suggested_ecutwfc_ry": 71.0,
        "suggested_ecutrho_ry": 496.0,
        "is_ultrasoft": True,
        "is_paw": True,
        "has_spin_orbit": False,
        "core_correction": True,
    }


def test_parse_upf_header_preserves_projector_metadata_without_interpreting_it(tmp_path):
    path = _write_upf(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'rho_cutoff="70.0"',
            'rho_cutoff="70.0" l_local="2" number_of_proj="4"',
        ),
        encoding="utf-8",
    )

    parsed = parse_upf_header(path)

    assert parsed["local_channel"] == 2
    assert parsed["projector_count"] == 4


def test_parse_upf_header_counts_declared_projectors_by_angular_momentum(tmp_path):
    path = _write(tmp_path, "Fe.UPF", """\
<UPF version="2.0.1">
  <PP_HEADER element="Fe" pseudo_type="NC" z_valence="8" number_of_proj="3"/>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0">0</PP_BETA.1>
    <PP_BETA.2 index="2" angular_momentum="2">0</PP_BETA.2>
    <PP_BETA.3 index="3" angular_momentum="2">0</PP_BETA.3>
  </PP_NONLOCAL>
</UPF>
""")

    parsed = parse_upf_header(path)

    assert parsed["projector_channel_evidence"] == {
        "status": "complete",
        "declared_total": 3,
        "observed_total": 3,
        "invalid_angular_momentum_count": 0,
        "counts_by_angular_momentum": {"0": 1, "2": 2},
        "declared_total_matches_observed": True,
    }


def test_parse_upf_header_marks_incomplete_projector_evidence(tmp_path):
    path = _write(tmp_path, "Fe.UPF", """\
<UPF version="2.0.1">
  <PP_HEADER element="Fe" pseudo_type="NC" z_valence="8" number_of_proj="2"/>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="d">0</PP_BETA.1>
  </PP_NONLOCAL>
</UPF>
""")

    parsed = parse_upf_header(path)

    assert parsed["projector_channel_evidence"] == {
        "status": "partial",
        "declared_total": 2,
        "observed_total": 1,
        "invalid_angular_momentum_count": 1,
        "counts_by_angular_momentum": {},
        "declared_total_matches_observed": False,
    }


def test_parse_upf_header_does_not_treat_zero_as_a_cutoff_suggestion(tmp_path):
    path = _write_upf(tmp_path, wfc_cutoff=0.0, rho_cutoff=0.0)

    parsed = parse_upf_header(path)

    assert parsed["suggested_ecutwfc_ry"] is None
    assert parsed["suggested_ecutrho_ry"] is None


def test_qe_guided_review_warns_below_upf_cutoff_suggestions(tmp_path):
    path = _write(tmp_path, "scf.in", SCF_INPUT)
    _write_upf(tmp_path, wfc_cutoff=44.0, rho_cutoff=175.0)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"] == {
        "label": "review_required",
        "confidence": 0.8,
        "reasons": ["The configured linter found 2 warning(s)."],
    }
    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == [
        "ecutwfc=18 Ry is below the hardest positive UPF suggestion of 44 Ry.",
        (
            "The qe_default_4x ecutrho=72 Ry is below the hardest positive "
            "UPF suggestion of 175 Ry."
        ),
    ]
    assert reviewed["evidence"]["parser"]["result"][
        "pseudopotential_review"
    ]["cutoff_review"]["convergence_established"] is False


def test_qe_guided_review_rejects_upf_element_mismatch(tmp_path):
    path = _write(tmp_path, "scf.in", SCF_INPUT)
    _write_upf(tmp_path, element="S")

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "error",
        "message": (
            "Species label 'Si' references a UPF whose PP_HEADER element is 'S'."
        ),
        "line": 12,
        "suggested_fix": "Use a pseudopotential generated for this species.",
    }]


def test_qe_guided_review_reports_unresolved_runtime_default(tmp_path):
    input_without_pseudo_dir = SCF_INPUT.replace(
        ", pseudo_dir = '.'",
        "",
        1,
    )
    path = _write(tmp_path, "scf.in", input_without_pseudo_dir)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "review_required"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "warning",
        "message": (
            "pseudo_dir is not explicit, so referenced pseudopotentials "
            "could not be inspected before execution."
        ),
        "line": None,
        "suggested_fix": (
            "Set pseudo_dir to the directory containing the UPF files."
        ),
    }]


def test_qe_guided_review_rejects_explicit_nspin_with_noncolin(tmp_path):
    noncollinear = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        (
            "ecutwfc = 1.8d1, nspin = 2, noncolin = .true., "
            "starting_magnetization(1) = 0.5,"
        ),
        1,
    )
    path = _write(tmp_path, "noncollinear.in", noncollinear)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == ["noncolin=.true. must be used without an explicit nspin value."]
    assert reviewed["evidence"]["lint"]["issues"][0]["line"] == 6


def test_qe_guided_review_rejects_fixed_and_starting_magnetization(tmp_path):
    contradictory = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        (
            "ecutwfc = 1.8d1, nspin = 2, tot_magnetization = 2, "
            "starting_magnetization(1) = 0.5,"
        ),
        1,
    )
    path = _write(tmp_path, "contradictory.in", contradictory)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == [
        "tot_magnetization and starting_magnetization must not be specified together."
    ]


def test_qe_guided_review_warns_when_collinear_spin_has_no_seed(tmp_path):
    unseeded = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        "ecutwfc = 1.8d1, nspin = 2,",
        1,
    )
    path = _write(tmp_path, "unseeded.in", unseeded)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "review_required"
    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == [
        "nspin=2 has no nonzero starting_magnetization or fixed tot_magnetization."
    ]


def test_qe_guided_review_checks_species_index_bounds(tmp_path):
    invalid_index = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        "ecutwfc = 1.8d1, nspin = 2, starting_magnetization(2) = 0.5,",
        1,
    )
    path = _write(tmp_path, "invalid-index.in", invalid_index)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == [
        "starting_magnetization species indices [2] fall outside 1..ntyp (1)."
    ]
    assert reviewed["evidence"]["lint"]["issues"][0]["line"] == 6


def test_qe_guided_review_checks_spin_orbit_upf_metadata(tmp_path):
    spin_orbit = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        (
            "ecutwfc = 1.8d1, noncolin = .true., lspinorb = .true., "
            "starting_magnetization(1) = 0.5,"
        ),
        1,
    )
    path = _write(tmp_path, "spin-orbit.in", spin_orbit)
    _write_upf(tmp_path, has_spin_orbit=False)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "review_required"
    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == [
        "lspinorb=.true., but none of the inspected UPFs advertises spin-orbit data."
    ]


def test_qe_guided_review_reports_nonmagnetic_spin_orbit_as_info(tmp_path):
    spin_orbit = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        (
            "ecutwfc = 1.8d1, noncolin = .true., lspinorb = .true., "
            "starting_magnetization(1) = 0.0,"
        ),
        1,
    )
    path = _write(tmp_path, "nonmagnetic-spin-orbit.in", spin_orbit)
    _write_upf(tmp_path, has_spin_orbit=True)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "checks_passed"
    assert reviewed["evidence"]["lint"]["summary"] == {
        "errors": 0,
        "warnings": 0,
        "info": 1,
    }
    assert reviewed["evidence"]["lint"]["issues"][0]["message"] == (
        "The spin-orbit calculation has no nonzero magnetic seed or constraint, "
        "so it retains time-reversal symmetry and zero magnetization."
    )


def test_qe_guided_review_warns_on_zero_valence_electron_count(tmp_path):
    overcharged = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        "ecutwfc = 1.8d1, tot_charge = 8,",
        1,
    )
    path = _write(tmp_path, "overcharged.in", overcharged)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["evidence"]["parser"]["result"][
        "charge_spin_review"
    ]["electron_accounting"]["electron_count"] == 0.0
    assert reviewed["assessment"]["verdict"]["label"] == "review_required"
    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == ["UPF valence accounting gives zero electrons after tot_charge."]


def test_qe_guided_review_rejects_negative_valence_electron_count(tmp_path):
    overcharged = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        "ecutwfc = 1.8d1, tot_charge = 9,",
        1,
    )
    path = _write(tmp_path, "negative-electron-count.in", overcharged)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert [
        issue["message"] for issue in reviewed["evidence"]["lint"]["issues"]
    ] == ["UPF valence accounting gives -1 electrons after tot_charge."]


def test_qe_guided_review_requires_automatic_grid_for_tetrahedra(tmp_path):
    tetrahedra_gamma = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        "ecutwfc = 1.8d1, occupations = 'tetrahedra',",
        1,
    ).replace(
        "K_POINTS {automatic}\n 4 4 4 1 1 1",
        "K_POINTS gamma",
        1,
    )
    path = _write(tmp_path, "tetrahedra-gamma.in", tetrahedra_gamma)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "errors_found"
    assert reviewed["evidence"]["lint"]["issues"] == [{
        "level": "error",
        "message": "occupations='tetrahedra' requires K_POINTS automatic.",
        "line": 16,
        "suggested_fix": (
            "Use an automatically generated uniform grid, or choose an "
            "occupation method compatible with the intended sampling."
        ),
    }]
    assert reviewed["evidence"]["parser"]["result"][
        "k_point_review"
    ]["convergence_plan"]["status"] == "sampling_design_required"


def test_qe_guided_review_flags_offset_tetrahedron_grid(tmp_path):
    tetrahedra_shifted = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        "ecutwfc = 1.8d1, occupations = 'tetrahedra',",
        1,
    )
    path = _write(tmp_path, "tetrahedra-shifted.in", tetrahedra_shifted)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")

    assert reviewed["assessment"]["verdict"]["label"] == "review_required"
    assert reviewed["evidence"]["lint"]["issues"][0]["message"] == (
        "The tetrahedron calculation uses an offset automatic grid; QE warns "
        "that some offset grids lack the full crystal symmetry."
    )


def test_lint_pw_input_checks_explicit_band_path_shape(tmp_path):
    band_path = SCF_INPUT.replace("'scf'", "'bands'", 1).replace(
        "K_POINTS {automatic}\n 4 4 4 1 1 1",
        (
            "K_POINTS crystal_b\n"
            " 3\n"
            " 0.0 0.0 0.0 10\n"
            " 0.5 0.0 0.0 10\n"
            " 0.5 0.5 0.0 1"
        ),
        1,
    )

    issues = lint_pw_input(band_path)

    assert [issue["message"] for issue in issues] == [
        (
            "calculation='bands' parses, but the current Chemtools review "
            "covers only scf, relax, and vc-relax semantics."
        )
    ]
    parsed = parse_pw_input(_write(tmp_path, "bands.in", band_path))
    assert parsed["k_points"] == {
        "option": "crystal_b",
        "declared_count": 3,
        "points": [
            {"coordinates": [0.0, 0.0, 0.0], "weight": 10.0, "line": 18},
            {"coordinates": [0.5, 0.0, 0.0], "weight": 10.0, "line": 19},
            {"coordinates": [0.5, 0.5, 0.0], "weight": 1.0, "line": 20},
        ],
    }


def test_lint_pw_input_rejects_band_path_for_scf():
    band_path = SCF_INPUT.replace(
        "K_POINTS {automatic}\n 4 4 4 1 1 1",
        "K_POINTS crystal_b\n 2\n 0.0 0.0 0.0 10\n 0.5 0.0 0.0 1",
        1,
    )

    assert [issue["message"] for issue in lint_pw_input(band_path)] == [
        "K_POINTS crystal_b defines a band path but calculation='scf'."
    ]


def test_lint_pw_input_checks_declared_explicit_point_count():
    incomplete_path = SCF_INPUT.replace("'scf'", "'bands'", 1).replace(
        "K_POINTS {automatic}\n 4 4 4 1 1 1",
        "K_POINTS crystal_b\n 3\n 0.0 0.0 0.0 10\n 0.5 0.0 0.0 1",
        1,
    )

    assert [issue["message"] for issue in lint_pw_input(incomplete_path)] == [
        (
            "calculation='bands' parses, but the current Chemtools review "
            "covers only scf, relax, and vc-relax semantics."
        ),
        "K_POINTS crystal_b declares 3 point(s) but contains 2 row(s).",
    ]


def test_qe_k_point_plan_preserves_single_point_axis(tmp_path):
    slab_mesh = SCF_INPUT.replace(
        "4 4 4 1 1 1",
        "12 12 1 0 0 0",
        1,
    )
    path = _write(tmp_path, "slab.in", slab_mesh)
    _write_upf(tmp_path)

    parsed = QE.parser.parse_input(str(path))
    candidates = parsed["k_point_review"]["convergence_plan"][
        "candidate_meshes"
    ]

    assert [candidate["mesh"] for candidate in candidates] == [
        [12, 12, 1],
        [16, 16, 1],
        [18, 18, 1],
    ]


def test_qe_k_point_review_records_symmetry_expansion_flags(tmp_path):
    no_symmetry = SCF_INPUT.replace(
        "ecutwfc = 1.8d1,",
        "ecutwfc = 1.8d1, nosym = .true., noinv = .true.,",
        1,
    )
    path = _write(tmp_path, "no-symmetry.in", no_symmetry)
    _write_upf(tmp_path)

    reviewed = review_input(QE, path, resolved_by="content")
    symmetry = reviewed["evidence"]["parser"]["result"][
        "k_point_review"
    ]["symmetry"]

    assert symmetry["flags"] == {
        "nosym": True,
        "nosym_evc": False,
        "noinv": True,
        "no_t_rev": False,
        "force_symmorphic": False,
    }
    assert symmetry["effects"] == [
        "Uniform grids expand over the full Brillouin zone.",
        "k and -k are not treated as equivalent during generation.",
    ]
    assert symmetry["irreducible_k_point_count"] is None


def test_qe_detector_requires_pwscf_output_banner():
    banner = "     Program PWSCF v.7.5 starts on  3Dec2025 at 13:45:35\n"

    assert QE.detector.detect(banner) is True
    assert QE.detector.detect_version(banner) == "7.5"
    assert QE.detector.detect("title: PWSCF comparison\n") is False
