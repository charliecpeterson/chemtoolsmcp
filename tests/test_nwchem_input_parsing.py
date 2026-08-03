r"""Regression tests for NWChem fragment-guess input parsing, the lint checks
built on it, and the launch resource-sanity warnings.

These cover the parsing bugs where a `vectors input fragment` line had its
`output` keyword, output filename, inline comment, or `\` line-continuation
marker captured as fragment movecs files, plus the two preflight warnings.
"""

from chemtools.programs.nwchem.parse.input import (
    _split_vectors_fragment_files,
    parse_start_blocks,
    parse_task_states,
)
from chemtools.programs.nwchem.input.lint_restart import lint_nwchem_input
from chemtools.programs.nwchem.runner import _resource_warnings


def _write(tmp_path, text):
    p = tmp_path / "job.nw"
    p.write_text(text)
    return str(p)


def test_split_stops_at_output_keyword():
    assert _split_vectors_fragment_files(
        "fe.mos o.mos output full.movecs"
    ) == ["fe.mos", "o.mos"]


def test_split_strips_inline_comment():
    assert _split_vectors_fragment_files(
        "a.mos b.mos output c.movecs #same as title"
    ) == ["a.mos", "b.mos"]


def test_split_stops_at_swap():
    assert _split_vectors_fragment_files("a.mos b.mos swap 5 6") == ["a.mos", "b.mos"]


def test_split_drops_continuation_marker():
    assert _split_vectors_fragment_files("a.mos b.mos \\") == ["a.mos", "b.mos"]


def test_split_plain_file_list():
    assert _split_vectors_fragment_files("a.mos b.mos c.mos") == [
        "a.mos",
        "b.mos",
        "c.mos",
    ]


_INLINE_FRAGMENT_JOB = """start mol
geometry frag1
  Fe 0.0 0.0 0.0
end
geometry full
  Fe 0.0 0.0 0.0
  O  0.0 0.0 1.6
end
basis
  Fe library def2-svp
  O  library def2-svp
end
set geometry frag1
charge 0
dft
  mult 5
  vectors input atomic output fe.mos
end
task dft energy
dft
  mult 3
  vectors input atomic output o.mos
end
task dft energy
set geometry full
charge 0
dft
  mult 5
  vectors input fragment fe.mos o.mos output full.movecs
end
task dft energy
"""


def test_parse_captures_inline_vectors_output(tmp_path):
    blocks = parse_start_blocks(_write(tmp_path, _INLINE_FRAGMENT_JOB))
    assert len(blocks) == 1
    # All three `output X` clauses are captured, including the inline ones.
    assert set(blocks[0]["vectors_outputs"]) == {"fe.mos", "o.mos", "full.movecs"}
    # Fragment list is exactly the two fragment files, no output/keyword tokens.
    assert blocks[0]["fragment_inputs"] == ["fe.mos", "o.mos"]


def test_parse_joins_line_continuation(tmp_path):
    job = _INLINE_FRAGMENT_JOB.replace(
        "  vectors input fragment fe.mos o.mos output full.movecs",
        "  vectors input fragment fe.mos o.mos \\\n          output full.movecs",
    )
    blocks = parse_start_blocks(_write(tmp_path, job))
    # The `\` continuation must not leak into the fragment list.
    assert blocks[0]["fragment_inputs"] == ["fe.mos", "o.mos"]
    assert "full.movecs" in blocks[0]["vectors_outputs"]


def test_task_states_follow_charge_spin_and_named_geometry(tmp_path):
    states = parse_task_states(_write(tmp_path, _INLINE_FRAGMENT_JOB))

    assert states == [
        {
            "task_index": 0,
            "module": "dft",
            "operation": "energy",
            "charge": 0,
            "charge_source": "explicit",
            "multiplicity": 5,
            "multiplicity_source": "mult",
            "reference": {
                "kind": "open_shell",
                "class": "open_shell",
                "source": "default",
            },
            "basis": {
                "name": "ao basis",
                "mode": "cartesian",
                "mode_source": "default",
                "source": "input",
            },
            "ecp": {
                "source": "none",
                "core_electrons": {},
                "library_elements": [],
                "default_library": False,
            },
            "geometry": {
                "name": "frag1",
                "block_index": 0,
                "source": "input",
            },
        },
        {
            "task_index": 1,
            "module": "dft",
            "operation": "energy",
            "charge": 0,
            "charge_source": "explicit",
            "multiplicity": 3,
            "multiplicity_source": "mult",
            "reference": {
                "kind": "open_shell",
                "class": "open_shell",
                "source": "default",
            },
            "basis": {
                "name": "ao basis",
                "mode": "cartesian",
                "mode_source": "default",
                "source": "input",
            },
            "ecp": {
                "source": "none",
                "core_electrons": {},
                "library_elements": [],
                "default_library": False,
            },
            "geometry": {
                "name": "frag1",
                "block_index": 0,
                "source": "input",
            },
        },
        {
            "task_index": 2,
            "module": "dft",
            "operation": "energy",
            "charge": 0,
            "charge_source": "explicit",
            "multiplicity": 5,
            "multiplicity_source": "mult",
            "reference": {
                "kind": "open_shell",
                "class": "open_shell",
                "source": "default",
            },
            "basis": {
                "name": "ao basis",
                "mode": "cartesian",
                "mode_source": "default",
                "source": "input",
            },
            "ecp": {
                "source": "none",
                "core_electrons": {},
                "library_elements": [],
                "default_library": False,
            },
            "geometry": {
                "name": "full",
                "block_index": 1,
                "source": "input",
            },
        },
    ]


def test_task_states_mark_geometry_produced_by_prior_optimization(tmp_path):
    path = _write(
        tmp_path,
        (
            "geometry units angstrom\n"
            "  H 0 0 0\n"
            "  H 0 0 0.74\n"
            "end\n"
            "scf; singlet; end\n"
            "task scf optimize\n"
            "task scf freq\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["geometry"] == {
        "name": "geometry",
        "block_index": 0,
        "source": "input",
    }
    assert states[1]["geometry"] == {
        "name": "geometry",
        "block_index": 0,
        "source": "task_result",
        "source_task_index": 0,
    }
    assert states[0]["charge"] == 0
    assert states[0]["charge_source"] == "default"
    assert states[0]["multiplicity"] == 1


def test_task_states_track_explicit_and_library_ecps(tmp_path):
    path = _write(
        tmp_path,
        (
            "start th\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "  F 0 0 2\n"
            "end\n"
            "ecp\n"
            "  Th nelec 60\n"
            "  F library cc-pvdz-pp\n"
            "end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["ecp"] == {
        "source": "explicit",
        "core_electrons": {"Th": 60},
        "library_elements": ["F"],
        "library_assignments": {"F": "cc-pvdz-pp"},
        "default_library": False,
    }


def test_task_states_track_default_and_external_ecp_library(tmp_path):
    path = _write(
        tmp_path,
        (
            "start th\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            "ecp\n"
            "  * library Th stuttgart_rsc_1997_ecp file custom.lib\n"
            "end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["ecp"] == {
        "source": "explicit",
        "core_electrons": {},
        "library_elements": [],
        "default_library": True,
        "default_library_name": "stuttgart_rsc_1997_ecp",
        "uses_external_library_file": True,
    }


def test_task_states_track_basis_mode_and_named_basis_alias(tmp_path):
    path = _write(
        tmp_path,
        (
            "start water\n"
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "end\n"
            'basis "large basis" spherical\n'
            "  O library cc-pvdz\n"
            "end\n"
            'set "ao basis" "large basis"\n'
            "task dft energy\n"
            "basis\n"
            "  O library 6-31g*\n"
            "end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["basis"] == {
        "name": "large basis",
        "mode": "spherical",
        "mode_source": "explicit",
        "source": "input",
    }
    assert states[1]["basis"] == {
        "name": "ao basis",
        "mode": "cartesian",
        "mode_source": "default",
        "source": "input",
    }


def test_task_states_track_named_and_component_xc_state(tmp_path):
    path = _write(
        tmp_path,
        (
            "start water\n"
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "end\n"
            "dft; xc pbe0; end\n"
            "task dft energy\n"
            "dft; mult 3; end\n"
            "task dft energy\n"
            "dft; xc B3LYP; end\n"
            "task dft energy\n"
            "dft; xc xpbe96 cpbe96; end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert [state["xc"] for state in states] == [
        {
            "name": "pbe0",
            "tokens": ["pbe0"],
            "source": "explicit_alias",
        },
        {
            "name": "pbe0",
            "tokens": ["pbe0"],
            "source": "explicit_alias",
        },
        {
            "name": "b3lyp",
            "tokens": ["B3LYP"],
            "source": "explicit_alias",
        },
        {
            "name": None,
            "tokens": ["xpbe96", "cpbe96"],
            "source": "explicit_expression",
        },
    ]


def test_tddft_task_inherits_dft_xc_state(tmp_path):
    path = _write(
        tmp_path,
        (
            "start water\n"
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "end\n"
            "dft; xc bhlyp; end\n"
            "tddft; nroots 5; end\n"
            "task tddft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["module"] == "tddft"
    assert states[0]["xc"] == {
        "name": "bhlyp",
        "tokens": ["bhlyp"],
        "source": "explicit_alias",
    }


def test_task_states_leave_restart_basis_mode_unresolved(tmp_path):
    path = _write(
        tmp_path,
        (
            "restart water\n"
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["basis"] == {
        "name": "ao basis",
        "mode": None,
        "mode_source": None,
        "source": "restart",
    }


def test_task_states_do_not_treat_restart_ecp_state_as_complete(tmp_path):
    path = _write(
        tmp_path,
        (
            "restart old\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            "ecp; Th nelec 60; end\n"
            "charge 0\n"
            "dft; singlet; end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["ecp"]["source"] == "ambiguous"
    assert states[0]["ecp"]["core_electrons"] == {"Th": 60}
    assert states[0]["reference"]["source"] == "restart"


def test_task_states_follow_named_ecp_selection(tmp_path):
    path = _write(
        tmp_path,
        (
            "start thorium\n"
            "geometry units angstrom\n"
            "  Th 0 0 0\n"
            "end\n"
            'ecp "small core"\n'
            "  Th nelec 60\n"
            "end\n"
            'ecp "large core"\n'
            "  Th nelec 78\n"
            "end\n"
            "task dft energy\n"
            'set "ecp basis" "small core"\n'
            "task dft energy\n"
            'set "ecp basis" "large core"\n'
            "task dft energy\n"
            "ecp\n"
            "  Th nelec 60\n"
            "end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert [state["ecp"] for state in states] == [
        {
            "source": "none",
            "core_electrons": {},
            "library_elements": [],
            "default_library": False,
        },
        {
            "source": "explicit",
            "core_electrons": {"Th": 60},
            "library_elements": [],
            "default_library": False,
            "name": "small core",
        },
        {
            "source": "explicit",
            "core_electrons": {"Th": 78},
            "library_elements": [],
            "default_library": False,
            "name": "large core",
        },
        {
            "source": "explicit",
            "core_electrons": {"Th": 60},
            "library_elements": [],
            "default_library": False,
        },
    ]


def test_task_states_normalize_negative_dft_mult(tmp_path):
    path = _write(
        tmp_path,
        (
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "end\n"
            "dft; mult -3; end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["multiplicity"] == 3
    assert states[0]["multiplicity_source"] == "mult"
    assert states[0]["reference"]["class"] == "open_shell"


def test_task_states_use_default_dft_multiplicity(tmp_path):
    path = _write(
        tmp_path,
        (
            "start water\n"
            "geometry units angstrom\n"
            "  O 0 0 0\n"
            "end\n"
            "dft; xc pbe0; end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["multiplicity"] == 1
    assert states[0]["multiplicity_source"] == "default"
    assert states[0]["reference"] == {
        "kind": "closed_shell",
        "class": "closed_shell",
        "source": "default",
    }


def test_task_states_preserve_activated_open_shell_dft_reference(tmp_path):
    path = _write(
        tmp_path,
        (
            "start fragments\n"
            "geometry units angstrom\n"
            "  H 0 0 0\n"
            "end\n"
            "dft; mult 2; end\n"
            "task dft energy\n"
            "dft; mult 1; end\n"
            "task dft energy\n"
        ),
    )

    states = parse_task_states(path)

    assert [state["multiplicity"] for state in states] == [2, 1]
    assert [state["reference"] for state in states] == [
        {
            "kind": "open_shell",
            "class": "open_shell",
            "source": "default",
        },
        {
            "kind": "open_shell",
            "class": "open_shell",
            "source": "default",
        },
    ]


def test_task_states_track_explicit_open_shell_references(tmp_path):
    path = _write(
        tmp_path,
        (
            "geometry units angstrom\n"
            "  H 0 0 0\n"
            "end\n"
            "dft; odft; singlet; end\n"
            "task dft energy\n"
            "scf; uhf; doublet; end\n"
            "task scf energy\n"
            "task mp2 energy\n"
        ),
    )

    states = parse_task_states(path)

    assert states[0]["reference"] == {
        "kind": "odft",
        "class": "open_shell",
        "source": "explicit",
    }
    assert states[1]["reference"] == {
        "kind": "uhf",
        "class": "open_shell",
        "source": "explicit",
    }
    assert states[2]["reference"] == states[1]["reference"]


def test_task_states_track_tce_method_separately_from_module(tmp_path):
    path = _write(
        tmp_path,
        (
            "start correlated\n"
            "geometry units angstrom\n"
            "  N 0 0 0\n"
            "end\n"
            "scf; uhf; triplet; end\n"
            "tce; mbpt2; end\n"
            "task tce energy\n"
            "tce; ccsd; end\n"
            "task tce energy\n"
            "tce; ccsd(t); end\n"
            "task tce energy\n"
        ),
    )

    states = parse_task_states(path)

    assert [state["module"] for state in states] == [
        "tce",
        "tce",
        "tce",
    ]
    assert [state["method"] for state in states] == [
        "MP2",
        "CCSD",
        "CCSD(T)",
    ]
    assert [state["method_source"] for state in states] == [
        "explicit_tce_keyword",
        "explicit_tce_keyword",
        "explicit_tce_keyword",
    ]
    assert [state["multiplicity"] for state in states] == [3, 3, 3]
    assert [state["reference"]["kind"] for state in states] == [
        "uhf",
        "uhf",
        "uhf",
    ]


def test_task_states_do_not_guess_unrecognized_tce_method(tmp_path):
    path = _write(
        tmp_path,
        (
            "start custom\n"
            "tce; custom_model; end\n"
            "task tce energy\n"
        ),
    )

    state = parse_task_states(path)[0]

    assert state["module"] == "tce"
    assert "method" not in state
    assert "method_source" not in state


def test_lint_inline_fragments_is_info_not_warning(tmp_path):
    result = lint_nwchem_input(_write(tmp_path, _INLINE_FRAGMENT_JOB))
    codes = {i["code"]: i["level"] for i in result["issues"]}
    assert "fragment_source_not_found" not in codes
    assert codes.get("fragment_sources_inline") == "info"


def test_lint_flags_incore_deadlock_risk(tmp_path):
    job = _INLINE_FRAGMENT_JOB.replace(
        "  mult 5\n  vectors input fragment",
        "  mult 5\n  noio\n  grid nodisk\n  vectors input fragment",
    )
    result = lint_nwchem_input(_write(tmp_path, job))
    codes = {i["code"] for i in result["issues"]}
    assert "incore_scf_deadlock_risk" in codes


def test_resource_warnings_oversubscription(tmp_path):
    job = _write(tmp_path, "start x\nmemory total 100 mb\ngeometry\n He 0 0 0\nend\n")
    huge = 10 ** 6  # ranks far beyond any host core count
    warnings = _resource_warnings(job, {"mpi_ranks": huge})
    assert any("exceeds" in w and "cores" in w for w in warnings)


def test_resource_warnings_memory_overcommit(tmp_path):
    # 10^9 MB/rank is guaranteed to exceed host RAM.
    job = _write(tmp_path, "start x\nmemory total 1000000000 mb\ngeometry\n He 0 0 0\nend\n")
    warnings = _resource_warnings(job, {"mpi_ranks": 2})
    assert any("host RAM" in w for w in warnings)


def test_resource_warnings_quiet_for_sane_config(tmp_path):
    job = _write(tmp_path, "start x\nmemory total 500 mb\ngeometry\n He 0 0 0\nend\n")
    assert _resource_warnings(job, {"mpi_ranks": 2}) == []
