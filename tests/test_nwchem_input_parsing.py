r"""Regression tests for NWChem fragment-guess input parsing, the lint checks
built on it, and the launch resource-sanity warnings.

These cover the parsing bugs where a `vectors input fragment` line had its
`output` keyword, output filename, inline comment, or `\` line-continuation
marker captured as fragment movecs files, plus the two preflight warnings.
"""

from chemtools.programs.nwchem.parse.input import (
    _split_vectors_fragment_files,
    parse_start_blocks,
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
