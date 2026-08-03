"""Bounded text-evidence contracts for guided run inspection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chemtools.application.run_inspection import (
    RELATED_TEXT_LIMIT_BYTES,
    RELATED_TEXT_TOTAL_LIMIT_BYTES,
    inspect_run,
)
from chemtools.mcp.catalog import BUILTIN_BACKENDS, load_backend


FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"


def test_inspect_run_bounds_stderr_to_a_tail_excerpt(tmp_path):
    output_path = FIXTURES / "nwchem_scf.out"
    stderr_path = tmp_path / "nwchem.err"
    stderr_path.write_bytes(
        b"discard this prefix\n"
        + b"x" * RELATED_TEXT_LIMIT_BYTES
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(stderr_path,),
    )

    excerpt = inspected["evidence"]["artifacts"][1]["text_excerpt"]
    assert excerpt == {
        "role": "stderr",
        "position": "tail",
        "limit_bytes": RELATED_TEXT_LIMIT_BYTES,
        "bytes_read": RELATED_TEXT_LIMIT_BYTES,
        "boundary_bytes_discarded": 0,
        "truncated": True,
        "encoding": "utf-8",
        "decode_status": "decoded",
        "text": "x" * RELATED_TEXT_LIMIT_BYTES,
    }
    assert inspected["uncertainty"] == [{
        "code": "related_artifact_text_truncated",
        "message": (
            f"Related stderr artifact exceeds "
            f"{RELATED_TEXT_LIMIT_BYTES} bytes: {stderr_path.resolve()}"
        ),
        "impact": "Only the final bounded excerpt was inspected.",
    }]


def test_inspect_run_uses_utf8_safe_head_and_tail_segments(tmp_path):
    output_path = FIXTURES / "nwchem_scf.out"
    input_path = tmp_path / "large.nw"
    euro = "\u20ac".encode("utf-8")
    input_path.write_bytes(
        b"a" * 8191
        + euro
        + b"middle"
        + euro
        + b"z" * 8191
    )

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(input_path,),
    )

    excerpt = inspected["evidence"]["artifacts"][1]["text_excerpt"]
    assert excerpt == {
        "position": "head_tail",
        "limit_bytes": RELATED_TEXT_LIMIT_BYTES,
        "bytes_read": RELATED_TEXT_LIMIT_BYTES,
        "truncated": True,
        "encoding": "utf-8",
        "decode_status": "decoded",
        "segments": [
            {
                "position": "head",
                "byte_offset": 0,
                "bytes_read": 8192,
                "boundary_bytes_discarded": 1,
                "text": "a" * 8191,
            },
            {
                "position": "tail",
                "byte_offset": 8202,
                "bytes_read": 8192,
                "boundary_bytes_discarded": 1,
                "text": "z" * 8191,
            },
        ],
    }
    assert inspected["uncertainty"] == [{
        "code": "related_artifact_text_truncated",
        "message": (
            f"Related text artifact exceeds "
            f"{RELATED_TEXT_LIMIT_BYTES} bytes: {input_path.resolve()}"
        ),
        "impact": "Only bounded head and tail segments were inspected.",
    }]


def test_inspect_run_caps_total_related_text_evidence(tmp_path):
    output_path = FIXTURES / "nwchem_scf.out"
    input_paths = []
    for index in range(5):
        input_path = tmp_path / f"related-{index}.nw"
        input_path.write_bytes(
            bytes([ord("a") + index]) * RELATED_TEXT_LIMIT_BYTES
        )
        input_paths.append(input_path)

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=input_paths,
    )

    artifacts = inspected["evidence"]["artifacts"]
    assert [
        "text_excerpt" in artifact
        for artifact in artifacts
    ] == [False, True, True, True, True, False]
    assert [
        artifact.get("text_excerpt", {}).get("bytes_read", 0)
        for artifact in artifacts
    ] == [0, 16384, 16384, 16384, 16384, 0]
    assert inspected["evidence"]["text_excerpt_budget"] == {
        "limit_bytes": RELATED_TEXT_TOTAL_LIMIT_BYTES,
        "bytes_read": RELATED_TEXT_TOTAL_LIMIT_BYTES,
        "remaining_bytes": 0,
        "skipped_artifacts": 1,
    }
    assert inspected["uncertainty"] == [
        {
            "code": "related_artifact_text_budget_exhausted",
            "message": (
                "Text excerpt budget was exhausted before reading: "
                f"{input_paths[4].resolve()}"
            ),
            "impact": (
                "Artifact metadata is present, but its text was omitted."
            ),
        },
        {
            "code": "primary_input_ambiguous",
            "message": (
                "Input-output consistency was skipped because multiple "
                "primary inputs were supplied."
            ),
            "impact": "Supply only the input that produced this output.",
        },
    ]


def test_inspect_run_marks_invalid_utf8_in_stderr(tmp_path):
    output_path = FIXTURES / "nwchem_scf.out"
    stderr_path = tmp_path / "nwchem.err"
    stderr_path.write_bytes(b"rank failure: \xff\n")

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(stderr_path,),
    )

    excerpt = inspected["evidence"]["artifacts"][1]["text_excerpt"]
    assert excerpt["decode_status"] == "replacement_characters"
    assert excerpt["text"] == "rank failure: \ufffd\n"
    assert inspected["uncertainty"] == [{
        "code": "related_artifact_decode_replaced",
        "message": (
            f"Related stderr artifact is not valid UTF-8: "
            f"{stderr_path.resolve()}"
        ),
        "impact": (
            "Invalid byte sequences were replaced in the text excerpt."
        ),
    }]


def test_inspect_run_requires_text_declaration_before_reading_stderr(
    tmp_path,
):
    backend = load_backend(BUILTIN_BACKENDS[0])
    artifact_kinds = dict(backend.artifact_kinds)
    artifact_kinds["nwchem.error"] = replace(
        artifact_kinds["nwchem.error"],
        content_kind="binary",
    )
    backend = replace(backend, artifact_kinds=artifact_kinds)
    output_path = FIXTURES / "nwchem_scf.out"
    stderr_path = tmp_path / "nwchem.err"
    stderr_path.write_text("must remain unread\n", encoding="utf-8")

    inspected = inspect_run(
        backend,
        output_path,
        resolved_by="explicit",
        artifact_files=(stderr_path,),
    )

    stderr_evidence = inspected["evidence"]["artifacts"][1]
    assert stderr_evidence["classification"]["candidates"][0][
        "content_kind"
    ] == "binary"
    assert "text_excerpt" not in stderr_evidence
    assert inspected["uncertainty"] == []
