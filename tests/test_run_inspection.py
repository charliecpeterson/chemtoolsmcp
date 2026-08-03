"""Exact guided run-inspection contracts across current program backends."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chemtools.application.run_inspection import (
    PRIMARY_OUTPUT_LIMIT_BYTES,
    RELATED_TEXT_LIMIT_BYTES,
    RELATED_TEXT_TOTAL_LIMIT_BYTES,
    RunInspectionError,
    inspect_run,
    validate_primary_output_format,
)
from chemtools.core import registry
from chemtools.core.program import ProgramCapability
from chemtools.mcp.catalog import (
    BUILTIN_BACKENDS,
    load_backend,
    register_builtin_backends,
)
from chemtools.mcp.tools import generic, guided
from chemtools.programs._adapter_helpers import to_task_summary
from chemtools.programs.nwchem.parse.tasks import (
    detect_basis_token,
    detect_energy_token,
    parse_tasks,
)


FIXTURES = Path(__file__).parent / "golden" / "mcp" / "fixtures"


class _UnexpectedOutputParser:
    def parse_output(self, path: str) -> dict:
        raise AssertionError(f"parser must not read standalone NBO output: {path}")


def test_inspect_run_rejects_oversized_primary_output_before_parser(tmp_path):
    path = tmp_path / "oversized.out"
    path.touch()
    with path.open("r+b") as handle:
        handle.truncate(PRIMARY_OUTPUT_LIMIT_BYTES + 1)
    backend = replace(
        load_backend(BUILTIN_BACKENDS[0]),
        parser=_UnexpectedOutputParser(),
    )

    with pytest.raises(RunInspectionError) as caught:
        inspect_run(backend, path, resolved_by="explicit")

    assert caught.value.as_dict() == {
        "error": "primary_output_too_large",
        "message": (
            f"run output exceeds the {PRIMARY_OUTPUT_LIMIT_BYTES}-byte "
            f"inspection limit: {path.resolve()}"
        ),
        "program": "nwchem",
    }


def test_task_summary_does_not_report_byte_offsets_as_line_numbers():
    summary = to_task_summary(
        0,
        {"kind": "single_point"},
        {"boundary": {"start_byte": 4188, "end_byte": 39613}},
    )

    assert summary["line_range"] == (0, 0)


def test_inspect_run_rejects_standalone_nbo_before_parser(tmp_path):
    path = tmp_path / "analysis.out"
    path.write_text(
        "*********************************** NBO 6.0 "
        "***********************************\n"
        "N A T U R A L   B O N D   O R B I T A L   A N A L Y S I S\n",
        encoding="utf-8",
    )
    backend = replace(
        load_backend(BUILTIN_BACKENDS[0]),
        parser=_UnexpectedOutputParser(),
    )

    with pytest.raises(RunInspectionError) as caught:
        inspect_run(backend, path, resolved_by="explicit")

    assert caught.value.as_dict() == {
        "error": "unsupported_output_format",
        "message": (
            "standalone NBO analysis cannot be inspected as a nwchem "
            "run output; provide the parent quantum-chemistry output"
        ),
        "program": "nwchem",
        "detected_format": "nbo",
    }


def test_output_identity_keeps_nwchem_run_with_embedded_nbo(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(
        "Northwest Computational Chemistry Package\n"
        "N A T U R A L   B O N D   O R B I T A L   A N A L Y S I S\n",
        encoding="utf-8",
    )

    assert validate_primary_output_format(path, program="nwchem") is None


@pytest.mark.parametrize(("banner", "detected_format", "program"), [
    ("BANDS", "qe-bands", "bands.x"),
    ("DOS", "qe-dos", "dos.x"),
    ("PROJWFC", "qe-projwfc", "projwfc.x"),
    ("POST-PROC", "qe-pp", "pp.x"),
])
def test_inspect_run_rejects_unsupported_qe_companion_output(
    tmp_path,
    banner,
    detected_format,
    program,
):
    path = tmp_path / "postprocess.out"
    path.write_text(
        f"Program {banner} v.7.5 starts on 31Jul2026 at 10:00:00\n",
        encoding="utf-8",
    )
    backend = replace(
        load_backend(BUILTIN_BACKENDS[4]),
        parser=_UnexpectedOutputParser(),
    )

    with pytest.raises(RunInspectionError) as caught:
        inspect_run(backend, path, resolved_by="explicit")

    assert caught.value.as_dict() == {
        "error": "unsupported_output_format",
        "message": (
            f"Quantum ESPRESSO {program} output is not a supported primary "
            "run output; inspect the preceding pw.x output instead"
        ),
        "program": "qe",
        "detected_format": detected_format,
    }


def test_nwchem_sparse_energy_fragment_retains_numeric_evidence(tmp_path):
    backend = load_backend(BUILTIN_BACKENDS[0])
    cases = (
        ("dft", "Total DFT energy = -75.123\n", "DFT", -75.123),
        (
            "scf",
            "NWChem SCF Module\nTotal SCF energy = -1.117349034\n",
            "SCF",
            -1.117349034,
        ),
    )

    for name, contents, method, energy in cases:
        path = tmp_path / f"{name}.out"
        path.write_text(contents, encoding="utf-8")

        task = backend.parser.parse_output(str(path))["tasks"][0]

        assert task["kind"] == "energy"
        assert task["method"] == method
        assert task["energy_hartree"] == energy
        assert task["outcome"] == "incomplete"
        assert task["has_usable_data"] is True


def test_nwchem_energy_record_must_start_the_stripped_line():
    assert detect_energy_token("Total DFT energy = -75.123") == (
        2,
        -75.123,
    )
    assert detect_energy_token('title "Total DFT energy = -999"') is None
    assert detect_energy_token("# Total SCF energy = -999") is None


def test_nwchem_tddft_gradient_keeps_operation_and_excited_energy():
    contents = (
        "NWChem Input Module\n"
        "Total DFT energy = -112.5\n"
        "NWChem TDDFT Module\n"
        "Excited state energy = -112.2\n"
        "NWChem TDDFT Gradient Module\n"
        "TDDFT ENERGY GRADIENTS\n"
        "Task times cpu: 0.2s wall: 0.2s\n"
    )

    parsed = parse_tasks("tddft-gradient.out", contents)
    task = parsed["program_summary"]["raw"]["tasks"][0]
    generic_task = parsed["generic_tasks"][0]

    assert task["kind"] == "gradient"
    assert task["method"] == "TDDFT"
    assert task["total_energy_hartree"] == -112.2
    assert generic_task["kind"] == "gradient"


def test_nwchem_property_module_preserves_operation_kind():
    contents = (
        "NWChem Input Module\n"
        "NWChem Property Module\n"
        "NWChem DFT Module\n"
        "Task times cpu: 0.2s wall: 0.2s\n"
    )

    parsed = parse_tasks("property.out", contents)
    task = parsed["program_summary"]["raw"]["tasks"][0]
    generic_task = parsed["generic_tasks"][0]

    assert task["kind"] == "property"
    assert task["method"] == "DFT"
    assert task["outcome"] == "success"
    assert generic_task["kind"] == "property"


def test_nwchem_property_energy_failure_marks_task_failed():
    contents = (
        "NWChem Input Module\n"
        "NWChem Property Module\n"
        "NWChem DFT Module\n"
        "hnd_property: energy failure                 555\n"
    )

    parsed = parse_tasks("failed-property.out", contents)
    task = parsed["program_summary"]["raw"]["tasks"][0]

    assert task["kind"] == "property"
    assert task["outcome"] == "failed"
    assert parsed["program_summary"]["outcome"] == "failed"
    assert parsed["program_summary"]["diagnostics"] == [{
        "kind": "error",
        "message": "hnd_property: energy failure                 555",
        "line": 4,
    }]


def test_nwchem_correlated_energy_formats_and_optimization_priority():
    assert detect_energy_token(
        "Total MP2 energy:            -76.113291510749477"
    ) == (3, -76.113291510749477)
    assert detect_energy_token(
        "Total MP2 energy           -76.234714409690"
    ) == (3, -76.23471440969)
    assert detect_energy_token(
        "CCSD total energy / hartree = -76.120753972936896"
    ) == (4, -76.120753972936896)
    assert detect_energy_token(
        "Total CCSD(T) energy:       -76.121793080326924"
    ) == (5, -76.121793080326924)

    contents = (
        "NWChem Input Module\n"
        "NWChem Geometry Optimization\n"
        "Total SCF energy = -75.0\n"
        "Total MP2 energy: -76.1\n"
        "Total CCSD energy: -76.2\n"
        "Total CCSD(T) energy: -76.3\n"
        "Total SCF energy = -75.1\n"
        "Total MP2 energy -76.11\n"
        "Total CCSD energy: -76.21\n"
        "Total CCSD(T) energy: -76.31\n"
        "Task times cpu: 1.0s wall: 1.0s\n"
    )

    task = parse_tasks("ccsdt-opt.out", contents)["program_summary"][
        "raw"
    ]["tasks"][0]

    assert task["method"] == "CCSD(T)"
    assert task["total_energy_hartree"] == -76.31
    assert task["energy_profile"] == [-76.3, -76.31]


def test_nwchem_basis_family_rejects_runtime_labels_and_library_prose():
    assert detect_basis_token("* library cc-pVDZ") == "cc-pVDZ"
    assert detect_basis_token('H library "aug-cc-pvtz"') == (
        "aug-cc-pvtz"
    )
    for line in (
        "basis",
        "basis spherical",
        'Basis "ao basis" -> "ao basis" (cartesian)',
        "Basis functions       =     25",
        "basis label          566",
        "library name resolved from: environment",
        "library file name is: </apps/nwchem/data/libraries/>",
        "SETTING BASIS = ao basis",
    ):
        assert detect_basis_token(line) is None

    contents = (
        "* library cc-pVDZ\n"
        "NWChem Input Module\n"
        'Basis "ao basis" -> "ao basis" (cartesian)\n'
        "Basis functions = 25\n"
        "Total MP2 energy: -76.1\n"
        "Task times cpu: 1.0s wall: 1.0s\n"
    )

    task = parse_tasks("mp2.out", contents)["program_summary"]["raw"][
        "tasks"
    ][0]

    assert task["basis"] == "cc-pVDZ"

    mixed = parse_tasks(
        "mixed.out",
        (
            "H library cc-pVDZ\n"
            "O library aug-cc-pvtz\n"
            "NWChem Input Module\n"
            "Total MP2 energy: -76.1\n"
        ),
    )["program_summary"]["raw"]["tasks"][0]

    assert mixed["basis"] is None


def test_nwchem_sparse_energy_fragment_retains_following_error(tmp_path):
    path = tmp_path / "failed.out"
    path.write_text(
        "Total DFT energy = -75.123\nERROR: convergence failed\n",
        encoding="utf-8",
    )

    parsed = load_backend(BUILTIN_BACKENDS[0]).parser.parse_output(str(path))

    assert parsed["tasks"][0]["energy_hartree"] == -75.123
    assert parsed["tasks"][0]["outcome"] == "failed"
    assert parsed["diagnostics"] == [{
        "kind": "error",
        "message": "ERROR: convergence failed",
        "line": 2,
        "file": str(path),
    }]


def test_nwchem_module_header_retains_incomplete_task_before_energy():
    contents = (
        "NWChem Input Module\n"
        "NWChem DFT Module\n"
        "d= 0,ls=0.5,diis  12  -4585.1\n"
    )

    parsed = parse_tasks("running-dft.out", contents)
    task = parsed["program_summary"]["raw"]["tasks"][0]

    assert task["kind"] == "single_point"
    assert task["method"] == "DFT"
    assert task["total_energy_hartree"] is None
    assert task["outcome"] == "incomplete"
    assert parsed["generic_tasks"][0]["kind"] == "single_point"


def test_nwchem_input_error_before_module_emits_failed_unknown_task():
    contents = (
        "NWChem Input Module\n"
        "current input line :\n"
        "  17: force_field uff\n"
        "There is an error in the input file\n"
    )

    parsed = parse_tasks("input-error.out", contents)
    tasks = parsed["program_summary"]["raw"]["tasks"]

    assert len(tasks) == 1
    assert tasks[0]["kind"] == "unknown"
    assert tasks[0]["method"] is None
    assert tasks[0]["outcome"] == "failed"
    assert parsed["program_summary"]["outcome"] == "failed"
    assert parsed["program_summary"]["diagnostics"] == [{
        "kind": "error",
        "message": "There is an error in the input file",
        "line": 4,
    }]


def test_inspect_run_uses_backend_diagnosis_for_nwchem():
    path = FIXTURES / "nwchem_scf.out"

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        path,
        resolved_by="content",
    )

    assert inspected["schema_version"] == "chemtools.inspect-run/1"
    assert inspected["program"] == {
        "name": "nwchem",
        "version": "7.2.3",
        "resolved_by": "content",
    }
    assert inspected["source"] == {
        "path": str(path.resolve()),
        "size_bytes": 213,
    }
    assert inspected["assessment"] == {
        "source": "backend_diagnosis",
        "verdict": {
            "label": "success",
            "confidence": 0.6,
            "reasons": ["stage: single_point"],
        },
    }
    assert inspected["evidence"]["artifact_classification"] == {
        "status": "matched",
        "candidates": [{
            "kind": "nwchem.output",
            "roles": ["primary_output"],
            "content_kind": "text",
            "evidence": "inferred",
            "matched_by": "extension",
            "matched_value": ".out",
        }],
    }
    assert inspected["evidence"]["artifacts"] == [{
        "path": str(path.resolve()),
        "relationship": "primary_output",
        "exists": True,
        "entry_type": "file",
        "size_bytes": 213,
        "classification": {
            "status": "matched",
            "candidates": [{
                "kind": "nwchem.output",
                "roles": ["primary_output"],
                "content_kind": "text",
                "evidence": "inferred",
                "matched_by": "extension",
                "matched_value": ".out",
            }],
        },
    }]
    assert inspected["evidence"]["derived"]["final_energy_hartree"] == (
        -1.117349034
    )
    assert inspected["uncertainty"] == []
    assert inspected["next_actions"] == [{
        "tool": "analyze_nwchem_frontier_orbitals",
        "params": {},
        "reason": "verify state quality before accepting result",
        "confidence": 0.6,
        "priority": 1,
    }]


def test_inspect_run_marks_task_outcome_fallback_for_dirac():
    path = FIXTURES / "dirac_scf.out"

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[2]),
        path,
        resolved_by="explicit",
    )

    assert inspected["program"] == {
        "name": "dirac",
        "version": "25.0",
        "resolved_by": "explicit",
    }
    assert inspected["assessment"] == {
        "source": "task_outcomes",
        "verdict": {
            "label": "converged",
            "confidence": 0.7,
            "reasons": ["The parser reports SCF convergence."],
        },
    }
    assert inspected["uncertainty"] == [{
        "code": "scientific_diagnosis_unavailable",
        "message": "dirac has no guided scientific diagnosis adapter.",
        "impact": (
            "The verdict reflects parser task outcomes, not a full review."
        ),
    }]
    assert inspected["evidence"]["diagnostics"] == []


def test_inspect_run_uses_clean_molcas_module_return_code():
    path = FIXTURES / "molcas_scf.out"

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[1]),
        path,
        resolved_by="content",
    )

    assert inspected["program"] == {
        "name": "molcas",
        "version": None,
        "resolved_by": "content",
    }
    assert inspected["assessment"] == {
        "source": "task_outcomes",
        "verdict": {
            "label": "completed",
            "confidence": 0.7,
            "reasons": ["All 1 parsed task(s) completed."],
        },
    }
    assert inspected["evidence"]["tasks"][0]["outcome"] == "success"
    assert inspected["evidence"]["derived"][
        "primary_energy_hartree"
    ] == -75.0239826189
    assert inspected["uncertainty"] == [{
        "code": "scientific_diagnosis_unavailable",
        "message": "molcas has no guided scientific diagnosis adapter.",
        "impact": (
            "The verdict reflects parser task outcomes, not a full review."
        ),
    }]


def test_inspect_run_reports_specific_grasp_sum_artifact():
    path = FIXTURES / "grasp.sum"

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[3]),
        path,
        resolved_by="content",
    )

    assert inspected["program"] == {
        "name": "grasp",
        "version": "2018",
        "resolved_by": "content",
    }
    assert inspected["assessment"] == {
        "source": "task_outcomes",
        "verdict": {
            "label": "completed",
            "confidence": 0.7,
            "reasons": ["All 1 parsed task(s) completed."],
        },
    }
    assert inspected["evidence"]["artifact_classification"]["status"] == "matched"
    assert [
        candidate["kind"]
        for candidate in inspected["evidence"][
            "artifact_classification"
        ]["candidates"]
    ] == ["grasp.rmcdhf_summary"]
    assert [item["code"] for item in inspected["uncertainty"]] == [
        "scientific_diagnosis_unavailable",
    ]


def test_inspect_run_preserves_parse_when_optional_diagnosis_fails():
    backend = load_backend(BUILTIN_BACKENDS[0])

    class _BrokenDiagnostics:
        def diagnose(self, parsed):
            raise RuntimeError("diagnostic fixture failure")

        def suggest_recovery(self, parsed, diagnosis):
            return []

    broken = replace(backend, diagnostics=_BrokenDiagnostics())
    path = FIXTURES / "nwchem_scf.out"

    inspected = inspect_run(
        broken,
        path,
        resolved_by="explicit",
    )

    assert ProgramCapability.DIAGNOSIS_RUN in broken.capabilities
    assert inspected["assessment"] == {
        "source": "task_outcomes",
        "verdict": {
            "label": "completed",
            "confidence": 0.7,
            "reasons": ["All 1 parsed task(s) completed."],
        },
    }
    assert inspected["uncertainty"] == [{
        "code": "backend_diagnosis_failed",
        "message": (
            "nwchem diagnosis failed with RuntimeError: "
            "diagnostic fixture failure"
        ),
        "impact": "The verdict uses parsed task outcomes only.",
    }]


def test_inspect_run_observes_only_explicit_related_artifacts(tmp_path):
    output_path = FIXTURES / "nwchem_scf.out"
    input_path = FIXTURES / "nwchem_h2.nw"
    stderr_path = tmp_path / "nwchem.err"
    checkpoint_path = tmp_path / "nwchem.movecs"
    stderr_path.write_text("rank 0 stderr\n", encoding="utf-8")
    checkpoint_path.write_bytes(b"\x00\x01\x02")

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="explicit",
        artifact_files=(
            input_path,
            stderr_path,
            checkpoint_path,
            output_path,
            checkpoint_path,
        ),
    )

    assert inspected["evidence"]["artifacts"] == [
        {
            "path": str(output_path.resolve()),
            "relationship": "primary_output",
            "exists": True,
            "entry_type": "file",
            "size_bytes": 213,
            "classification": {
                "status": "matched",
                "candidates": [{
                    "kind": "nwchem.output",
                    "roles": ["primary_output"],
                    "content_kind": "text",
                    "evidence": "inferred",
                    "matched_by": "extension",
                    "matched_value": ".out",
                }],
            },
        },
        {
            "path": str(input_path.resolve()),
            "relationship": "related",
            "exists": True,
            "entry_type": "file",
            "size_bytes": 171,
            "classification": {
                "status": "matched",
                "candidates": [{
                    "kind": "nwchem.input",
                    "roles": ["primary_input"],
                    "content_kind": "text",
                    "evidence": "inferred",
                    "matched_by": "extension",
                    "matched_value": ".nw",
                }],
            },
            "text_excerpt": {
                "position": "whole",
                "limit_bytes": RELATED_TEXT_LIMIT_BYTES,
                "bytes_read": 171,
                "truncated": False,
                "encoding": "utf-8",
                "decode_status": "decoded",
                "segments": [{
                    "position": "whole",
                    "byte_offset": 0,
                    "bytes_read": 171,
                    "boundary_bytes_discarded": 0,
                    "text": (
                        "start chemtools_h2_review\n"
                        "\n"
                        "geometry units angstroms\n"
                        "  H 0.0 0.0 0.0\n"
                        "  H 0.0 0.0 0.74\n"
                        "end\n"
                        "\n"
                        "basis\n"
                        "  * library sto-3g\n"
                        "end\n"
                        "\n"
                        "scf\n"
                        "  singlet\n"
                        "  thresh 1.0e-8\n"
                        "end\n"
                        "\n"
                        "task scf energy\n"
                    ),
                }],
            },
        },
        {
            "path": str(stderr_path.resolve()),
            "relationship": "related",
            "exists": True,
            "entry_type": "file",
            "size_bytes": 14,
            "classification": {
                "status": "matched",
                "candidates": [{
                    "kind": "nwchem.error",
                    "roles": ["stderr"],
                    "content_kind": "text",
                    "evidence": "inferred",
                    "matched_by": "extension",
                    "matched_value": ".err",
                }],
            },
            "text_excerpt": {
                "role": "stderr",
                "position": "tail",
                "limit_bytes": RELATED_TEXT_LIMIT_BYTES,
                "bytes_read": 14,
                "boundary_bytes_discarded": 0,
                "truncated": False,
                "encoding": "utf-8",
                "decode_status": "decoded",
                "text": "rank 0 stderr\n",
            },
        },
        {
            "path": str(checkpoint_path.resolve()),
            "relationship": "related",
            "exists": True,
            "entry_type": "file",
            "size_bytes": 3,
            "classification": {
                "status": "matched",
                "candidates": [{
                    "kind": "nwchem.movecs",
                    "roles": ["checkpoint", "orbital"],
                    "content_kind": "binary",
                    "evidence": "inferred",
                    "matched_by": "extension",
                    "matched_value": ".movecs",
                }],
            },
        },
    ]
    assert inspected["evidence"]["text_excerpt_budget"] == {
        "limit_bytes": RELATED_TEXT_TOTAL_LIMIT_BYTES,
        "bytes_read": 185,
        "remaining_bytes": RELATED_TEXT_TOTAL_LIMIT_BYTES - 185,
        "skipped_artifacts": 0,
    }
    assert inspected["uncertainty"] == []


def test_inspect_run_keeps_molcas_jobiph_as_binary_metadata(tmp_path):
    output_path = FIXTURES / "molcas_scf.out"
    jobiph_path = tmp_path / "JOBIPH"
    jobiph_path.write_bytes(b"\x00job interface\xff")

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[1]),
        output_path,
        resolved_by="explicit",
        artifact_files=(jobiph_path,),
    )

    jobiph_evidence = inspected["evidence"]["artifacts"][1]
    assert jobiph_evidence == {
        "path": str(jobiph_path.resolve()),
        "relationship": "related",
        "exists": True,
        "entry_type": "file",
        "size_bytes": 15,
        "classification": {
            "status": "matched",
            "candidates": [{
                "kind": "molcas.jobiph",
                "roles": ["checkpoint", "orbital", "wavefunction"],
                "content_kind": "binary",
                "evidence": "inferred",
                "matched_by": "filename",
                "matched_value": "JOBIPH",
            }],
        },
    }
    assert "text_excerpt" not in jobiph_evidence
    assert inspected["uncertainty"] == [{
        "code": "scientific_diagnosis_unavailable",
        "message": "molcas has no guided scientific diagnosis adapter.",
        "impact": (
            "The verdict reflects parser task outcomes, not a full review."
        ),
    }]


def test_inspect_run_reports_related_file_state_without_scanning(tmp_path):
    output_path = FIXTURES / "nwchem_scf.out"
    missing_path = tmp_path / "missing.movecs"
    directory_path = tmp_path / "scratch.movecs"
    nested_path = directory_path / "hidden.err"
    unknown_path = tmp_path / "scheduler.notes"
    directory_path.mkdir()
    nested_path.write_text("must not be discovered\n", encoding="utf-8")
    unknown_path.write_text("node failure\n", encoding="utf-8")

    inspected = inspect_run(
        load_backend(BUILTIN_BACKENDS[0]),
        output_path,
        resolved_by="content",
        artifact_files=(missing_path, directory_path, unknown_path),
    )

    artifacts = inspected["evidence"]["artifacts"]
    assert [artifact["path"] for artifact in artifacts] == [
        str(output_path.resolve()),
        str(missing_path.resolve()),
        str(directory_path.resolve()),
        str(unknown_path.resolve()),
    ]
    assert artifacts[1] == {
        "path": str(missing_path.resolve()),
        "relationship": "related",
        "exists": False,
        "entry_type": None,
        "size_bytes": None,
        "classification": {
            "status": "matched",
            "candidates": [{
                "kind": "nwchem.movecs",
                "roles": ["checkpoint", "orbital"],
                "content_kind": "binary",
                "evidence": "inferred",
                "matched_by": "extension",
                "matched_value": ".movecs",
            }],
        },
    }
    assert artifacts[2] == {
        "path": str(directory_path.resolve()),
        "relationship": "related",
        "exists": True,
        "entry_type": "directory",
        "size_bytes": None,
        "classification": None,
    }
    assert nested_path.resolve() not in {
        Path(artifact["path"]) for artifact in artifacts
    }
    assert artifacts[3]["classification"] == {
        "status": "unmatched",
        "candidates": [],
    }
    assert inspected["uncertainty"] == [
        {
            "code": "related_artifact_missing",
            "message": (
                f"Related artifact does not exist: {missing_path.resolve()}"
            ),
            "impact": "That artifact could not contribute run evidence.",
        },
        {
            "code": "related_artifact_not_file",
            "message": (
                f"Related artifact is not a file: {directory_path.resolve()}"
            ),
            "impact": (
                "Directories and special filesystem entries are not "
                "inspected."
            ),
        },
        {
            "code": "related_artifact_kind_unmatched",
            "message": (
                f"Related artifact matches no declared kind: "
                f"{unknown_path.resolve()}"
            ),
            "impact": "Its role in the run could not be established.",
        },
    ]


def test_inspect_run_handler_rejects_missing_source_before_detection(tmp_path):
    missing = tmp_path / "missing.out"

    assert guided._handle_inspect_run(
        {"output_file": str(missing)}
    ) == {
        "error": "source_not_file",
        "message": f"run output is not a readable file: {missing}",
    }


def test_inspect_run_handler_rejects_invalid_artifact_paths():
    path = FIXTURES / "nwchem_scf.out"

    assert guided._handle_inspect_run({
        "output_file": str(path),
        "artifact_files": str(FIXTURES / "nwchem_h2.nw"),
    }) == {
        "error": "invalid_artifact_files",
        "message": (
            "artifact_files must be an array of at most 64 non-empty path "
            "strings."
        ),
    }


def test_inspect_run_handler_identifies_standalone_nbo(tmp_path):
    path = tmp_path / "analysis.out"
    path.write_text(
        "*********************************** NBO 6.0 "
        "***********************************\n"
        "N A T U R A L   B O N D   O R B I T A L   A N A L Y S I S\n",
        encoding="utf-8",
    )
    if not registry.has("nwchem"):
        register_builtin_backends()

    assert guided._handle_inspect_run({
        "output_file": str(path),
    }) == {
        "error": "unsupported_output_format",
        "message": (
            "standalone NBO analysis is not a supported primary run output; "
            "provide the parent quantum-chemistry output that contains the "
            "electronic-structure calculation"
        ),
        "detected_format": "nbo",
    }
    assert guided._handle_inspect_run({
        "output_file": str(path),
        "program": "nwchem",
    }) == {
        "error": "unsupported_output_format",
        "message": (
            "standalone NBO analysis cannot be inspected as a nwchem run "
            "output; provide the parent quantum-chemistry output"
        ),
        "program": "nwchem",
        "detected_format": "nbo",
    }


def test_inspect_run_handler_identifies_unsupported_qe_companion_output(tmp_path):
    path = tmp_path / "bands.out"
    path.write_text(
        "Program BANDS v.7.5 starts on 31Jul2026 at 10:00:00\n",
        encoding="utf-8",
    )
    if not registry.has("qe"):
        register_builtin_backends()

    assert guided._handle_inspect_run({
        "output_file": str(path),
    }) == {
        "error": "unsupported_output_format",
        "message": (
            "Quantum ESPRESSO bands.x output is not a supported primary run "
            "output; inspect the preceding pw.x output instead"
        ),
        "detected_format": "qe-bands",
    }


@pytest.mark.parametrize(
    ("filename", "detected_program"),
    (
        ("molcas_scf.out", "molcas"),
        ("dirac_scf.out", "dirac"),
        ("grasp.sum", "grasp"),
    ),
)
def test_inspect_run_handler_rejects_conflicting_program_override(
    filename,
    detected_program,
):
    if not registry.has("nwchem"):
        register_builtin_backends()

    assert guided._handle_inspect_run({
        "output_file": str(FIXTURES / filename),
        "program": "nwchem",
    }) == {
        "error": "program_content_mismatch",
        "message": (
            f"run output content matches {detected_program}, but program "
            "override selected nwchem"
        ),
        "program": "nwchem",
        "detected_programs": [detected_program],
    }
    assert generic._handle_parse_output_generic({
        "output_file": str(FIXTURES / filename),
        "program": "nwchem",
    }) == {
        "error": "program_content_mismatch",
        "message": (
            f"run output content matches {detected_program}, but program "
            "override selected nwchem"
        ),
        "program": "nwchem",
        "detected_programs": [detected_program],
    }


def test_apply_recovery_rejects_input_from_another_program(tmp_path):
    input_path = tmp_path / "job.nw"
    target_path = tmp_path / "job_recovered.nw"
    input_path.write_text(
        "geometry\nH 0 0 0\nend\ntask scf energy\n",
        encoding="utf-8",
    )
    if not registry.has("nwchem"):
        register_builtin_backends()

    recovered = generic._handle_apply_recovery_generic({
        "program": "molcas",
        "input_file": str(input_path),
        "recovery": {
            "failure_class": "memory_exceeded",
            "current_memory_mb": 4000,
        },
        "write_to": str(target_path),
    })

    assert recovered == {
        "error": "program_content_mismatch",
        "message": (
            "recovery input content matches nwchem, but selected program is molcas"
        ),
        "program": "molcas",
        "detected_programs": ["nwchem"],
    }
    assert not target_path.exists()


def test_inspect_run_handler_allows_sparse_explicit_output(tmp_path):
    path = tmp_path / "fragment.out"
    path.write_text("Total DFT energy = -75.123\n", encoding="utf-8")
    if not registry.has("nwchem"):
        register_builtin_backends()

    inspected = guided._handle_inspect_run({
        "output_file": str(path),
        "program": "nwchem",
    })

    assert "error" not in inspected
    assert inspected["program"]["resolved_by"] == "explicit"
    assert inspected["evidence"]["tasks"][0]["energy_hartree"] == -75.123


def test_inspect_run_handler_rejects_ambiguous_automatic_detection(tmp_path):
    path = tmp_path / "ambiguous.out"
    path.write_text(
        "Northwest Computational Chemistry Package\n"
        "Release DIRAC 25.0\n",
        encoding="utf-8",
    )
    if not registry.has("nwchem"):
        register_builtin_backends()

    assert guided._handle_inspect_run({
        "output_file": str(path),
    }) == {
        "error": "program_detection_ambiguous",
        "message": (
            f"Could not auto-detect one program from {str(path.resolve())!r}; "
            "content matches multiple registered programs: "
            "['nwchem', 'dirac']. Pass program explicitly."
        ),
        "candidates": ["nwchem", "dirac"],
    }

    inspected = guided._handle_inspect_run({
        "output_file": str(path),
        "program": "nwchem",
    })
    assert "error" not in inspected
    assert inspected["program"]["resolved_by"] == "explicit"


def test_inspect_run_handler_passes_explicit_artifacts():
    output_path = FIXTURES / "nwchem_scf.out"
    input_path = FIXTURES / "nwchem_h2.nw"
    if not registry.has("nwchem"):
        register_builtin_backends()

    inspected = guided._handle_inspect_run({
        "output_file": str(output_path),
        "artifact_files": [str(input_path)],
        "program": "nwchem",
    })

    assert [
        artifact["path"]
        for artifact in inspected["evidence"]["artifacts"]
    ] == [
        str(output_path.resolve()),
        str(input_path.resolve()),
    ]
