"""Phase 1 contracts for program capabilities and built-in composition."""

from __future__ import annotations

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from chemtools.mcp import dispatch, inventory, modes
from chemtools.core import registry
from chemtools.core.program import (
    ArtifactKindSpec,
    InvalidProgramBackend,
    ProgramBackend,
    ProgramCapability,
    UnsupportedCapabilityError,
    validate_backend,
)
from chemtools.core.registry import (
    ProgramAlreadyRegistered,
    ProgramDetectionAmbiguous,
    ProgramDetectionProbe,
    ProgramDetectionSourceError,
    ProgramDetectorError,
    ProgramDetectorFailure,
    ProgramSourceFailure,
)
from chemtools.mcp.catalog import (
    BUILTIN_BACKENDS,
    GENERIC_TOOL_DEFINITIONS,
    GUIDED_TOOL_DEFINITIONS,
    KNOWLEDGE_TOOL_DEFINITIONS,
    ORBITRON_TOOL_DEFINITIONS,
    REFERENCE_TOOL_DEFINITIONS,
    SCIENCE_RUNTIME_TOOL_DEFINITIONS,
    builtin_program_names,
    catalog_tool_definitions,
    load_backend,
    load_tool_definitions,
    validate_catalog,
)
from chemtools.mcp.tools import generic, guided


class _Detector:
    def detect(self, output_head: str) -> bool:
        return output_head.startswith("TEST")

    def detect_version(self, output_head: str) -> str | None:
        return "1.0" if self.detect(output_head) else None


class _BrokenDetector:
    def detect(self, output_head: str) -> bool:
        raise RuntimeError("detector exploded")

    def detect_version(self, output_head: str) -> str | None:
        return None


class _Parser:
    def parse_output(self, path: str) -> dict:
        return {"path": path}

    def task_index(self, path: str) -> list:
        return [{"path": path}]


class _UnexpectedDiagnostics:
    def diagnose(self, parsed: dict) -> dict:
        raise AssertionError("undeclared diagnosis capability was called")


def _backend(
    *,
    capabilities: frozenset[ProgramCapability],
    parser: object | None = None,
) -> ProgramBackend:
    return ProgramBackend(
        name="test",
        capabilities=capabilities,
        artifact_kinds={
            "test.output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
            )
        },
        detector=_Detector(),
        parser=parser,
    )


def test_program_capability_values_are_exact():
    assert tuple(item.value for item in ProgramCapability) == (
        "output.parse",
        "output.task_index",
        "output.geometry",
        "output.orbitals",
        "output.frequencies",
        "output.trajectory",
        "output.thermochemistry",
        "input.parse",
        "input.draft",
        "input.lint",
        "input.patch",
        "binary.read",
        "binary.write",
        "diagnosis.run",
        "diagnosis.recovery",
        "resources.estimate",
        "progress.inspect",
        "run.consistency",
        "calculation.plan",
        "execution.plan",
        "examples.read",
    )


def test_backend_support_and_require_contract():
    backend = _backend(
        capabilities=frozenset({ProgramCapability.OUTPUT_PARSE}),
        parser=_Parser(),
    )

    assert backend.supports(ProgramCapability.OUTPUT_PARSE) is True
    assert backend.supports(ProgramCapability.OUTPUT_GEOMETRY) is False
    assert backend.require(ProgramCapability.OUTPUT_PARSE) is backend

    with pytest.raises(UnsupportedCapabilityError) as caught:
        backend.require(ProgramCapability.OUTPUT_GEOMETRY)

    assert caught.value.program == "test"
    assert caught.value.capability is ProgramCapability.OUTPUT_GEOMETRY
    assert caught.value.available_capabilities == ("output.parse",)
    assert str(caught.value) == (
        "'test' does not support 'output.geometry'; "
        "available capabilities: ['output.parse']"
    )


def test_backend_validation_accepts_matching_provider_methods():
    backend = _backend(
        capabilities=frozenset(
            {
                ProgramCapability.OUTPUT_PARSE,
                ProgramCapability.OUTPUT_TASK_INDEX,
            }
        ),
        parser=_Parser(),
    )

    assert validate_backend(backend) is backend


def test_backend_validation_rejects_missing_capability_method():
    backend = _backend(
        capabilities=frozenset({ProgramCapability.OUTPUT_GEOMETRY}),
        parser=_Parser(),
    )

    with pytest.raises(
        InvalidProgramBackend,
        match=(
            "^backend 'test' declares 'output.geometry' "
            "but parser.get_geometry is unavailable$"
        ),
    ):
        validate_backend(backend)


def test_backend_validation_rejects_missing_consistency_adapter():
    backend = _backend(
        capabilities=frozenset({ProgramCapability.RUN_CONSISTENCY}),
    )

    with pytest.raises(
        InvalidProgramBackend,
        match=(
            "^backend 'test' declares 'run.consistency' but "
            "consistency.compare_input_output is unavailable$"
        ),
    ):
        validate_backend(backend)


def test_backend_validation_rejects_unqualified_artifact_kind():
    backend = _backend(capabilities=frozenset())
    backend = replace(
        backend,
        artifact_kinds={
            "output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
            )
        },
    )

    with pytest.raises(
        InvalidProgramBackend,
        match=r"^artifact kind 'output' must start with 'test\.'$",
    ):
        validate_backend(backend)


def test_artifact_kind_rejects_unknown_content_declaration():
    assert ArtifactKindSpec(
        extensions=(".out",),
        default_roles=frozenset({"primary_output"}),
    ).content_kind == "unknown"

    with pytest.raises(
        ValueError,
        match="^content_kind must be 'text', 'binary', or 'unknown'$",
    ):
        ArtifactKindSpec(
            extensions=(".out",),
            default_roles=frozenset({"primary_output"}),
            content_kind="archive",
        )


def test_registry_rejects_duplicate_program_names():
    backend = _backend(capabilities=frozenset())
    registry.register(backend)
    try:
        with pytest.raises(
            ProgramAlreadyRegistered,
            match="^A program is already registered as 'test'$",
        ):
            registry.register(backend)
        assert registry.get("test") is backend
    finally:
        registry.unregister("test")


def test_registry_returns_every_positive_detector_match(tmp_path):
    first = _backend(
        capabilities=frozenset({ProgramCapability.OUTPUT_PARSE}),
        parser=_Parser(),
    )
    second = replace(
        first,
        name="other",
        artifact_kinds={
            "other.output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
            )
        },
    )
    path = tmp_path / "run.out"
    path.write_text("TEST output\n", encoding="utf-8")
    registry.register(first)
    registry.register(second)
    try:
        assert registry.detect_candidates_from_text("TEST output") == (
            "test",
            "other",
        )
        assert registry.detect_candidates_from_file(str(path)) == (
            "test",
            "other",
        )
        assert registry.detect_from_file(str(path)) == "test"
        with pytest.raises(ProgramDetectionAmbiguous) as caught:
            registry.resolve(None, path=str(path))

        assert caught.value.candidates == ("test", "other")
        assert str(caught.value) == (
            f"Could not auto-detect one program from {str(path)!r}; content "
            "matches multiple registered programs: ['test', 'other']. "
            "Pass program explicitly."
        )
        assert registry.resolve("other", path=str(path)) is second
        assert generic._handle_parse_output_generic({
            "output_file": str(path),
        }) == {
            "error": "program_detection_ambiguous",
            "message": str(caught.value),
            "candidates": ["test", "other"],
        }
        assert generic._handle_apply_recovery_generic({
            "output_file": str(path),
        }) == {
            "error": "program_detection_ambiguous",
            "message": str(caught.value),
            "candidates": ["test", "other"],
        }
    finally:
        registry.unregister("test")
        registry.unregister("other")


def test_strict_resolution_reports_detector_failures_without_changing_compatibility(
    tmp_path,
):
    broken = replace(
        _backend(capabilities=frozenset()),
        name="broken",
        artifact_kinds={
            "broken.output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
            )
        },
        detector=_BrokenDetector(),
    )
    loose = replace(
        _backend(capabilities=frozenset()),
        name="loose",
        artifact_kinds={
            "loose.output": ArtifactKindSpec(
                extensions=(".out",),
                default_roles=frozenset({"primary_output"}),
            )
        },
    )
    path = tmp_path / "run.out"
    path.write_text("TEST output\n", encoding="utf-8")
    registry.register(broken)
    registry.register(loose)
    try:
        failure = ProgramDetectorFailure(
            program="broken",
            error_type="RuntimeError",
            message="detector exploded",
        )
        assert registry.detect_candidates_from_text("TEST output") == (
            "loose",
        )
        assert registry.detect_candidates_from_file(str(path)) == ("loose",)
        assert registry.detect_from_file(str(path)) == "loose"
        assert registry.probe_from_file(str(path)) == ProgramDetectionProbe(
            candidates=("loose",),
            detector_failures=(failure,),
        )

        with pytest.raises(ProgramDetectorError) as caught:
            registry.resolve(None, path=str(path))

        expected_message = (
            f"Could not safely resolve a program from {str(path)!r}; detector "
            "failure(s): broken (RuntimeError: detector exploded). Successful "
            "candidates: ['loose']."
        )
        assert str(caught.value) == expected_message
        assert caught.value.failures == (failure,)
        assert caught.value.candidates == ("loose",)

        with pytest.raises(ProgramDetectorError):
            registry.resolve("broken", path=str(path))
        assert registry.resolve("loose", path=str(path)) is loose
        sparse_path = tmp_path / "fragment.out"
        sparse_path.write_text("partial energy record\n", encoding="utf-8")
        assert registry.resolve("loose", path=str(sparse_path)) is loose

        expected_payload = {
            "error": "program_detector_error",
            "message": expected_message,
            "candidates": ["loose"],
            "detector_failures": [{
                "program": "broken",
                "error_type": "RuntimeError",
                "message": "detector exploded",
            }],
        }
        assert generic._resolve_plugin_or_error({
            "output_file": str(path),
        }) == (None, expected_payload)
        assert guided._handle_inspect_run({
            "output_file": str(path),
        }) == expected_payload
        assert generic._handle_apply_recovery_generic({
            "output_file": str(path),
        }) == expected_payload
    finally:
        registry.unregister("broken")
        registry.unregister("loose")


def test_strict_resolution_reports_source_read_failures(tmp_path):
    missing = tmp_path / "missing.out"

    assert registry.detect_candidates_from_file(str(missing)) == ()
    assert registry.detect_from_file(str(missing)) is None
    probe = registry.probe_from_file(str(missing))
    assert probe.candidates == ()
    assert probe.detector_failures == ()
    assert probe.source_failure == ProgramSourceFailure(
        error_type="FileNotFoundError",
        message=f"[Errno 2] No such file or directory: {str(missing)!r}",
        errno=2,
    )

    with pytest.raises(ProgramDetectionSourceError) as caught:
        registry.resolve(None, path=str(missing))

    expected_message = (
        f"Could not read program-detection source {str(missing)!r}: "
        f"FileNotFoundError: [Errno 2] No such file or directory: "
        f"{str(missing)!r}"
    )
    assert str(caught.value) == expected_message
    expected_payload = {
        "error": "program_source_error",
        "message": expected_message,
        "path": str(missing),
        "source_failure": {
            "error_type": "FileNotFoundError",
            "message": (
                f"[Errno 2] No such file or directory: {str(missing)!r}"
            ),
            "errno": 2,
        },
    }
    assert generic._resolve_plugin_or_error({
        "output_file": str(missing),
    }) == (None, expected_payload)
    assert generic._handle_apply_recovery_generic({
        "output_file": str(missing),
    }) == expected_payload


def test_program_imports_do_not_register_builtins():
    probe = (
        "import json; "
        "from chemtools.core import registry; "
        "import chemtools.programs.nwchem; "
        "import chemtools.programs.molcas; "
        "import chemtools.programs.dirac; "
        "import chemtools.programs.grasp; "
        "print(json.dumps(registry.list_programs()))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_catalog_registers_builtins_in_exact_order():
    probe = (
        "import json; "
        "from chemtools.core import registry; "
        "from chemtools.mcp.catalog import register_builtin_backends; "
        "loaded = register_builtin_backends(); "
        "print(json.dumps({"
        "'loaded': [backend.name for backend in loaded], "
        "'registered': registry.list_programs()"
        "}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "loaded": [
            "nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack", "orca",
        ],
        "registered": [
            "dirac", "grasp", "molcas", "nwchem", "orca", "qe", "qmcpack",
        ],
    }


@pytest.mark.parametrize(
    ("handler", "capability"),
    (
        (
            generic._handle_extract_geometry,
            ProgramCapability.OUTPUT_GEOMETRY,
        ),
        (
            generic._handle_parse_thermochem_generic,
            ProgramCapability.OUTPUT_THERMOCHEMISTRY,
        ),
        (
            generic._handle_parse_frequencies_generic,
            ProgramCapability.OUTPUT_FREQUENCIES,
        ),
        (
            generic._handle_parse_trajectory_generic,
            ProgramCapability.OUTPUT_TRAJECTORY,
        ),
        (
            generic._handle_inspect_geometry_generic,
            ProgramCapability.OUTPUT_GEOMETRY,
        ),
    ),
)
def test_generic_handlers_return_exact_unsupported_capability_error(
    monkeypatch,
    handler,
    capability,
):
    grasp = load_backend(BUILTIN_BACKENDS[3])
    monkeypatch.setattr(registry, "resolve", lambda **_kwargs: grasp)

    assert handler(
        {"output_file": "must-not-be-read.out", "program": "grasp"}
    ) == {
        "error": "unsupported_capability",
        "program": "grasp",
        "capability": capability.value,
            "available_capabilities": [
                "binary.read",
                "binary.write",
                "output.orbitals",
            "output.parse",
            "output.task_index",
        ],
    }


@pytest.mark.parametrize(
    "handler",
    (
        generic._handle_parse_output_generic,
        generic._handle_summarize_run,
    ),
)
def test_generic_parse_handlers_require_output_parse(monkeypatch, handler):
    backend = _backend(capabilities=frozenset(), parser=_Parser())
    monkeypatch.setattr(registry, "resolve", lambda **_kwargs: backend)

    assert handler(
        {"output_file": "must-not-be-read.out", "program": "test"}
    ) == {
        "error": "unsupported_capability",
        "program": "test",
        "capability": "output.parse",
        "available_capabilities": [],
    }


def test_summarize_run_skips_undeclared_diagnosis_provider(monkeypatch):
    backend = replace(
        _backend(
            capabilities=frozenset({ProgramCapability.OUTPUT_PARSE}),
            parser=_Parser(),
        ),
        diagnostics=_UnexpectedDiagnostics(),
    )
    monkeypatch.setattr(registry, "resolve", lambda **_kwargs: backend)

    assert generic._handle_summarize_run(
        {"output_file": "run.out", "program": "test"}
    ) == {
        "program": "test",
        "parsed": {"path": "run.out"},
        "diagnosis": None,
    }


def test_builtin_catalog_membership_and_providers_are_exact():
    assert builtin_program_names() == (
        "nwchem", "molcas", "dirac", "grasp", "qe", "qmcpack", "orca"
    )
    assert [
        (
            spec.name,
            spec.program_module,
            spec.backend_attribute,
            spec.tools_module,
            spec.definitions_attribute,
        )
        for spec in BUILTIN_BACKENDS
    ] == [
        (
            "nwchem",
            "chemtools.programs.nwchem",
            "NWCHEM",
            "chemtools.mcp.tools._nwchem_provider",
            "_nwchem_tool_definitions",
        ),
        (
            "molcas",
            "chemtools.programs.molcas",
            "MOLCAS",
            "chemtools.mcp.tools.molcas",
            "molcas_tool_definitions",
        ),
        (
            "dirac",
            "chemtools.programs.dirac",
            "DIRAC",
            "chemtools.mcp.tools.dirac",
            "dirac_tool_definitions",
        ),
        (
            "grasp",
            "chemtools.programs.grasp",
            "GRASP",
            "chemtools.mcp.tools.grasp",
            "grasp_tool_definitions",
        ),
        (
            "qe",
            "chemtools.programs.qe",
            "QE",
            "chemtools.mcp.tools.qe",
            "qe_tool_definitions",
        ),
        (
            "qmcpack",
            "chemtools.programs.qmcpack",
            "QMCPACK",
            "chemtools.mcp.tools.qmcpack",
            "qmcpack_tool_definitions",
        ),
        (
            "orca",
            "chemtools.programs.orca",
            "ORCA",
            "chemtools.mcp.tools.orca",
            "orca_tool_definitions",
        ),
    ]


def test_builtin_detectors_require_output_shaped_signatures():
    backends = {
        spec.name: load_backend(spec)
        for spec in BUILTIN_BACKENDS
    }

    assert not backends["nwchem"].detector.detect(
        "MOLPRO 2025.3\nTitle: compare against NWChem\nTotal energy -75.0\n"
    )
    assert backends["nwchem"].detector.detect("NWChem 7.2.3\n")

    nwchem_head = "Northwest Computational Chemistry Package\n"
    assert not backends["molcas"].detector.detect(
        nwchem_head + "title OpenMolcas comparison\n"
    )
    assert not backends["grasp"].detector.detect(
        nwchem_head + "title write orbitals in GRASP format\n"
    )
    assert not backends["dirac"].detector.detect(
        nwchem_head + "title Release DIRAC 25 comparison\n"
    )

    assert backends["molcas"].detector.detect("OpenMolcas\n")
    assert backends["dirac"].detector.detect("Release DIRAC 25.0\n")
    assert backends["orca"].detector.detect(
        "* O R C A *\nProgram Version 6.1.1 - RELEASE -\n"
    )


def test_builtin_catalog_rejects_duplicate_names():
    duplicate = replace(BUILTIN_BACKENDS[0], tools_module="example.tools")

    with pytest.raises(
        ValueError,
        match=(
            "^duplicate built-in backend names: "
            "\\['nwchem', 'nwchem'\\]$"
        ),
    ):
        validate_catalog((BUILTIN_BACKENDS[0], duplicate))


def test_catalog_loads_current_plugins_without_changing_provider_state():
    provider_state = {
        spec.name: tuple(
            getattr(load_backend(spec), field) is not None
            for field in (
                "parser",
                "drafter",
                "strategist",
                "binary",
                "consistency",
                "examples",
            )
        )
        for spec in BUILTIN_BACKENDS
    }

    assert provider_state == {
        "nwchem": (True, True, True, True, True, True),
        "molcas": (True, True, False, True, False, False),
        "dirac": (True, False, False, True, False, False),
        "grasp": (True, False, False, True, False, False),
        "qe": (True, True, False, False, True, False),
        "qmcpack": (True, True, False, False, True, False),
        "orca": (True, False, False, False, False, False),
    }


def test_builtin_backends_declare_exact_capabilities():
    expected = {
        "nwchem": {item.value for item in ProgramCapability},
        "molcas": {
            "binary.read",
            "binary.write",
            "execution.plan",
            "input.draft",
            "input.lint",
            "output.frequencies",
            "output.geometry",
            "output.trajectory",
            "output.orbitals",
            "output.parse",
            "output.task_index",
            "output.thermochemistry",
        },
        "dirac": {
            "binary.read",
            "execution.plan",
            "input.parse",
            "output.geometry",
            "output.parse",
            "output.task_index",
        },
        "grasp": {
            "binary.read",
            "binary.write",
            "output.orbitals",
            "output.parse",
            "output.task_index",
        },
        "qe": {
            "diagnosis.run",
            "execution.plan",
            "input.lint",
            "input.parse",
            "output.geometry",
            "output.parse",
            "output.task_index",
            "output.trajectory",
            "run.consistency",
        },
        "qmcpack": {
            "execution.plan",
            "input.lint",
            "input.parse",
            "output.parse",
            "output.task_index",
            "run.consistency",
        },
        "orca": {
            "input.parse",
            "output.frequencies",
            "output.geometry",
            "output.parse",
            "output.task_index",
        },
    }

    actual = {}
    for spec in BUILTIN_BACKENDS:
        backend = load_backend(spec)
        assert isinstance(backend, ProgramBackend)
        assert validate_backend(backend) is backend
        assert registry.get(spec.name) is backend
        actual[spec.name] = {item.value for item in backend.capabilities}

    assert actual == expected


def test_builtin_backends_preserve_legacy_extension_maps():
    assert {
        spec.name: load_backend(spec).file_extensions
        for spec in BUILTIN_BACKENDS
    } == {
        "nwchem": {
            "input": [".nw", ".nwi"],
            "output": [".out", ".nwo", ".log"],
            "error": [".err"],
            "movecs": [".movecs"],
            "hessian": [".hess"],
            "freq_restart": [".fdrst"],
            "trajectory": [".xyz"],
            "jobid": [".jobid"],
            "scratch": [".db", ".rmd"],
            "normal_modes": [".normal", ".nmode"],
        },
        "molcas": {
            "input": [".input", ".inp"],
            "output": [".out", ".log"],
            "error": [".err"],
            "runfile": [".RunFile"],
            "orbitals": ["INPORB"],
            "jobiph": ["JOBIPH"],
        },
        "dirac": {
            "input": [".inp"],
            "molecule": [".mol"],
            "output": [".out", ".log"],
            "error": [".err"],
            "checkpoint": [".h5"],
            "orbitals": ["DFCOEF", "DFPCMO", "DFACMO"],
        },
        "grasp": {
            "rmcdhf_summary": [".sum"],
            "rci_summary": [".csum"],
            "hfs": [".h", ".ch", ".hlsj", ".chlsj"],
            "isotope_shift": [".i", ".ci"],
            "transition": [".t.lsj", ".ct.lsj"],
            "lsj_label": [".lsj.lbl"],
            "mixing": [".m", ".cm"],
            "csf_list": [".c"],
            "radial_wfn": [".w"],
            "scf_log": [".log", ".alog"],
            "output": [".out"],
            "error": [".err"],
        },
        "qe": {
            "input": [".in"],
            "output": [".out"],
            "error": [".err"],
            "pw2qmcpack_hdf5": [".pwscf.h5"],
        },
        "qmcpack": {
            "input": [".xml"],
            "output": [".out"],
            "error": [".err"],
            "wavefunction_hdf5": [".h5"],
            "scalar": [".scalar.dat"],
            "dmc": [".dmc.dat"],
        },
        "orca": {
            "input": [".inp"],
            "output": [".out"],
            "error": [".err"],
            "wavefunction": [".gbw"],
            "hessian": [".hess"],
            "gradient": [".engrad"],
            "geometry": [".xyz"],
            "properties": [".property.txt"],
            "bibliography": [".bibtex"],
            "densities": [".densities", ".densitiesinfo"],
            "optimization_state": [".opt"],
        },
    }


def test_builtin_artifact_roles_are_exact():
    assert {
        spec.name: {
            kind: tuple(sorted(kind_spec.default_roles))
            for kind, kind_spec in load_backend(spec).artifact_kinds.items()
        }
        for spec in BUILTIN_BACKENDS
    } == {
        "nwchem": {
            "nwchem.input": ("primary_input",),
            "nwchem.output": ("primary_output",),
            "nwchem.error": ("stderr",),
            "nwchem.movecs": ("checkpoint", "orbital"),
            "nwchem.hessian": ("auxiliary_output",),
            "nwchem.freq_restart": ("checkpoint",),
            "nwchem.trajectory": ("auxiliary_output",),
            "nwchem.jobid": ("auxiliary_output",),
            "nwchem.scratch": ("auxiliary_output",),
            "nwchem.normal_modes": ("auxiliary_output",),
        },
        "molcas": {
            "molcas.input": ("primary_input",),
            "molcas.output": ("primary_output",),
            "molcas.error": ("stderr",),
            "molcas.runfile": ("checkpoint",),
            "molcas.orbitals": ("checkpoint", "orbital"),
            "molcas.jobiph": ("checkpoint", "orbital", "wavefunction"),
        },
        "dirac": {
            "dirac.input": ("primary_input",),
            "dirac.molecule": ("auxiliary_input",),
            "dirac.output": ("primary_output",),
            "dirac.error": ("stderr",),
            "dirac.checkpoint": ("checkpoint", "wavefunction"),
            "dirac.orbitals": ("checkpoint", "orbital"),
        },
        "grasp": {
            "grasp.rmcdhf_summary": ("primary_output",),
            "grasp.rci_summary": ("primary_output",),
            "grasp.hfs": ("primary_output",),
            "grasp.isotope_shift": ("primary_output",),
            "grasp.transition": ("primary_output",),
            "grasp.lsj_label": ("auxiliary_output",),
            "grasp.mixing": ("wavefunction",),
            "grasp.csf_list": ("auxiliary_output",),
            "grasp.radial_wfn": ("orbital", "wavefunction"),
            "grasp.scf_log": ("stdout",),
            "grasp.output": ("primary_output",),
            "grasp.error": ("stderr",),
        },
        "qe": {
            "qe.input": ("primary_input",),
            "qe.output": ("primary_output",),
            "qe.error": ("stderr",),
            "qe.pw2qmcpack_hdf5": ("checkpoint", "wavefunction"),
        },
        "qmcpack": {
            "qmcpack.input": ("primary_input",),
            "qmcpack.output": ("primary_output",),
            "qmcpack.error": ("stderr",),
            "qmcpack.wavefunction_hdf5": ("checkpoint", "wavefunction"),
            "qmcpack.scalar": ("auxiliary_output",),
            "qmcpack.dmc": ("auxiliary_output",),
        },
        "orca": {
            "orca.input": ("primary_input",),
            "orca.output": ("primary_output",),
            "orca.error": ("stderr",),
            "orca.wavefunction": ("checkpoint", "wavefunction"),
            "orca.hessian": ("auxiliary_output",),
            "orca.gradient": ("auxiliary_output",),
            "orca.geometry": ("auxiliary_output",),
            "orca.properties": ("auxiliary_output",),
            "orca.bibliography": ("auxiliary_output",),
            "orca.densities": ("auxiliary_output",),
            "orca.optimization_state": ("checkpoint",),
        },
    }


def test_builtin_artifact_content_kinds_are_exact():
    assert {
        spec.name: {
            kind: kind_spec.content_kind
            for kind, kind_spec in load_backend(spec).artifact_kinds.items()
        }
        for spec in BUILTIN_BACKENDS
    } == {
        "nwchem": {
            "nwchem.input": "text",
            "nwchem.output": "text",
            "nwchem.error": "text",
            "nwchem.movecs": "binary",
            "nwchem.hessian": "text",
            "nwchem.freq_restart": "unknown",
            "nwchem.trajectory": "text",
            "nwchem.jobid": "text",
            "nwchem.scratch": "binary",
            "nwchem.normal_modes": "unknown",
        },
        "molcas": {
            "molcas.input": "text",
            "molcas.output": "text",
            "molcas.error": "text",
            "molcas.runfile": "binary",
            "molcas.orbitals": "text",
            "molcas.jobiph": "binary",
        },
        "dirac": {
            "dirac.input": "text",
            "dirac.molecule": "text",
            "dirac.output": "text",
            "dirac.error": "text",
            "dirac.checkpoint": "binary",
            "dirac.orbitals": "binary",
        },
        "grasp": {
            "grasp.rmcdhf_summary": "text",
            "grasp.rci_summary": "text",
            "grasp.hfs": "text",
            "grasp.isotope_shift": "text",
            "grasp.transition": "text",
            "grasp.lsj_label": "text",
            "grasp.mixing": "binary",
            "grasp.csf_list": "text",
            "grasp.radial_wfn": "binary",
            "grasp.scf_log": "text",
            "grasp.output": "text",
            "grasp.error": "text",
        },
        "qe": {
            "qe.input": "text",
            "qe.output": "text",
            "qe.error": "text",
            "qe.pw2qmcpack_hdf5": "binary",
        },
        "qmcpack": {
            "qmcpack.input": "text",
            "qmcpack.output": "text",
            "qmcpack.error": "text",
            "qmcpack.wavefunction_hdf5": "binary",
            "qmcpack.scalar": "text",
            "qmcpack.dmc": "text",
        },
        "orca": {
            "orca.input": "text",
            "orca.output": "text",
            "orca.error": "text",
            "orca.wavefunction": "binary",
            "orca.hessian": "text",
            "orca.gradient": "text",
            "orca.geometry": "text",
            "orca.properties": "text",
            "orca.bibliography": "text",
            "orca.densities": "binary",
            "orca.optimization_state": "binary",
        },
    }


def test_catalog_tool_aggregation_matches_current_dispatch_exactly():
    assert len(load_tool_definitions(GENERIC_TOOL_DEFINITIONS)) == 39
    assert len(load_tool_definitions(GUIDED_TOOL_DEFINITIONS)) == 8
    assert len(load_tool_definitions(ORBITRON_TOOL_DEFINITIONS)) == 6
    assert len(load_tool_definitions(SCIENCE_RUNTIME_TOOL_DEFINITIONS)) == 10
    assert len(load_tool_definitions(KNOWLEDGE_TOOL_DEFINITIONS)) == 1
    assert len(load_tool_definitions(REFERENCE_TOOL_DEFINITIONS)) == 6
    assert {
        spec.name: len(load_tool_definitions(spec))
        for spec in BUILTIN_BACKENDS
    } == {
        "nwchem": 101,
        "molcas": 45,
        "dirac": 39,
        "grasp": 49,
        "qe": 20,
            "qmcpack": 14,
            "orca": 0,
        }

    catalog_names = [
        definition["name"] for definition in catalog_tool_definitions()
    ]
    dispatch_names = [
        definition["name"] for definition in dispatch.tool_definitions()
    ]
    assert len(catalog_names) == 338
    assert len(set(catalog_names)) == 338
    assert catalog_names == dispatch_names


def test_modes_and_dispatch_delegate_membership_to_catalog(monkeypatch):
    assert modes.KNOWN_PROGRAMS == builtin_program_names()
    assert inventory.PROGRAM_ORDER == ("generic", *builtin_program_names())

    marker = [
        {
            "name": "catalog_marker",
            "description": "Catalog delegation test",
            "inputSchema": {"type": "object"},
        }
    ]
    monkeypatch.setattr(dispatch, "catalog_tool_definitions", lambda: marker)

    assert dispatch.tool_definitions() is marker


def test_tools_package_import_has_no_registration_side_effects():
    probe = (
        "import json, sys; "
        "import chemtools.mcp.tools; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.startswith('chemtools.mcp.tools.'))))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
