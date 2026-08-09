"""Program provider protocols and capability-backed backend declarations.

`ProgramBackend` is the current built-in contract. The broader `Program`
protocol remains temporarily for Python compatibility while callers move from
provider presence to operation-level capability checks.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from chemtools.core.artifacts import ArtifactRole
from chemtools.core.types import (
    ParsedRun,
    TaskSummary,
    GeometryAtom,
    InputSpec,
    LintIssue,
    Diagnosis,
    NextAction,
    ExampleEntry,
    TaskKind,
)
from chemtools.core.execution import PreparedLaunch


# ============================================================
# Parser — text output and input file reading
# ============================================================

@runtime_checkable
class Parser(Protocol):
    """Reads a program's text output and input files.

    `parse_output` is the cheap default — it must fit in an agent's context even
    for huge files. Drill-down methods (`get_orbitals`, `get_frequency`,
    `get_trajectory`, `get_thermochem`) load expensive per-task sections on
    demand. `task_index` is the cheapest of all: a single streaming pass that
    returns only the task summary list, with no derived quantities.
    """

    def parse_output(self, path: str) -> ParsedRun:
        """Cheap default: tasks list + flat derived dict + file-level diagnosis."""
        ...

    def task_index(self, path: str) -> list[TaskSummary]:
        """Single-pass scan returning just task summaries — no derived data."""
        ...

    def parse_input(self, path: str) -> dict[str, Any]:
        """Parse the program's input file format into a structured dict."""
        ...

    def get_orbitals(
        self, path: str, task_index: int | None = None
    ) -> dict[str, Any]:
        """MO energies, occupations, symmetries; coefficients if available.

        task_index=None means "use the primary task" (chosen by selection_priority).
        """
        ...

    def get_frequency(
        self, path: str, task_index: int | None = None
    ) -> dict[str, Any]:
        """Normal mode frequencies, IR intensities, (raman intensities if present)."""
        ...

    def get_trajectory(
        self, path: str, task_index: int | None = None
    ) -> dict[str, Any]:
        """Geometry trajectory from an optimization or dynamics task."""
        ...

    def get_thermochem(
        self, path: str, task_index: int | None = None
    ) -> dict[str, Any]:
        """ZPE, thermal corrections, S, Cv, H, G."""
        ...

    def get_geometry(
        self, path: str, task_index: int | None = None
    ) -> list[GeometryAtom]:
        """Extract a geometry snapshot. For optimize/saddle tasks: the converged
        geometry. For energy/frequency tasks: the input geometry."""
        ...


# ============================================================
# BinaryReader — Fortran-unformatted / hash-table / scratch files
# ============================================================

@runtime_checkable
class BinaryReader(Protocol):
    """Reads program-specific binary artifacts.

    The set of supported kinds is program-specific. NWChem starts with
    "movecs", "hessian", "fdrst"; Molpro/Molcas will have their own.
    Programs without binary artifacts (or without readers yet) leave this
    sub-protocol as None on their Program plugin.
    """

    def supported_kinds(self) -> list[str]:
        """List of kind strings this reader knows about."""
        ...

    def parse(self, path: str, kind: str) -> dict[str, Any]:
        """Parse a binary file. Raise ValueError if kind is not supported."""
        ...

    def write(self, path: str, kind: str, data: dict[str, Any]) -> None:
        """Write a binary file (e.g. swapped movecs). Optional — raise
        NotImplementedError if the program's binary reader is read-only."""
        ...


# ============================================================
# Drafter — input file generation, linting, patching
# ============================================================

@runtime_checkable
class Drafter(Protocol):
    """Generates and modifies a program's input files.

    `draft_input` consumes the program-agnostic `InputSpec` (with
    `program_options` as the escape hatch for program-specific knobs).
    `lint_input` returns `LintIssue` records with `suggested_fix` text ready
    for copy-paste. `patch_input` applies structured changes without
    re-drafting from scratch (used by recovery flows).
    """

    def draft_input(self, spec: InputSpec) -> str:
        """Render an input file from a program-neutral InputSpec."""
        ...

    def lint_input(self, text: str) -> list[LintIssue]:
        """Validate an input file's syntax + cross-references."""
        ...

    def patch_input(self, text: str, change: dict[str, Any]) -> str:
        """Apply a structured change. Change shape is program-specific but
        documented per-implementation. Common changes: swap vectors, change
        functional, add solvent block, raise SCF iter limit."""
        ...


@runtime_checkable
class PathInputReviewer(Protocol):
    """Optional path-aware extension for checks that inspect related files."""

    def lint_input_file(self, path: str) -> list[LintIssue]:
        ...


# ============================================================
# Strategist — diagnosis, recovery, resource recommendations
# ============================================================

@runtime_checkable
class Strategist(Protocol):
    """Turns parsed data into agent-actionable judgments.

    `diagnose` is the thick-tool engine — given a parsed run, produce a
    verdict plus next_actions ready for the agent to execute.
    `plan_recovery` is for the failure case specifically: given the source
    files and intended state, what concrete fixes are worth trying.
    `suggest_resources` recommends nodes/ranks/walltime/memory for a runner
    profile. `progress_summary` is for in-flight job monitoring.
    """

    def diagnose(self, parsed: ParsedRun) -> Diagnosis:
        """File-level verdict + next_actions."""
        ...

    def plan_recovery(
        self,
        output_path: str,
        input_path: str | None,
        target: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Build a read-only recovery plan from run files and target state."""
        ...

    def suggest_resources(
        self, input_path: str, profile: dict[str, Any]
    ) -> dict[str, Any]:
        """Recommend nodes, MPI ranks, walltime, memory directives for a job."""
        ...

    def progress_summary(self, output_path: str) -> dict[str, Any]:
        """Snapshot of an in-flight job: phase, iter count, % done estimate."""
        ...


# ============================================================
# ExamplesCorpus — bundled input file templates
# ============================================================

@runtime_checkable
class ExamplesCorpus(Protocol):
    """Searchable corpus of curated example inputs for a program.

    Tag-based discovery for now. Each entry is an `ExampleEntry` pointing at a
    bundled input file. Used by drafters as a template source and by agents
    looking for "show me an example of X" before drafting from scratch.
    """

    def find_example(
        self,
        task: TaskKind | None = None,
        tags: list[str] | None = None,
        methods: list[str] | None = None,
    ) -> ExampleEntry | None:
        """Return the best-matching example, or None if no match."""
        ...

    def list_examples(
        self,
        task: TaskKind | None = None,
        tags: list[str] | None = None,
    ) -> list[ExampleEntry]:
        """Return all examples matching the filter (empty list if none)."""
        ...

    def read_example(self, name: str) -> str:
        """Return the raw text of an example input by name."""
        ...


# ============================================================
# Legacy Program protocol
# ============================================================

@runtime_checkable
class Program(Protocol):
    """Compatibility shape used before `ProgramBackend`.

    Required attributes/methods:
        name                — short identifier, e.g. "nwchem"
        file_extensions     — by-role mapping, e.g.
                              {"input":[".nw"], "output":[".out",".log"],
                               "movecs":[".movecs"], "hessian":[".hess"], ...}
        detect              — given the first ~8KB of a file, is it ours?
        detect_version      — pull "X.Y.Z" out of the banner if present
        parser, drafter, strategist — required sub-protocols

    Optional sub-protocols (set to None if not applicable):
        binary              — Fortran-unformatted / scratch readers
        examples            — bundled example input corpus

    New built-ins use `ProgramBackend`. Keep this protocol until legacy Python
    callers no longer require `drafter`, `strategist`, or `file_extensions`.
    """

    name: str
    file_extensions: dict[str, list[str]]

    parser: Parser
    drafter: Drafter
    strategist: Strategist
    binary: BinaryReader | None
    examples: ExamplesCorpus | None

    def detect(self, output_head: str) -> bool:
        """Sniff the first ~8KB of an output file — does it look like ours?"""
        ...

    def detect_version(self, output_head: str) -> str | None:
        """Extract the program version from the banner, or None."""
        ...


class ProgramCapability(str, Enum):
    OUTPUT_PARSE = "output.parse"
    OUTPUT_TASK_INDEX = "output.task_index"
    OUTPUT_GEOMETRY = "output.geometry"
    OUTPUT_ORBITALS = "output.orbitals"
    OUTPUT_FREQUENCIES = "output.frequencies"
    OUTPUT_TRAJECTORY = "output.trajectory"
    OUTPUT_THERMOCHEMISTRY = "output.thermochemistry"
    INPUT_PARSE = "input.parse"
    INPUT_DRAFT = "input.draft"
    INPUT_LINT = "input.lint"
    INPUT_PATCH = "input.patch"
    BINARY_READ = "binary.read"
    BINARY_WRITE = "binary.write"
    DIAGNOSIS_RUN = "diagnosis.run"
    DIAGNOSIS_RECOVERY = "diagnosis.recovery"
    RESOURCES_ESTIMATE = "resources.estimate"
    PROGRESS_INSPECT = "progress.inspect"
    RUN_CONSISTENCY = "run.consistency"
    CALCULATION_PLAN = "calculation.plan"
    EXECUTION_PLAN = "execution.plan"
    EXAMPLES_READ = "examples.read"


@runtime_checkable
class ProgramDetector(Protocol):
    def detect(self, output_head: str) -> bool:
        ...

    def detect_version(self, output_head: str) -> str | None:
        ...


@runtime_checkable
class DiagnosticAdapter(Protocol):
    def diagnose(self, parsed: ParsedRun) -> Diagnosis:
        ...

    def plan_recovery(
        self,
        output_path: str,
        input_path: str | None,
        target: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


@runtime_checkable
class ResourceAdvisor(Protocol):
    def suggest_resources(
        self, input_path: str, profile: dict[str, Any]
    ) -> dict[str, Any]:
        ...


@runtime_checkable
class ProgressAdapter(Protocol):
    def progress_summary(self, output_path: str) -> dict[str, Any]:
        ...


@runtime_checkable
class RunConsistencyAdapter(Protocol):
    def compare_input_output(
        self,
        input_path: str,
        output_path: str,
        parsed_input: Mapping[str, Any],
        parsed_output: Mapping[str, Any],
        artifact_paths: tuple[str, ...],
    ) -> Mapping[str, Any]:
        ...


@runtime_checkable
class CalculationPlanner(Protocol):
    def plan_calculation(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


@runtime_checkable
class LaunchPlanner(Protocol):
    def prepare_launch(
        self,
        request: Mapping[str, Any],
    ) -> PreparedLaunch:
        ...


@dataclass(frozen=True)
class ArtifactKindSpec:
    extensions: tuple[str, ...] = ()
    filenames: tuple[str, ...] = ()
    default_roles: frozenset[ArtifactRole] = frozenset()
    content_kind: Literal["text", "binary", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        if self.content_kind not in {"text", "binary", "unknown"}:
            raise ValueError(
                "content_kind must be 'text', 'binary', or 'unknown'"
            )
        object.__setattr__(
            self,
            "default_roles",
            frozenset(ArtifactRole(role) for role in self.default_roles),
        )


class UnsupportedCapabilityError(LookupError):
    def __init__(
        self,
        program: str,
        capability: ProgramCapability,
        available_capabilities: frozenset[ProgramCapability],
    ) -> None:
        self.program = program
        self.capability = capability
        self.available_capabilities = tuple(
            sorted(item.value for item in available_capabilities)
        )
        super().__init__(
            f"{program!r} does not support {capability.value!r}; "
            f"available capabilities: {list(self.available_capabilities)}"
        )


class InvalidProgramBackend(ValueError):
    """Raised when a backend declaration contradicts its providers."""


@dataclass(frozen=True)
class ProgramBackend:
    name: str
    capabilities: frozenset[ProgramCapability]
    artifact_kinds: Mapping[str, ArtifactKindSpec]
    detector: ProgramDetector
    parser: Parser | None = None
    inputs: Drafter | None = None
    binary: BinaryReader | None = None
    diagnostics: DiagnosticAdapter | None = None
    resources: ResourceAdvisor | None = None
    progress: ProgressAdapter | None = None
    consistency: RunConsistencyAdapter | None = None
    planning: CalculationPlanner | None = None
    launches: LaunchPlanner | None = None
    examples: ExamplesCorpus | None = None

    def supports(self, capability: ProgramCapability) -> bool:
        return capability in self.capabilities

    @property
    def file_extensions(self) -> dict[str, list[str]]:
        return {
            kind.removeprefix(f"{self.name}."): [
                *spec.extensions,
                *spec.filenames,
            ]
            for kind, spec in self.artifact_kinds.items()
        }

    @property
    def drafter(self) -> Drafter | None:
        return self.inputs

    @property
    def strategist(self) -> Strategist | None:
        if (
            self.diagnostics is not None
            and self.diagnostics is self.resources
            and self.diagnostics is self.progress
        ):
            return self.diagnostics
        return None

    def detect(self, output_head: str) -> bool:
        return self.detector.detect(output_head)

    def detect_version(self, output_head: str) -> str | None:
        return self.detector.detect_version(output_head)

    def require(self, capability: ProgramCapability) -> ProgramBackend:
        if not self.supports(capability):
            raise UnsupportedCapabilityError(
                self.name, capability, self.capabilities
            )
        return self


_CAPABILITY_REQUIREMENTS: dict[
    ProgramCapability, tuple[tuple[str, str], ...]
] = {
    ProgramCapability.OUTPUT_PARSE: (("parser", "parse_output"),),
    ProgramCapability.OUTPUT_TASK_INDEX: (("parser", "task_index"),),
    ProgramCapability.OUTPUT_GEOMETRY: (("parser", "get_geometry"),),
    ProgramCapability.OUTPUT_ORBITALS: (("parser", "get_orbitals"),),
    ProgramCapability.OUTPUT_FREQUENCIES: (("parser", "get_frequency"),),
    ProgramCapability.OUTPUT_TRAJECTORY: (("parser", "get_trajectory"),),
    ProgramCapability.OUTPUT_THERMOCHEMISTRY: (("parser", "get_thermochem"),),
    ProgramCapability.INPUT_PARSE: (("parser", "parse_input"),),
    ProgramCapability.INPUT_DRAFT: (("inputs", "draft_input"),),
    ProgramCapability.INPUT_LINT: (("inputs", "lint_input"),),
    ProgramCapability.INPUT_PATCH: (("inputs", "patch_input"),),
    ProgramCapability.BINARY_READ: (
        ("binary", "supported_kinds"),
        ("binary", "parse"),
    ),
    ProgramCapability.BINARY_WRITE: (("binary", "write"),),
    ProgramCapability.DIAGNOSIS_RUN: (("diagnostics", "diagnose"),),
    ProgramCapability.DIAGNOSIS_RECOVERY: (
        ("diagnostics", "plan_recovery"),
    ),
    ProgramCapability.RESOURCES_ESTIMATE: (
        ("resources", "suggest_resources"),
    ),
    ProgramCapability.PROGRESS_INSPECT: (
        ("progress", "progress_summary"),
    ),
    ProgramCapability.RUN_CONSISTENCY: (
        ("consistency", "compare_input_output"),
    ),
    ProgramCapability.CALCULATION_PLAN: (
        ("planning", "plan_calculation"),
    ),
    ProgramCapability.EXECUTION_PLAN: (
        ("launches", "prepare_launch"),
    ),
    ProgramCapability.EXAMPLES_READ: (
        ("examples", "list_examples"),
        ("examples", "read_example"),
    ),
}


def validate_backend(backend: ProgramBackend) -> ProgramBackend:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", backend.name):
        raise InvalidProgramBackend(
            f"invalid backend name {backend.name!r}; expected lowercase identifier"
        )

    for detector_method in ("detect", "detect_version"):
        if not callable(getattr(backend.detector, detector_method, None)):
            raise InvalidProgramBackend(
                f"backend {backend.name!r} detector.{detector_method} is unavailable"
            )

    if not backend.artifact_kinds:
        raise InvalidProgramBackend(
            f"backend {backend.name!r} must declare at least one artifact kind"
        )
    for kind, spec in backend.artifact_kinds.items():
        if not kind.startswith(f"{backend.name}."):
            raise InvalidProgramBackend(
                f"artifact kind {kind!r} must start with {backend.name + '.'!r}"
            )
        if not isinstance(spec, ArtifactKindSpec):
            raise InvalidProgramBackend(
                f"artifact kind {kind!r} must use ArtifactKindSpec"
            )
        if not spec.extensions and not spec.filenames:
            raise InvalidProgramBackend(
                f"artifact kind {kind!r} has no accepted extension or filename"
            )
        if any(not extension.startswith(".") for extension in spec.extensions):
            raise InvalidProgramBackend(
                f"artifact kind {kind!r} contains an invalid extension"
            )
        if not spec.default_roles or any(not role for role in spec.default_roles):
            raise InvalidProgramBackend(
                f"artifact kind {kind!r} must declare non-empty default roles"
            )

    for capability in backend.capabilities:
        if not isinstance(capability, ProgramCapability):
            raise InvalidProgramBackend(
                f"backend {backend.name!r} has an unknown capability {capability!r}"
            )
        for provider_name, method_name in _CAPABILITY_REQUIREMENTS[capability]:
            provider = getattr(backend, provider_name)
            if not callable(getattr(provider, method_name, None)):
                raise InvalidProgramBackend(
                    f"backend {backend.name!r} declares {capability.value!r} "
                    f"but {provider_name}.{method_name} is unavailable"
                )

    return backend


__all__ = [
    "Parser",
    "BinaryReader",
    "Drafter",
    "PathInputReviewer",
    "Strategist",
    "ExamplesCorpus",
    "Program",
    "ProgramCapability",
    "ProgramDetector",
    "DiagnosticAdapter",
    "ResourceAdvisor",
    "ProgressAdapter",
    "RunConsistencyAdapter",
    "CalculationPlanner",
    "LaunchPlanner",
    "ArtifactKindSpec",
    "UnsupportedCapabilityError",
    "InvalidProgramBackend",
    "ProgramBackend",
    "validate_backend",
]
