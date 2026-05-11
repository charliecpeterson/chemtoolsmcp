"""Plugin protocol for per-program implementations.

A `Program` is a plugin instance assembled in `chemtools/programs/<name>/__init__.py`.
It bundles sub-protocols that match the planned directory layout:

    programs/<name>/parse.py       -> Parser
    programs/<name>/binary.py      -> BinaryReader  (optional)
    programs/<name>/input.py       -> Drafter
    programs/<name>/strategy.py    -> Strategist
    programs/<name>/examples.py    -> ExamplesCorpus (optional)

Each sub-protocol is `@runtime_checkable`, so the MCP layer can do
`isinstance(plugin.parser, Parser)` capability checks before registering tools.

A program does not have to implement every method on every sub-protocol —
unsupported methods raise `NotImplementedError` and the MCP wrapper turns that
into a clean "not supported for this program" tool error.

Registration: `chemtools/programs/<name>/__init__.py` calls
    chemtools.core.registry.register(PLUGIN)
on import. The CLI entry point (`chemtools-<name>`) imports the program module,
triggering registration, then registers MCP tools that dispatch through the
registry.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Any

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


# ============================================================
# Strategist — diagnosis, recovery, resource recommendations
# ============================================================

@runtime_checkable
class Strategist(Protocol):
    """Turns parsed data into agent-actionable judgments.

    `diagnose` is the thick-tool engine — given a parsed run, produce a
    verdict plus next_actions ready for the agent to execute.
    `suggest_recovery` is for the failure case specifically: when something
    went wrong, what concrete fixes are worth trying, in priority order.
    `suggest_resources` recommends nodes/ranks/walltime/memory for a runner
    profile. `progress_summary` is for in-flight job monitoring.
    """

    def diagnose(self, parsed: ParsedRun) -> Diagnosis:
        """File-level verdict + next_actions."""
        ...

    def suggest_recovery(
        self, parsed: ParsedRun, diagnosis: Diagnosis
    ) -> list[NextAction]:
        """Ordered list of recovery actions for a failed/incomplete run."""
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
# Program — the plugin instance that bundles the sub-protocols
# ============================================================

@runtime_checkable
class Program(Protocol):
    """A program plugin instance.

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

    Programs are registered with the global registry in
    `chemtools/programs/<name>/__init__.py` so importing the program module
    is the only side effect needed to make it discoverable.
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


__all__ = [
    "Parser",
    "BinaryReader",
    "Drafter",
    "Strategist",
    "ExamplesCorpus",
    "Program",
]
