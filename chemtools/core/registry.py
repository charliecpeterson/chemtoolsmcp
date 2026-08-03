"""Program backend registry and program-detection helpers.

The MCP composition catalog registers built-in backends explicitly. Program
packages only export backend objects, so importing one cannot mutate this
registry. The MCP tool layer dispatches through `resolve()`, which retains
detector and source-read failures. The `detect_from_*` compatibility helpers
continue to return only successful matches.

No I/O at module-load time. Files are only opened when an agent actually calls
a detection or dispatch function.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from chemtools.core.program import Program, ProgramBackend, validate_backend


_REGISTRY: dict[str, Program | ProgramBackend] = {}
# 32KB — large enough to catch the NWChem program banner after a typical
# input-deck echo (which can run 10-20KB), small enough to stay cheap for
# huge output files.
_DETECT_HEAD_BYTES: int = 32 * 1024


class ProgramNotRegistered(KeyError):
    """Raised when a requested program name is not in the registry."""


class ProgramAlreadyRegistered(ValueError):
    """Raised when registration would replace an existing program."""


class ProgramDetectionFailed(ValueError):
    """Raised when auto-detection fails to identify the program from a file."""


class ProgramDetectionAmbiguous(ProgramDetectionFailed):
    """Raised when auto-detection identifies more than one program."""

    def __init__(self, path: str, candidates: tuple[str, ...]) -> None:
        self.path = path
        self.candidates = candidates
        super().__init__(
            f"Could not auto-detect one program from {path!r}; content "
            f"matches multiple registered programs: {list(candidates)}. "
            "Pass program explicitly."
        )


class ProgramContentMismatch(ValueError):
    """Raised when an explicit program conflicts with detected content."""

    def __init__(
        self,
        path: str,
        program: str,
        candidates: tuple[str, ...],
    ) -> None:
        self.path = path
        self.program = program
        self.candidates = candidates
        super().__init__(
            f"run output content matches {', '.join(candidates)}, but "
            f"program override selected {program}"
        )


@dataclass(frozen=True)
class ProgramDetectorFailure:
    program: str
    error_type: str
    message: str


@dataclass(frozen=True)
class ProgramSourceFailure:
    error_type: str
    message: str
    errno: int | None


@dataclass(frozen=True)
class ProgramDetectionProbe:
    candidates: tuple[str, ...]
    detector_failures: tuple[ProgramDetectorFailure, ...] = ()
    source_failure: ProgramSourceFailure | None = None


class ProgramDetectorError(ProgramDetectionFailed):
    """Raised when a detector fails during authoritative resolution."""

    def __init__(
        self,
        path: str,
        failures: tuple[ProgramDetectorFailure, ...],
        candidates: tuple[str, ...],
    ) -> None:
        self.path = path
        self.failures = failures
        self.candidates = candidates
        summary = "; ".join(
            f"{failure.program} ({failure.error_type}: {failure.message})"
            for failure in failures
        )
        super().__init__(
            f"Could not safely resolve a program from {path!r}; detector "
            f"failure(s): {summary}. Successful candidates: {list(candidates)}."
        )


class ProgramDetectionSourceError(ProgramDetectionFailed):
    """Raised when authoritative resolution cannot read its source file."""

    def __init__(self, path: str, failure: ProgramSourceFailure) -> None:
        self.path = path
        self.failure = failure
        super().__init__(
            f"Could not read program-detection source {path!r}: "
            f"{failure.error_type}: {failure.message}"
        )


def register(plugin: Program | ProgramBackend) -> None:
    """Register a program under its name, validating current backend objects."""
    if isinstance(plugin, ProgramBackend):
        validate_backend(plugin)
    if plugin.name in _REGISTRY:
        raise ProgramAlreadyRegistered(
            f"A program is already registered as {plugin.name!r}"
        )
    _REGISTRY[plugin.name] = plugin


def unregister(name: str) -> None:
    """Remove a program from the registry (mainly for tests)."""
    _REGISTRY.pop(name, None)


def get(name: str) -> Program | ProgramBackend:
    """Return the program plugin registered under `name`."""
    if name not in _REGISTRY:
        raise ProgramNotRegistered(
            f"No program registered as {name!r}; "
            f"available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def has(name: str) -> bool:
    return name in _REGISTRY


def list_programs() -> list[str]:
    return sorted(_REGISTRY)


def iter_programs() -> Iterable[Program | ProgramBackend]:
    return _REGISTRY.values()


def detect_from_text(head: str) -> str | None:
    """Sniff the first chunk of an output file's text — return program name or None."""
    candidates = detect_candidates_from_text(head)
    return candidates[0] if candidates else None


def detect_candidates_from_text(head: str) -> tuple[str, ...]:
    """Return successful matches, suppressing detector errors for compatibility."""
    return probe_from_text(head).candidates


def probe_from_text(head: str) -> ProgramDetectionProbe:
    """Run every detector while retaining successful matches and failures."""
    candidates = []
    failures = []
    for name, plugin in _REGISTRY.items():
        try:
            if plugin.detect(head):
                candidates.append(name)
        except Exception as error:
            failures.append(ProgramDetectorFailure(
                program=name,
                error_type=type(error).__name__,
                message=str(error),
            ))
    return ProgramDetectionProbe(
        candidates=tuple(candidates),
        detector_failures=tuple(failures),
    )


def detect_from_file(path: str) -> str | None:
    """Read the bounded head of `path` and return the first program match."""
    candidates = detect_candidates_from_file(path)
    return candidates[0] if candidates else None


def detect_candidates_from_file(path: str) -> tuple[str, ...]:
    """Return successful file matches, suppressing read and detector errors."""
    return probe_from_file(path).candidates


def probe_from_file(path: str) -> ProgramDetectionProbe:
    """Read a bounded file head and retain source and detector failures."""
    try:
        with open(path, "rb") as f:
            raw = f.read(_DETECT_HEAD_BYTES)
    except OSError as error:
        return ProgramDetectionProbe(
            candidates=(),
            source_failure=ProgramSourceFailure(
                error_type=type(error).__name__,
                message=str(error),
                errno=error.errno,
            ),
        )
    head = raw.decode("utf-8", errors="replace")
    return probe_from_text(head)


def _raise_strict_probe_failures(
    path: str,
    probe: ProgramDetectionProbe,
    *,
    selected_program: str | None = None,
) -> None:
    if probe.source_failure is not None:
        raise ProgramDetectionSourceError(path, probe.source_failure)
    failures = probe.detector_failures
    if selected_program is not None:
        failures = tuple(
            failure
            for failure in failures
            if failure.program == selected_program
        )
    if failures:
        raise ProgramDetectorError(path, failures, probe.candidates)


def resolve(
    program: str | None, path: str | None = None
) -> Program | ProgramBackend:
    """Resolve a program by explicit name, or by detecting from `path`.

    Convenience for MCP tool dispatchers:

        plugin = registry.resolve(program, path=output_file)
        return plugin.parser.parse_output(output_file)
    """
    if program is not None:
        plugin = get(program)
        if path is not None:
            probe = probe_from_file(path)
            _raise_strict_probe_failures(
                path,
                probe,
                selected_program=program,
            )
            if probe.candidates and program not in probe.candidates:
                raise ProgramContentMismatch(path, program, probe.candidates)
        return plugin
    if path is None:
        raise ProgramDetectionFailed(
            "Cannot resolve program: no name provided and no file to sniff."
        )
    probe = probe_from_file(path)
    _raise_strict_probe_failures(path, probe)
    if not probe.candidates:
        raise ProgramDetectionFailed(
            f"Could not auto-detect a program from {path!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    if len(probe.candidates) > 1:
        raise ProgramDetectionAmbiguous(path, probe.candidates)
    return get(probe.candidates[0])


__all__ = [
    "ProgramNotRegistered",
    "ProgramAlreadyRegistered",
    "ProgramDetectionFailed",
    "ProgramDetectionAmbiguous",
    "ProgramContentMismatch",
    "ProgramDetectorFailure",
    "ProgramSourceFailure",
    "ProgramDetectionProbe",
    "ProgramDetectorError",
    "ProgramDetectionSourceError",
    "register",
    "unregister",
    "get",
    "has",
    "list_programs",
    "iter_programs",
    "detect_from_text",
    "detect_candidates_from_text",
    "probe_from_text",
    "detect_from_file",
    "detect_candidates_from_file",
    "probe_from_file",
    "resolve",
]
