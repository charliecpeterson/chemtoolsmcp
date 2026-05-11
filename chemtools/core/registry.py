"""Program plugin registry and program-detection helpers.

A program is registered by calling `register(plugin)` from its module's
__init__. CLI entry points (`chemtools-<name>`) import the program module they
serve, which triggers registration; the MCP tool layer then dispatches through
`get(name)` or, when the agent doesn't specify a program, through
`detect_from_file(path)`.

No I/O at module-load time. Files are only opened when an agent actually calls
a detection or dispatch function.
"""

from __future__ import annotations
from typing import Iterable

from chemtools.core.program import Program


_REGISTRY: dict[str, Program] = {}
_DETECT_HEAD_BYTES: int = 8192


class ProgramNotRegistered(KeyError):
    """Raised when a requested program name is not in the registry."""


class ProgramDetectionFailed(ValueError):
    """Raised when auto-detection fails to identify the program from a file."""


def register(plugin: Program) -> None:
    """Register a program plugin under its `name`. Overwrites any prior entry."""
    _REGISTRY[plugin.name] = plugin


def unregister(name: str) -> None:
    """Remove a program from the registry (mainly for tests)."""
    _REGISTRY.pop(name, None)


def get(name: str) -> Program:
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


def iter_programs() -> Iterable[Program]:
    return _REGISTRY.values()


def detect_from_text(head: str) -> str | None:
    """Sniff the first chunk of an output file's text — return program name or None."""
    for name, plugin in _REGISTRY.items():
        try:
            if plugin.detect(head):
                return name
        except Exception:
            continue
    return None


def detect_from_file(path: str) -> str | None:
    """Read the first ~8KB of `path` and dispatch to `detect_from_text`."""
    try:
        with open(path, "rb") as f:
            raw = f.read(_DETECT_HEAD_BYTES)
    except OSError:
        return None
    head = raw.decode("utf-8", errors="replace")
    return detect_from_text(head)


def resolve(program: str | None, path: str | None = None) -> Program:
    """Resolve a program by explicit name, or by detecting from `path`.

    Convenience for MCP tool dispatchers:

        plugin = registry.resolve(program, path=output_file)
        return plugin.parser.parse_output(output_file)
    """
    if program is not None:
        return get(program)
    if path is None:
        raise ProgramDetectionFailed(
            "Cannot resolve program: no name provided and no file to sniff."
        )
    detected = detect_from_file(path)
    if detected is None:
        raise ProgramDetectionFailed(
            f"Could not auto-detect a program from {path!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return get(detected)


__all__ = [
    "ProgramNotRegistered",
    "ProgramDetectionFailed",
    "register",
    "unregister",
    "get",
    "has",
    "list_programs",
    "iter_programs",
    "detect_from_text",
    "detect_from_file",
    "resolve",
]
