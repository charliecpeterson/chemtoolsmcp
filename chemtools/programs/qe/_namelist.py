"""Small Fortran-namelist reader shared by bounded QE post-processing inputs.

It retains only scalar assignments and deliberately does not model full QE syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_NAMELIST_RE = re.compile(r"^\s*&([A-Za-z][A-Za-z0-9_]*)\b(.*)$")
_ASSIGNMENT_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=\s*(.*?)\s*$"
)


@dataclass(frozen=True)
class Namelist:
    """A compact view of one named Fortran namelist."""

    line: int | None
    closed: bool
    next_line_index: int | None
    values: dict[str, Any]


def has_namelist(text: str, name: str) -> bool:
    """Return whether text contains the requested namelist declaration."""
    expected = name.casefold()
    return any(
        (match := _NAMELIST_RE.match(strip_fortran_comment(line))) is not None
        and match.group(1).casefold() == expected
        for line in text.splitlines()
    )


def parse_namelist(text: str, name: str) -> Namelist:
    """Read scalar assignments from the first requested namelist in text."""
    lines = text.splitlines()
    expected = name.casefold()
    for index, line in enumerate(lines):
        opening = _NAMELIST_RE.match(strip_fortran_comment(line))
        if opening is None or opening.group(1).casefold() != expected:
            continue
        values: dict[str, Any] = {}
        pending = opening.group(2)
        current_index = index
        while True:
            clean = strip_fortran_comment(pending).strip()
            if clean == "/":
                return Namelist(index + 1, True, current_index + 1, values)
            for fragment in split_unquoted(clean, ","):
                assignment = _ASSIGNMENT_RE.match(fragment)
                if assignment is not None:
                    values[assignment.group(1).lower()] = parse_scalar(
                        assignment.group(2)
                    )
            current_index += 1
            if current_index >= len(lines):
                return Namelist(index + 1, False, None, values)
            pending = lines[current_index]
    return Namelist(None, False, None, {})


def parse_scalar(value: str) -> Any:
    """Convert quoted strings and logical literals, retaining other scalar text."""
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    if stripped.casefold() in {".true.", "true"}:
        return True
    if stripped.casefold() in {".false.", "false"}:
        return False
    return stripped


def strip_fortran_comment(line: str) -> str:
    """Remove a Fortran ``!`` comment without splitting quoted text."""
    quote: str | None = None
    for index, character in enumerate(line):
        if character in {"'", '"'}:
            quote = None if quote == character else character if quote is None else quote
        elif character == "!" and quote is None:
            return line[:index]
    return line


def split_unquoted(value: str, delimiter: str) -> list[str]:
    """Split text at a delimiter outside quoted strings."""
    parts: list[str] = []
    start = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {"'", '"'}:
            quote = None if quote == character else character if quote is None else quote
        elif character == delimiter and quote is None:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


__all__ = ["Namelist", "has_namelist", "parse_namelist", "strip_fortran_comment"]
