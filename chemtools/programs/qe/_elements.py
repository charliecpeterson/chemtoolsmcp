"""Normalize Quantum ESPRESSO species labels to chemical elements."""

from __future__ import annotations

from chemtools.core.common import ATOMIC_SYMBOLS


_ELEMENTS = frozenset(ATOMIC_SYMBOLS.values())


def element_from_label(label: str) -> str | None:
    cleaned = label.strip()
    for width in (2, 1):
        candidate = cleaned[:width].capitalize()
        if candidate in _ELEMENTS:
            return candidate
    return None


__all__ = ["element_from_label"]
