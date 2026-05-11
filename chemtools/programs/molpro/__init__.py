"""Molpro program plugin.

Importing this package registers the plugin with chemtools.core.registry.
Currently a stub: parse_tasks and parse_mos are implemented (lifted from
the legacy chemtools/molpro.py); the rest of the sub-protocols will be
filled in as Molpro support deepens.
"""

from __future__ import annotations
import re

from chemtools.core import registry


class _MolproPlugin:
    """Minimal Molpro plugin instance — currently parse-only and partial."""

    name: str = "molpro"

    file_extensions: dict[str, list[str]] = {
        "input":   [".com", ".inp"],
        "output":  [".out", ".log"],
        "xml":     [".xml"],
        "wfu":     [".wfu"],
    }

    # Sub-protocols — to be filled in.
    parser = None
    drafter = None
    strategist = None
    binary = None
    examples = None

    _VERSION_RE = re.compile(r"Version\s+(\d{4}\.\d+(?:\.\d+)?)", re.IGNORECASE)

    def detect(self, output_head: str) -> bool:
        upper = output_head[:8192].upper()
        return (
            "PROGRAM SYSTEM MOLPRO" in upper
            or "***  PROGRAM SYSTEM MOLPRO  ***" in upper
        )

    def detect_version(self, output_head: str) -> str | None:
        match = self._VERSION_RE.search(output_head)
        return match.group(1) if match else None


MOLPRO = _MolproPlugin()
registry.register(MOLPRO)


__all__ = ["MOLPRO"]
