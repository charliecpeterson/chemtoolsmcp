"""Molcas / OpenMolcas program plugin.

Importing this package registers the plugin with chemtools.core.registry.
Currently a stub: only parse_tasks is implemented (lifted from the legacy
chemtools/molcas.py); the Parser, Drafter, Strategist sub-protocols will be
filled in as Molcas support deepens.
"""

from __future__ import annotations

from chemtools.core import registry


class _MolcasPlugin:
    """Minimal Molcas plugin instance — currently parse-only and partial."""

    name: str = "molcas"

    file_extensions: dict[str, list[str]] = {
        "input":   [".input", ".inp"],
        "output":  [".out", ".log"],
        "runfile": [".RunFile"],
        "orbitals": ["INPORB", "JOBIPH"],
    }

    # Sub-protocols — to be filled in.
    parser = None
    drafter = None
    strategist = None
    binary = None
    examples = None

    def detect(self, output_head: str) -> bool:
        upper = output_head[:8192].upper()
        return (
            "THIS RUN OF MOLCAS IS USING THE PYMOLCAS DRIVER" in upper
            or "OPENMOLCASOP" in upper
            or "DEFINITIONS: _MOLCAS_" in upper
        )

    def detect_version(self, output_head: str) -> str | None:
        return None


MOLCAS = _MolcasPlugin()

from chemtools.programs.molcas._plugin_parser import MOLCAS_PARSER as _MOLCAS_PARSER  # noqa: E402
MOLCAS.parser = _MOLCAS_PARSER

registry.register(MOLCAS)


__all__ = ["MOLCAS"]
