"""DIRAC program plugin.

Importing this package registers the plugin with chemtools.core.registry.
Phase DA ships parser + auto-detect; the binary reader (HDF5 checkpoints)
arrives in Phase DB and the drafter / strategist in DD/DE/DF.
"""

from __future__ import annotations

from chemtools.core import registry
from chemtools.programs.dirac.parse.output import (
    looks_like_dirac as _looks_like_dirac,
    parse_version as _parse_version,
)


class _DiracPlugin:
    name: str = "dirac"

    file_extensions: dict[str, list[str]] = {
        "input":     [".inp"],
        "molecule":  [".mol"],
        "output":    [".out", ".log"],
        "checkpoint": [".h5"],
        "orbitals":   ["DFCOEF", "DFPCMO", "DFACMO"],
    }

    parser = None
    drafter = None
    strategist = None
    binary = None
    examples = None

    def detect(self, output_head: str) -> bool:
        return _looks_like_dirac(output_head[:8192])

    def detect_version(self, output_head: str) -> str | None:
        return _parse_version(output_head[:5000])


DIRAC = _DiracPlugin()

from chemtools.programs.dirac._plugin_parser import DIRAC_PARSER as _DIRAC_PARSER  # noqa: E402
from chemtools.programs.dirac._plugin_binary import DIRAC_BINARY as _DIRAC_BINARY  # noqa: E402

DIRAC.parser = _DIRAC_PARSER
DIRAC.binary = _DIRAC_BINARY

registry.register(DIRAC)


__all__ = ["DIRAC"]
