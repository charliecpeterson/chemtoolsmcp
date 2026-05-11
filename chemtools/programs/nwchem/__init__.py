"""NWChem program plugin.

Assembled from submodules in `chemtools/programs/nwchem/{parse,binary,input,
strategy,examples}/`. Importing this package registers the plugin with
`chemtools.core.registry` so `registry.detect_from_file` and
`registry.resolve` can find it.

The sub-protocol attributes (`parser`, `drafter`, `strategist`, `binary`,
`examples`) are initialized to `None` during the multi-program refactor and
will be replaced with real implementations as code is moved into the
submodules. Until then, callers that need NWChem functionality continue to
import from the legacy flat modules (`chemtools.nwchem_tasks`, etc.).
"""

from __future__ import annotations
import re

from chemtools.core import registry


class _NwchemPlugin:
    """Minimal NWChem plugin instance — to be filled in as the refactor lands."""

    name: str = "nwchem"

    file_extensions: dict[str, list[str]] = {
        "input":         [".nw", ".nwi"],
        "output":        [".out", ".nwo", ".log"],
        "error":         [".err"],
        "movecs":        [".movecs"],
        "hessian":       [".hess"],
        "freq_restart":  [".fdrst"],
        "trajectory":    [".xyz"],
        "jobid":         [".jobid"],
        "scratch":       [".db", ".rmd"],
        "normal_modes":  [".normal", ".nmode"],
    }

    # Sub-protocols — filled in as code moves into the submodules.
    # (parser is assigned below, after the class is defined and after the
    # parser module is imported, to avoid an import-cycle with parser code
    # that may itself import this package.)
    parser = None
    drafter = None
    strategist = None
    binary = None
    examples = None

    _VERSION_RE = re.compile(r"NWChem(?:\s+version)?\s+(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)

    def detect(self, output_head: str) -> bool:
        upper = output_head.upper()
        return (
            "NORTHWEST COMPUTATIONAL CHEMISTRY PACKAGE" in upper
            or "NWCHEM" in upper
            # Earliest reliable NWChem-specific signal — appears at the top of
            # every output, before the banner. Useful when the head window
            # doesn't reach the banner because the input echo is large.
            or "ECHO OF INPUT DECK" in upper
        )

    def detect_version(self, output_head: str) -> str | None:
        match = self._VERSION_RE.search(output_head)
        return match.group(1) if match else None


NWCHEM = _NwchemPlugin()

# Wire up sub-protocols after the plugin instance exists. Imports are kept
# inside this block so a consumer that touches `chemtools.programs.nwchem`
# only pays for what it imports.
from chemtools.programs.nwchem._plugin_parser import NWCHEM_PARSER as _NWCHEM_PARSER  # noqa: E402
from chemtools.programs.nwchem._plugin_strategist import NWCHEM_STRATEGIST as _NWCHEM_STRATEGIST  # noqa: E402
NWCHEM.parser = _NWCHEM_PARSER
NWCHEM.strategist = _NWCHEM_STRATEGIST

registry.register(NWCHEM)


__all__ = ["NWCHEM"]
