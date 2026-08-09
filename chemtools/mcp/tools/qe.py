"""Quantum ESPRESSO MCP entry module.

Conversion tools live in ``qe_qmcpack`` so this backend entry point stays
small while preserving the catalog and import contract.
"""

from chemtools.mcp.tools.qe_qmcpack import *  # noqa: F401,F403
from chemtools.mcp.tools.qe_qmcpack_definitions import (
    qe_tool_definitions as _qe_conversion_tool_definitions,
)


def qe_tool_definitions() -> list[dict[str, object]]:
    return _qe_conversion_tool_definitions()
