"""Compatibility import for the relocated legacy NWChem action builder."""

from chemtools.programs.nwchem.strategy.legacy_next_actions import (
    build_legacy_next_actions as _build_next_actions,
)


__all__ = ["_build_next_actions"]
