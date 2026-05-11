"""NWChem BinaryReader sub-protocol implementation.

Dispatches by `kind` to the right binary-format reader. Currently
supported kinds:

  * "hessian"   -> parse_nwchem_hessian (ASCII lower-triangle Cartesian
                    Hessian in `.hess`)
  * "movecs"    -> parse_nwchem_movecs (Fortran-unformatted MO file —
                    lives in parse/tce.py for historical reasons)

Write-side support for "movecs" goes through swap_nwchem_movecs.

Planned additions:
  * "drv_hessian" — driver-module binary Hessian (`.drv.hess`)
  * "fdrst"       — frequency restart bookkeeping
"""

from __future__ import annotations
from typing import Any

from chemtools.programs.nwchem.binary.hessian import parse_nwchem_hessian


_SUPPORTED_READ_KINDS = ("hessian", "movecs")
_SUPPORTED_WRITE_KINDS = ("movecs",)


class _NwchemBinaryReader:
    """Implements chemtools.core.program.BinaryReader for NWChem."""

    def supported_kinds(self) -> list[str]:
        return list(_SUPPORTED_READ_KINDS)

    def parse(self, path: str, kind: str) -> dict[str, Any]:
        if kind == "hessian":
            return parse_nwchem_hessian(path)
        if kind == "movecs":
            # Lazy import — keeps binary/__init__ light when only one kind
            # is needed and avoids a circular import via TCE.
            from chemtools.programs.nwchem.parse.tce import parse_nwchem_movecs
            return parse_nwchem_movecs(path)
        raise ValueError(
            f"NWChem BinaryReader does not support kind={kind!r}; "
            f"supported: {_SUPPORTED_READ_KINDS}"
        )

    def write(self, path: str, kind: str, data: dict[str, Any]) -> None:
        if kind == "movecs":
            # The existing swap operation writes a modified movecs file in place.
            # `data` is expected to carry the swap_pairs; see swap_nwchem_movecs.
            from chemtools.programs.nwchem.parse.tce import swap_nwchem_movecs
            swap_pairs = data.get("swap_pairs") or []
            swap_nwchem_movecs(path, swap_pairs)
            return
        raise NotImplementedError(
            f"NWChem BinaryReader write is not implemented for kind={kind!r}; "
            f"supported: {_SUPPORTED_WRITE_KINDS}"
        )


NWCHEM_BINARY = _NwchemBinaryReader()


__all__ = ["NWCHEM_BINARY"]
