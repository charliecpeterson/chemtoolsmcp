"""Molcas BinaryReader sub-protocol implementation.

Currently supported `kind` values (read-only):

  * "inporb"    -> parse INPORB / RasOrb / ScfOrb / GssOrb / LprOrb / SpdOrb
                    (named-section text format, not binary in the strict sense
                    but lives here because it is non-output orbital data).
"""

from __future__ import annotations
from typing import Any

from chemtools.programs.molcas.binary.orbitals import (
    parse_inporb,
    swap_orbitals_in_inporb,
)


_SUPPORTED_READ_KINDS = ("inporb",)
_SUPPORTED_WRITE_KINDS = ("inporb_swap",)


class _MolcasBinaryReader:
    """Implements chemtools.core.program.BinaryReader for Molcas."""

    def supported_kinds(self) -> list[str]:
        return list(_SUPPORTED_READ_KINDS)

    def parse(self, path: str, kind: str) -> dict[str, Any]:
        if kind == "inporb":
            return parse_inporb(path, parse_coefficients=True)
        raise ValueError(
            f"Molcas BinaryReader does not support kind={kind!r}; "
            f"supported: {_SUPPORTED_READ_KINDS}"
        )

    def write(self, path: str, kind: str, data: dict[str, Any]) -> None:
        """For `kind="inporb_swap"`, `data` must include:
            input_path: source INPORB / RasOrb
            swaps: list of (orb_i, orb_j) 1-indexed orbital pairs
            symmetry: 1-indexed irrep (default 1)
        """
        if kind == "inporb_swap":
            swap_orbitals_in_inporb(
                input_path=data["input_path"],
                output_path=path,
                swaps=[tuple(s) for s in data["swaps"]],
                symmetry=int(data.get("symmetry", 1)),
            )
            return
        raise NotImplementedError(
            f"Molcas BinaryReader write does not support kind={kind!r}; "
            f"supported: {_SUPPORTED_WRITE_KINDS}"
        )


MOLCAS_BINARY = _MolcasBinaryReader()


__all__ = ["MOLCAS_BINARY"]
