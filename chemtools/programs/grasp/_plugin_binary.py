"""GRASP2018 binary artifact provider."""

from __future__ import annotations

from typing import Any

from chemtools.programs.grasp.binary import (
    inspect_grasp_mixing,
    inspect_grasp_radial_wfn,
    merge_grasp_radial_wfns,
)


class _GraspBinaryReader:
    def supported_kinds(self) -> list[str]:
        return ["radial_wfn", "mixing"]

    def parse(self, path: str, kind: str) -> dict[str, Any]:
        if kind == "radial_wfn":
            return inspect_grasp_radial_wfn(path)
        if kind == "mixing":
            return inspect_grasp_mixing(path)
        raise ValueError(
            "GRASP BinaryReader does not support "
            f"kind={kind!r}; supported: {self.supported_kinds()}"
        )

    def write(self, path: str, kind: str, data: dict[str, Any]) -> None:
        if kind == "radial_wfn":
            merge_grasp_radial_wfns(
                data["donor_paths"],
                path,
                overwrite=data.get("overwrite", False),
            )
            return
        raise NotImplementedError(
            "GRASP BinaryReader write does not support "
            f"kind={kind!r}; supported: {self.supported_kinds()}"
        )


GRASP_BINARY = _GraspBinaryReader()

__all__ = ["GRASP_BINARY"]
