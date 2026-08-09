"""Molcas Drafter sub-protocol implementation.

Wraps chemtools.programs.molcas.input.draft.draft_molcas_input and
chemtools.programs.molcas.input.lint.lint_molcas_input behind the standard
Drafter Protocol.

`patch_input` is not yet implemented — the agent's typical edit cycle is
"redraft from spec" rather than "patch existing text", so this can wait.
"""

from __future__ import annotations

from typing import Any

from chemtools.core.types import InputSpec, LintIssue
from chemtools.programs.molcas.input.draft import draft_molcas_input as _draft
from chemtools.programs.molcas.input.lint import lint_molcas_input as _lint


_PROGRAM_OPTION_KEYS = {
    "cas_active_electrons",
    "cas_active_orbitals",
    "caspt2",
    "cholesky",
    "expert",
    "inline_basis",
    "memory_mb",
    "n_basis_per_symmetry",
    "n_symmetries",
    "occupied_per_symmetry",
    "pkthrs",
    "rasscf",
    "ricd",
    "scf",
    "seward_extra_keywords",
    "symmetry",
}


class _MolcasDrafter:
    def draft_input(self, spec: InputSpec) -> str:
        options = spec.get("program_options") or {}
        unknown_options = sorted(set(options) - _PROGRAM_OPTION_KEYS)
        if unknown_options:
            raise ValueError(
                "Unsupported OpenMolcas program_options: "
                + ", ".join(unknown_options)
            )
        if spec.get("task", "energy") != "energy":
            raise ValueError(
                "OpenMolcas InputSpec drafting currently supports "
                "task='energy' only"
            )
        if spec.get("ecp"):
            raise ValueError(
                "OpenMolcas InputSpec ECP rendering is not implemented"
            )
        if spec.get("solvent") is not None:
            raise ValueError(
                "OpenMolcas InputSpec solvent rendering is not implemented"
            )
        return _draft(dict(spec))

    def lint_input(self, text: str) -> list[LintIssue]:
        return _lint(text)  # type: ignore[return-value]

    def patch_input(self, text: str, change: dict[str, Any]) -> str:
        raise NotImplementedError(
            "Molcas Drafter patch_input not yet implemented. "
            "Use draft_input with a fresh InputSpec instead."
        )


MOLCAS_DRAFTER = _MolcasDrafter()


__all__ = ["MOLCAS_DRAFTER"]
