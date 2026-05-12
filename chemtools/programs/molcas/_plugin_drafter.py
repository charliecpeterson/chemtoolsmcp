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


class _MolcasDrafter:
    def draft_input(self, spec: InputSpec) -> str:
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
