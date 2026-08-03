"""Expose QMCPACK XML checks through the backend input contract."""

from __future__ import annotations

from chemtools.core.types import LintIssue
from chemtools.programs.qmcpack.input import (
    lint_qmcpack_input,
    parse_qmcpack_input,
)
from chemtools.programs.qmcpack.includes import (
    include_issues,
    included_xml_lint_issues,
    inspect_xml_includes,
)
from chemtools.programs.qmcpack.sidecars import (
    hdf5_sidecar_issues,
    inspect_hdf5_sidecars,
)


class _QmcpackInputs:
    def lint_input(self, text: str) -> list[LintIssue]:
        return lint_qmcpack_input(text)

    def lint_input_file(self, path: str) -> list[LintIssue]:
        with open(path, encoding="utf-8", errors="replace") as handle:
            issues = lint_qmcpack_input(handle.read())
        if any(issue["level"] == "error" for issue in issues):
            return issues
        parsed = parse_qmcpack_input(path)
        include_review = inspect_xml_includes(path, parsed)
        return [
            *issues,
            *include_issues(include_review),
            *included_xml_lint_issues(include_review),
            *hdf5_sidecar_issues(
                inspect_hdf5_sidecars(path, parsed, include_review)
            ),
        ]


QMCPACK_INPUTS = _QmcpackInputs()


__all__ = ["QMCPACK_INPUTS"]
