"""Adapter exposing pw.x checks through the backend input contract."""

from __future__ import annotations

from chemtools.core.types import LintIssue
from chemtools.programs.qe.charge_spin import (
    charge_spin_issues,
    inspect_charge_spin,
)
from chemtools.programs.qe.input import (
    lint_pw_input,
    parse_pw_input,
    parse_pw_text,
    unsupported_qe_program,
)
from chemtools.programs.qe.input_geometry import (
    analyze_pw_input_geometry,
    input_geometry_issues,
)
from chemtools.programs.qe.kpoints import inspect_k_points, k_point_issues
from chemtools.programs.qe.phonon import is_ph_x_input, lint_ph_x_input
from chemtools.programs.qe.pw2qmcpack import (
    is_pw2qmcpack_input,
    lint_pw2qmcpack_input,
)
from chemtools.programs.qe.pseudopotentials import (
    inspect_input_pseudopotentials,
    pseudopotential_issues,
)


class _QeInputs:
    def lint_input(self, text: str) -> list[LintIssue]:
        if is_ph_x_input(text):
            return lint_ph_x_input(text)  # type: ignore[return-value]
        if is_pw2qmcpack_input(text):
            return lint_pw2qmcpack_input(text)  # type: ignore[return-value]
        if unsupported_qe_program(text) is not None:
            return lint_pw_input(text)
        parsed = parse_pw_text(text)
        charge_spin = inspect_charge_spin(parsed)
        k_points = inspect_k_points(parsed)
        geometry = analyze_pw_input_geometry(parsed)
        return [
            *lint_pw_input(text),
            *charge_spin_issues(charge_spin),
            *k_point_issues(k_points),
            *input_geometry_issues(parsed, geometry),
        ]

    def lint_input_file(self, path: str) -> list[LintIssue]:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        if is_ph_x_input(text):
            return lint_ph_x_input(text)  # type: ignore[return-value]
        if is_pw2qmcpack_input(text):
            return lint_pw2qmcpack_input(text)  # type: ignore[return-value]
        if unsupported_qe_program(text) is not None:
            return lint_pw_input(text)
        issues = lint_pw_input(text)
        parsed = parse_pw_input(path)
        pseudo_review = inspect_input_pseudopotentials(path, parsed)
        charge_spin = inspect_charge_spin(parsed, pseudo_review)
        k_points = inspect_k_points(parsed)
        geometry = analyze_pw_input_geometry(parsed)
        return [
            *issues,
            *pseudopotential_issues(pseudo_review),
            *charge_spin_issues(charge_spin),
            *k_point_issues(k_points),
            *input_geometry_issues(parsed, geometry),
        ]


QE_INPUTS = _QeInputs()


__all__ = ["QE_INPUTS"]
