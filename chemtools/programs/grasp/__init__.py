"""Validated GRASP2018 backend declaration."""

from __future__ import annotations

from chemtools.core.program import (
    ArtifactKindSpec,
    ProgramBackend,
    ProgramCapability,
    validate_backend,
)
from chemtools.programs.grasp._plugin_parser import GRASP_PARSER
from chemtools.programs.grasp._plugin_binary import GRASP_BINARY
from chemtools.programs.grasp._plugin_launcher import GRASP_LAUNCH_PLANNER
from chemtools.programs.grasp._plugin_planner import GRASP_CALCULATION_PLANNER


_TERMINAL_FAILURE_MARKERS = (
    "Maximum iterations in SCF Exceeded.",
    "Convergence not obtained",
    "BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES",
    "ERROR STOP",
)


def _looks_like_grasp(output_head: str) -> bool:
    chunk = output_head[:8192]
    if "relativistic CSFs" in chunk and "relativistic subshells" in chunk:
        return True
    if "EOL calculation" in chunk:
        return True
    if "GRASP92" in chunk:
        return True
    if "No Pos  J Parity" in chunk or "Energy levels for ..." in chunk:
        return True
    if "Pos   J   Parity" in chunk and "Comp. of ASF" in chunk:
        return True
    if "Nuclear spin" in chunk and (
        "A(MHz)" in chunk or "A (MHz)" in chunk
    ):
        return True
    if (
        "Normal mass shift parameter" in chunk
        and "Specific mass shift parameter" in chunk
    ):
        return True
    if "ANGS(VAC)" in chunk and "AKI =" in chunk:
        return True
    banners = (
        "RMCDHF\n",
        " RMCDHF\n",
        "RCSFGENERATE\n",
        "RWFNESTIMATE\n",
        "RANGULAR\n",
        "RNUCLEUS\n",
        "JJ2LSJ\n",
    )
    return any(banner in chunk for banner in banners)


class _GraspDetector:
    def detect(self, output_head: str) -> bool:
        return _looks_like_grasp(output_head)

    def detect_version(self, output_head: str) -> str | None:
        return "2018" if self.detect(output_head) else None


GRASP = validate_backend(
    ProgramBackend(
        name="grasp",
        capabilities=frozenset(
            {
                ProgramCapability.OUTPUT_PARSE,
                ProgramCapability.OUTPUT_TASK_INDEX,
                ProgramCapability.OUTPUT_ORBITALS,
                ProgramCapability.BINARY_READ,
                ProgramCapability.BINARY_WRITE,
                ProgramCapability.CALCULATION_PLAN,
                ProgramCapability.EXECUTION_PLAN,
            }
        ),
        artifact_kinds={
            "grasp.stdin_capture": ArtifactKindSpec(
                extensions=(".in",),
                default_roles=frozenset({"auxiliary_input"}),
                content_kind="text",
            ),
            "grasp.nuclear_data": ArtifactKindSpec(
                filenames=("isodata",),
                default_roles=frozenset({"auxiliary_input"}),
                content_kind="text",
            ),
            "grasp.csf_input": ArtifactKindSpec(
                filenames=("rcsf.inp",),
                default_roles=frozenset({"auxiliary_input"}),
                content_kind="text",
            ),
            "grasp.radial_wfn_seed": ArtifactKindSpec(
                filenames=("rwfn.inp",),
                default_roles=frozenset({"orbital", "wavefunction_seed"}),
                content_kind="binary",
            ),
            "grasp.radial_wfn_output": ArtifactKindSpec(
                filenames=("rwfn.out",),
                default_roles=frozenset({"orbital", "wavefunction"}),
                content_kind="binary",
            ),
            "grasp.mixing_output": ArtifactKindSpec(
                filenames=("rmix.out",),
                default_roles=frozenset({"wavefunction"}),
                content_kind="binary",
            ),
            "grasp.rmcdhf_input_log": ArtifactKindSpec(
                filenames=("rmcdhf.log",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "grasp.rci_input_log": ArtifactKindSpec(
                extensions=(".clog",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "grasp.rmcdhf_summary": ArtifactKindSpec(
                extensions=(".sum",),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "grasp.rci_summary": ArtifactKindSpec(
                extensions=(".csum",),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "grasp.hfs": ArtifactKindSpec(
                extensions=(".h", ".ch", ".hlsj", ".chlsj"),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "grasp.isotope_shift": ArtifactKindSpec(
                extensions=(".i", ".ci"),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "grasp.transition": ArtifactKindSpec(
                extensions=(".t.lsj", ".ct.lsj"),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
            ),
            "grasp.lsj_label": ArtifactKindSpec(
                extensions=(".lsj.lbl",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "grasp.mixing": ArtifactKindSpec(
                extensions=(".m", ".cm"),
                default_roles=frozenset({"wavefunction"}),
                content_kind="binary",
            ),
            "grasp.csf_list": ArtifactKindSpec(
                extensions=(".c",),
                default_roles=frozenset({"auxiliary_output"}),
                content_kind="text",
            ),
            "grasp.radial_wfn": ArtifactKindSpec(
                extensions=(".w",),
                default_roles=frozenset({"orbital", "wavefunction"}),
                content_kind="binary",
            ),
            "grasp.scf_log": ArtifactKindSpec(
                extensions=(".log", ".alog"),
                default_roles=frozenset({"stdout"}),
                content_kind="text",
            ),
            "grasp.output": ArtifactKindSpec(
                extensions=(".out", ".stdout"),
                default_roles=frozenset({"primary_output"}),
                content_kind="text",
                failure_markers=_TERMINAL_FAILURE_MARKERS,
            ),
            "grasp.error": ArtifactKindSpec(
                extensions=(".err", ".stderr"),
                default_roles=frozenset({"stderr"}),
                content_kind="text",
                failure_markers=_TERMINAL_FAILURE_MARKERS,
            ),
        },
        detector=_GraspDetector(),
        parser=GRASP_PARSER,
        binary=GRASP_BINARY,
        launches=GRASP_LAUNCH_PLANNER,
        planning=GRASP_CALCULATION_PLANNER,
    )
)


__all__ = ["GRASP"]
