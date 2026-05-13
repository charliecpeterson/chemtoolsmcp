"""GRASP2018 program plugin.

Importing this package registers the plugin with chemtools.core.registry.
GRASP is structurally different from the other programs — there's no single
"input file" or "output file". Instead it produces a family of artifacts
written by each step in the workflow:

  * isodata                 — nuclear params from rnucleus
  * rcsf.{out,inp}          — CSF list from rcsfgenerate
  * mcp.30..mcp.39          — angular integrals from rangular
  * rwfn.{out,inp}          — radial orbitals from rwfnestimate
  * rmcdhf.{sum,log}        — SCF summary + input log from rmcdhf
  * <name>.{w,c,m,sum}      — final results from rsave
  * <name>.lsj.lbl          — LSJ-coupled composition from jj2lsj

For Program.detect/parse_output, the canonical "output" target is the
rmcdhf summary file (``*.sum`` or ``rmcdhf.sum``) — that's where the final
SCF energy + per-subshell orbital data ends up.
"""

from __future__ import annotations

from chemtools.core import registry


def _looks_like_grasp(head: str) -> bool:
    """Detect a GRASP output/summary file from its first few KB.

    Looks for the joint markers that uniquely identify GRASP2018:
      * "relativistic CSFs" + "relativistic subshells" (rmcdhf .sum files)
      * "EOL calculation" (rmcdhf summary)
      * "RMCDHF\n", "RCSFGENERATE\n", "RWFNESTIMATE\n" banners (run logs)
      * "GRASP92 File" / "GRASP format" mentions in stdout
    """
    chunk = head[:8192]
    if "relativistic CSFs" in chunk and "relativistic subshells" in chunk:
        return True
    if "EOL calculation" in chunk:
        return True
    if "GRASP92" in chunk or "GRASP format" in chunk:
        return True
    # rlevels stdout
    if "No Pos  J Parity" in chunk or "Energy levels for ..." in chunk:
        return True
    # jj2lsj .lsj.lbl header
    if "Pos   J   Parity" in chunk and "Comp. of ASF" in chunk:
        return True
    # Banner headers from individual run logs
    banners = ("RMCDHF\n", " RMCDHF\n", "RCSFGENERATE\n", "RWFNESTIMATE\n",
               "RANGULAR\n", "RNUCLEUS\n", "JJ2LSJ\n")
    return any(b in chunk for b in banners)


def _parse_grasp_version(head: str) -> str | None:
    """GRASP2018 outputs don't print a version banner. Return the fixed
    GRASP2018 tag if the detector confirms this is a GRASP run."""
    return "2018" if _looks_like_grasp(head) else None


class _GraspPlugin:
    """GRASP2018 plugin instance. Currently parser-only (the rest of the
    sub-protocols don't fit GRASP's many-exe model and aren't planned)."""

    name: str = "grasp"

    file_extensions: dict[str, list[str]] = {
        # Each entry maps a logical "kind" to the file suffixes GRASP writes.
        "rmcdhf_summary":  [".sum"],
        "lsj_label":       [".lsj.lbl"],
        "mixing":          [".m", ".cm"],
        "csf_list":        [".c"],
        "radial_wfn":      [".w"],
        "scf_log":         [".log", ".alog"],
        "output":          [".out", ".sum"],  # canonical text-output kind
    }

    # Sub-protocols — wired below.
    parser = None
    drafter = None
    strategist = None
    binary = None
    examples = None

    def detect(self, output_head: str) -> bool:
        return _looks_like_grasp(output_head)

    def detect_version(self, output_head: str) -> str | None:
        return _parse_grasp_version(output_head)


GRASP = _GraspPlugin()

# Wire the parser sub-protocol (split into its own module like the others).
from chemtools.programs.grasp._plugin_parser import GRASP_PARSER as _GRASP_PARSER  # noqa: E402

GRASP.parser = _GRASP_PARSER

registry.register(GRASP)


__all__ = ["GRASP"]
