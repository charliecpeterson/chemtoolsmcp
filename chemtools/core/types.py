"""Cross-program data shapes for chemtoolsmcp.

Everything that crosses an MCP tool boundary is expressed in terms of these
types. They are TypedDicts (not dataclasses) so they serialize cleanly to JSON
over JSON-RPC and so optional fields stay friendly.

Design conventions baked in here:

1. parse_output returns are SMALL by default. The default ParsedRun fits in an
   agent's context even for huge output files — it is a summary plus pre-computed
   derived quantities. Heavy sections (MO coefficients, full trajectories,
   normal-mode eigenvectors) load via separate drill-down tools:
       get_orbitals(path, task_index=...)
       get_frequency_modes(path, task_index=...)
       get_trajectory(path, task_index=...)
       get_thermochem(path, task_index=...)

2. Any tool whose purpose is to drive agent action returns a Diagnosis envelope
   (verdict + next_actions + anchors). This is the "thick tool" contract — a
   small LLM should be able to execute next_actions[0] without further reasoning.

3. Diagnostics carry line anchors. When a parser flags an error or warning, it
   records the line number so the agent can quote the exact source. The
   DiagnosticAnchor type is the same shape across all programs.

4. Selection priority. When an output file has multiple tasks, each TaskSummary
   carries a selection_priority (opt=3 > freq=2 > single_point=1) so tools that
   need to pick "the main task" can do so deterministically.
"""

from __future__ import annotations
from typing import TypedDict, Literal, Any


# ============================================================
# Verdicts and next actions — the "thick tool" envelope
# ============================================================

class NextAction(TypedDict, total=False):
    """A concrete action an agent should take, with parameters ready to use.

    A thick tool returns these so the agent doesn't have to compute its own
    next step. priority orders them (lower runs first); ties broken by list order.
    """
    tool: str                       # MCP tool name to call
    params: dict[str, Any]          # arguments ready to pass through
    reason: str                     # one-sentence rationale
    confidence: float               # 0.0-1.0
    priority: int                   # 1 = highest


class Verdict(TypedDict, total=False):
    """A labeled judgment about the state of something (a run, an input, an MO set).

    label is the machine-friendly tag ("converged", "scf_failed_at_iter_42",
    "imaginary_modes_present"). reasons are short bullets in human language.
    """
    label: str
    confidence: float
    reasons: list[str]


class DiagnosticAnchor(TypedDict, total=False):
    """A line-anchored note tied to a specific spot in an output or input file."""
    kind: Literal["error", "warning", "info"]
    message: str
    line: int | None                # 1-based line number; None if file-level
    file: str | None                # absolute path; None = the current file


class Diagnosis(TypedDict, total=False):
    """Standard envelope for any analysis tool that should drive agent action.

    Tools that produce a Diagnosis are "thick" — the agent reads verdict,
    runs next_actions[0], and uses anchors only if it needs to dig deeper.
    """
    verdict: Verdict
    next_actions: list[NextAction]
    anchors: list[DiagnosticAnchor]


# ============================================================
# Task / output summaries
# ============================================================

TaskKind = Literal[
    "energy",       # task <method> energy   — single-point energy of any method
    "gradient",     # task <method> gradient — energy + nuclear gradient
    "optimize",     # task <method> optimize — geometry optimization to a minimum
    "saddle",       # task <method> saddle   — transition-state optimization
    "frequency",    # task <method> frequency — vibrational analysis (raman intensities,
                    #   if requested, surface via derived["raman_intensities"] = True)
    "property",     # task <method> property — multipoles, NMR, polarizabilities, etc.
    "dynamics",     # MD / BOMD
    "unknown",
]
# Note on the kind × method split:
#
#   kind  = what is being computed (energy, optimize, frequency, ...)
#   method = how it is being computed ("DFT (PBE0)", "CCSD(T)", "TDDFT/B3LYP",
#            "CASSCF(8,8)", ...)
#
# In NWChem terms, `task <method> <operation>` decomposes directly:
#   task dft energy      ->  kind="energy",    method="DFT"
#   task ccsd(t) energy  ->  kind="energy",    method="CCSD(T)"
#   task dft optimize    ->  kind="optimize",  method="DFT"
#   task tddft energy    ->  kind="energy",    method="TDDFT"
# Molpro and Molcas follow the same conceptual split.

TaskOutcome = Literal[
    "success",      # task completed cleanly
    "failed",       # task printed an error / aborted
    "incomplete",   # task started but never finished (still running, or crashed without error message)
    "unknown",
]


class TaskSummary(TypedDict, total=False):
    """One task within an output file.

    NWChem, Molpro, and Molcas all support multi-task input files; this is the
    common shape for "what tasks did this run perform". has_usable_data is
    independent of outcome — a freq that crashed at mode 5/30 still has 4 modes.
    """
    index: int                      # 0-based position within the file
    kind: TaskKind                  # operation only — see TaskKind note above
    name: str                       # human-friendly: "Optimization", "CCSD(T) Energy"
    method: str | None              # "DFT (PBE0)", "CCSD(T)", "CASSCF(8,8)", "TDDFT/B3LYP"
    basis: str | None               # "def2-TZVP"
    energy_hartree: float | None
    line_range: tuple[int, int]     # 1-based inclusive [start, end]
    outcome: TaskOutcome
    has_usable_data: bool
    selection_priority: int         # optimize/saddle=3, frequency=2, energy=1
                                    # higher = preferred auto-pick when picking a "main" task


# ============================================================
# parse_output return shape — small by default
# ============================================================

class ParsedRun(TypedDict, total=False):
    """Cheap, default return from parse_<program>_output.

    Stays small enough to fit in an agent's context even for huge output files.
    Drill into expensive sections via dedicated tools — see module docstring.

    derived holds pre-computed scalars an agent typically wants without
    re-deriving: homo_lumo_ev, n_imaginary_modes, final_energy_hartree,
    n_atoms, final_geometry_xyz, walltime_used_sec, etc. Programs may add
    their own keys; namespaced keys (e.g. "nwchem:tce_freeze_count") are fine.
    """
    program: str                    # "nwchem" | "molcas" | "dirac" | "grasp" | ...
    program_version: str | None
    file: str                       # absolute path to the output file
    file_size_bytes: int

    tasks: list[TaskSummary]
    primary_task_index: int | None  # auto-picked via selection_priority

    derived: dict[str, Any]         # pre-computed flat key/value pairs
    diagnostics: list[DiagnosticAnchor]
    diagnosis: Diagnosis            # file-level verdict + next_actions


# ============================================================
# Input drafting (flat InputSpec)
# ============================================================

class GeometryAtom(TypedDict):
    element: str                    # element symbol
    x: float
    y: float
    z: float


class InputSpec(TypedDict, total=False):
    """Program-neutral input specification consumed by program.draft_input.

    Flat by design — easier for MCP tool calls (JSON), and program-specific
    knobs go in program_options rather than being shoehorned into nested types.
    """
    # Required-ish (every drafter expects these)
    atoms: list[GeometryAtom]
    charge: int
    multiplicity: int
    method: str                     # "DFT", "HF", "CCSD(T)", "CASSCF", "TDDFT", ...
    basis: str | dict[str, str]     # "def2-TZVP" OR per-element {"H": "...", "Fe": "..."}
    task: TaskKind                  # operation only; see TaskKind note above.
                                    # The method is specified separately in `method`.

    # Common optional knobs (drafters use what applies)
    title: str
    geometry_units: Literal["angstrom", "bohr"]
    functional: str                 # for DFT
    ecp: dict[str, str]             # per-element effective core potential names
    solvent: dict[str, Any]         # {"model": "cosmo", "epsilon": 78.4}

    # Program-specific extras — drafter is responsible for translating these.
    # Examples: NWChem {"tce": {"freeze": 4, "vectors": "..."}},
    #           Molpro {"casscf": {"closed": 4, "occ": 8}}.
    program_options: dict[str, Any]


# ============================================================
# Lint issues
# ============================================================

class LintIssue(TypedDict, total=False):
    """One issue found by program.lint_input.

    suggested_fix is a patched text fragment ready to drop in (not a unified diff
    — keep it copy-paste friendly for small LLMs).
    """
    level: Literal["error", "warning", "info"]
    message: str
    line: int | None
    suggested_fix: str | None


# ============================================================
# Examples corpus (tag-based)
# ============================================================

class ExampleEntry(TypedDict, total=False):
    """One entry in a program's bundled example corpus.

    Tag-based discovery for now: program.find_example(task_type, tags=[...])
    filters and returns the best match. Future versions may add embedding search.
    """
    name: str                       # short identifier, e.g. "fe_porphyrin_casscf_opt"
    task_type: TaskKind
    methods: list[str]              # ["DFT", "B3LYP"] — any reasonable label
    basis: list[str]                # ["def2-TZVP"]
    tags: list[str]                 # ["transition_metal", "open_shell", "solvent"]
    description: str                # one-sentence summary
    file: str                       # path relative to the program's examples dir
    notes: str | None               # author commentary, gotchas, expected runtime


__all__ = [
    "NextAction",
    "Verdict",
    "DiagnosticAnchor",
    "Diagnosis",
    "TaskKind",
    "TaskOutcome",
    "TaskSummary",
    "ParsedRun",
    "GeometryAtom",
    "InputSpec",
    "LintIssue",
    "ExampleEntry",
]
