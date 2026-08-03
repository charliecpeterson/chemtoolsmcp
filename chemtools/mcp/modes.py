"""Server modes: capability-tag filtering for the MCP tool list.

Modes:
  analysis — pure parsing/drafting/planning. No NWChem executable, no scheduler.
  local    — NWChem runs as a foreground subprocess via a "direct" runner profile.
  hpc      — NWChem submitted to a scheduler via a "scheduler" runner profile.

Each registered tool carries a capability tag set through the ``needs=`` kwarg
on ``@_tool`` in the MCP tool modules. The active mode defines which tags are
exposed at ``tools/list`` time; blocked tools are also refused at
``tools/call`` time with an explanatory error.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from chemtools.mcp.catalog import builtin_program_names

VALID_MODES = ("analysis", "local", "hpc")

MODE_CAPABILITIES: dict[str, frozenset[str]] = {
    "analysis": frozenset({"none", "registry"}),
    "local": frozenset({
        "none", "registry", "runner_profile",
        "executable_or_scheduler", "executable",
    }),
    "hpc": frozenset({
        "none", "registry", "runner_profile",
        "executable_or_scheduler", "executable", "scheduler",
    }),
}

# "generic" is treated specially and is not a program backend.
KNOWN_PROGRAMS = builtin_program_names()

# Env var that selects mode explicitly (overrides auto-detect).
MODE_ENV = "CHEMTOOLS_MODE"
# Env var pointing at the user's runner profiles file (drives auto-detect).
PROFILES_ENV = "CHEMTOOLS_RUNNER_PROFILES"
# Env var that selects which programs to load (comma-separated).
PROGRAMS_ENV = "CHEMTOOLS_PROGRAMS"
# Env var that selects a curated tool subset (a preset name, or a comma-separated
# list of tool names). Trims the surface for small models / focused workflows.
TOOLSET_ENV = "CHEMTOOLS_TOOLSET"

# Named tool presets. "triage" is the lean output-assessment set: enough to scan
# a batch, diagnose a file, and follow up, without the full draft/registry/HPC
# surface a small model would otherwise have to choose from.
TOOLSETS: dict[str, frozenset[str]] = {
    "guided": frozenset({
        "review_input",
        "inspect_run",
        "search_knowledge_cards",
    }),
    "triage": frozenset({
        "summarize_nwchem_outputs",          # batch triage entry point
        "analyze_nwchem_case",               # deep single-file diagnosis
        "summarize_nwchem_output",           # quick single-file summary
        "parse_nwchem_output",               # structured sections on demand
        "suggest_nwchem_recovery",           # what to do about a failure
        "suggest_nwchem_multiplicity_scan",  # verify spin ground state
        "analyze_nwchem_frontier_orbitals",  # is the open shell on the metal?
        "check_nwchem_spin_charge_state",    # spin/charge sanity
        "extract_nwchem_geometry",           # pull the geometry
        "parse_nwchem_thermochem",           # thermochemistry
        "compare_nwchem_runs",               # compare energies across runs
        "get_server_mode",                   # introspect what's available
    }),
    "molcas-triage": frozenset({
        "summarize_molcas_outputs",          # batch triage entry point
        "analyze_molcas_case",               # deep single-file diagnosis + verdict
        "summarize_molcas_output",           # quick single-file summary
        "parse_molcas_output",               # structured per-module detail
        "suggest_molcas_recovery",           # what to do about a failure
        "analyze_molcas_active_space",       # CAS quality
        "validate_molcas_caspt2_setup",      # CASPT2 reference-weight / intruders
        "parse_molcas_frequencies",          # frequencies / imaginary modes
        "parse_molcas_thermochem",           # thermochemistry
        "extract_molcas_geometry",           # pull the geometry
        "get_server_mode",                   # introspect what's available
    }),
    "dirac-triage": frozenset({
        "summarize_dirac_outputs",           # batch triage entry point
        "summarize_dirac_run",               # deep single-file rollup
        "parse_dirac_output",                # structured SCF/spinor detail
        "parse_dirac_scf_iterations",        # convergence trace
        "parse_dirac_spinor_spectrum",       # spinor eigenvalues
        "analyze_dirac_open_shell",          # open-shell / AOC quality
        "parse_dirac_cosci_energies",        # open-shell CI states
        "read_dirac_h5_metadata",            # checkpoint metadata
        "suggest_relativistic_correction",   # 4c / X2C / ECP advice
        "get_server_mode",                   # introspect what's available
    }),
    "grasp-triage": frozenset({
        "summarize_grasp_runs",              # batch triage entry point (per working dir)
        "analyze_grasp_case",                # deep single working-dir audit
        "parse_grasp_sum",                   # nucleus/grid/subshell summary
        "parse_grasp_lsjlbl",                # LSJ-coupled levels
        "parse_grasp_levels",                # rlevels energy table
        "summarize_grasp_terms",             # term grouping
        "compare_grasp_levels",              # DHF vs non-rel comparison
        "parse_grasp_rmcdhf_log",            # SCF convergence trace
        "suggest_grasp_recovery",            # failure classification
        "get_server_mode",                   # introspect what's available
    }),
}


def filter_tools(
    definitions: list[dict[str, Any]],
    capabilities: dict[str, str],
    mode: str,
    programs: Iterable[str] | None = None,
    program_tags: dict[str, str] | None = None,
    toolset: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Return tool defs visible under *mode* and the optional program/toolset filters.

    A tool is visible iff:
      1. Its capability tag is in MODE_CAPABILITIES[mode], AND
      2. Either:
         - programs is None (no filter — show all programs' tools), OR
         - the tool's program tag is "generic", OR
         - the tool's program tag is in the *programs* set, AND
      3. toolset is None, or the tool's name is in *toolset* (an exact allowlist
         applied after the program filter — generics are NOT auto-included).
    """
    allowed_caps = MODE_CAPABILITIES[mode]
    program_set: set[str] | None = None
    if programs is not None:
        program_set = {p.strip().lower() for p in programs if p and p.strip()}

    out: list[dict[str, Any]] = []
    for d in definitions:
        name = d["name"]
        if capabilities.get(name, "none") not in allowed_caps:
            continue
        if program_set is not None and program_tags is not None:
            tag = program_tags.get(name, "generic")
            if tag != "generic" and tag not in program_set:
                continue
        if toolset is not None and name not in toolset:
            continue
        out.append(d)
    return out


def is_tool_allowed(
    name: str,
    capabilities: dict[str, str],
    mode: str,
    programs: Iterable[str] | None = None,
    program_tags: dict[str, str] | None = None,
    toolset: frozenset[str] | None = None,
) -> bool:
    if capabilities.get(name, "none") not in MODE_CAPABILITIES[mode]:
        return False
    if toolset is not None and name not in toolset:
        return False
    if programs is None or program_tags is None:
        return True
    tag = program_tags.get(name, "generic")
    if tag == "generic":
        return True
    program_set = {p.strip().lower() for p in programs if p and p.strip()}
    return tag in program_set


def resolve_programs(
    explicit: str | Iterable[str] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> tuple[set[str] | None, str]:
    """Resolve the active program filter.

    Returns ``(programs_set, reason)``. ``programs_set`` is None if no filter
    is active (all programs visible). Otherwise it's a set of program names.

    Priority:
      1. *explicit* — from --programs CLI flag (string or iterable)
      2. CHEMTOOLS_PROGRAMS env var (comma-separated)
      3. None (no filter)
    """
    env = env if env is not None else os.environ

    def _normalize(value: str | Iterable[str]) -> set[str]:
        if isinstance(value, str):
            tokens = [t.strip() for t in value.split(",") if t.strip()]
        else:
            tokens = [str(t).strip() for t in value if str(t).strip()]
        return {t.lower() for t in tokens}

    if explicit is not None:
        if isinstance(explicit, str) and not explicit.strip():
            return None, "explicit empty — no filter"
        programs = _normalize(explicit)
        if not programs:
            return None, "explicit empty — no filter"
        unknown = programs - set(KNOWN_PROGRAMS)
        if unknown:
            # We don't hard-fail on unknown program names — a future program
            # plugin might register itself with a name we don't know about
            # at module-load time. Warn via the reason string.
            return programs, (
                f"set by --programs={','.join(sorted(programs))} "
                f"(unrecognized: {sorted(unknown)})"
            )
        return programs, f"set by --programs={','.join(sorted(programs))}"

    env_val = env.get(PROGRAMS_ENV)
    if env_val:
        programs = _normalize(env_val)
        if programs:
            return programs, f"set by {PROGRAMS_ENV}={','.join(sorted(programs))}"

    return None, f"no {PROGRAMS_ENV} set — all programs visible"


def resolve_toolset(
    explicit: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> tuple[frozenset[str] | None, str]:
    """Resolve the active tool-name allowlist.

    Returns ``(names, reason)``. ``names`` is None for no filter. The value is a
    preset name from TOOLSETS, or a comma-separated list of tool names for an
    ad-hoc set. Priority: *explicit* (--toolset) then CHEMTOOLS_TOOLSET.
    """
    env = env if env is not None else os.environ
    raw = explicit if explicit is not None else env.get(TOOLSET_ENV)
    if not raw or not raw.strip():
        return None, f"no {TOOLSET_ENV} set — full tool surface"
    key = raw.strip().lower()
    if key in TOOLSETS:
        return TOOLSETS[key], f"preset {key!r} ({len(TOOLSETS[key])} tools)"
    names = frozenset(t.strip() for t in raw.split(",") if t.strip())
    return names, f"custom list ({len(names)} tools)"


def resolve_mode(
    explicit: str | None = None,
    *,
    profiles_path: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Pick a mode using the documented priority order.

    Returns ``(mode, reason)`` where *reason* is a short human-readable string
    explaining how the mode was chosen — surfaced at startup for transparency.

    Priority:
      1. *explicit* argument (from --mode CLI flag)
      2. ``CHEMTOOLS_MODE`` env var
      3. auto-detect:
         - no ``CHEMTOOLS_RUNNER_PROFILES`` set       → analysis
         - profiles file unreadable or empty          → analysis (with warning in reason)
         - any profile has launcher.kind=="scheduler" → hpc
         - otherwise                                  → local
    """
    env = env if env is not None else os.environ

    if explicit:
        if explicit not in VALID_MODES:
            raise ValueError(f"unknown mode {explicit!r}; expected one of {VALID_MODES}")
        return explicit, f"set by --mode={explicit}"

    env_mode = env.get(MODE_ENV)
    if env_mode:
        if env_mode not in VALID_MODES:
            raise ValueError(f"{MODE_ENV}={env_mode!r}; expected one of {VALID_MODES}")
        return env_mode, f"set by {MODE_ENV}={env_mode}"

    profiles_path = profiles_path or env.get(PROFILES_ENV)
    if not profiles_path:
        return "analysis", f"auto: {PROFILES_ENV} not set"

    kinds = _profile_launcher_kinds(profiles_path)
    if kinds is None:
        return "analysis", f"auto: could not read {profiles_path}"
    if not kinds:
        return "analysis", f"auto: no profiles found in {profiles_path}"
    if "scheduler" in kinds:
        return "hpc", f"auto: scheduler profile present in {profiles_path}"
    return "local", f"auto: only direct profiles in {profiles_path}"


def _profile_launcher_kinds(path: str) -> set[str] | None:
    """Return the set of launcher.kind values found in a profiles file, or None on failure.

    Parses JSON natively; for YAML, requires PyYAML (returns None if unavailable).
    """
    source = Path(path)
    if not source.is_file():
        return None
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        return None

    payload: Any
    if source.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            # Try JSON sidecar, same behavior as runner.load_runner_profiles
            sidecar = source.with_suffix(".json")
            if sidecar.is_file():
                return _profile_launcher_kinds(str(sidecar))
            return None
        try:
            payload = yaml.safe_load(text)
        except Exception:
            return None

    if not isinstance(payload, dict):
        return None

    # Profile schema nests entries under `profiles:` (see
    # runner_profiles.example.json + runner.load_runner_profiles). Older
    # ad-hoc files may put profiles at the top level, so try both shapes.
    if isinstance(payload.get("profiles"), dict):
        profile_map = payload["profiles"]
    else:
        profile_map = payload

    kinds: set[str] = set()
    for key, profile in profile_map.items():
        if key.startswith("__") or not isinstance(profile, dict):
            continue
        launcher = profile.get("launcher")
        if isinstance(launcher, dict):
            kind = launcher.get("kind")
            if isinstance(kind, str):
                kinds.add(kind)
    return kinds


def summarize_mode(
    mode: str,
    capabilities: dict[str, str],
    definitions: list[dict[str, Any]],
    programs: Iterable[str] | None = None,
    program_tags: dict[str, str] | None = None,
    toolset: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Compact summary used by the get_server_mode tool and startup logging."""
    allowed = MODE_CAPABILITIES[mode]
    available = filter_tools(
        definitions, capabilities, mode,
        programs=programs, program_tags=program_tags, toolset=toolset,
    )
    blocked_by_tag: dict[str, list[str]] = {}
    for name, tag in capabilities.items():
        if tag not in allowed:
            blocked_by_tag.setdefault(tag, []).append(name)
    for names in blocked_by_tag.values():
        names.sort()

    summary = {
        "mode": mode,
        "allowed_capability_tags": sorted(allowed),
        "available_tool_count": len(available),
        "total_tool_count": len(definitions),
        "blocked_tools_by_tag": blocked_by_tag,
    }
    if programs is not None and program_tags is not None:
        program_set = {p.strip().lower() for p in programs if p and p.strip()}
        blocked_by_program: dict[str, list[str]] = {}
        for name, tag in program_tags.items():
            if tag == "generic":
                continue
            if tag not in program_set:
                blocked_by_program.setdefault(tag, []).append(name)
        for names in blocked_by_program.values():
            names.sort()
        summary["active_programs"] = sorted(program_set)
        summary["blocked_tools_by_program"] = blocked_by_program
    return summary
