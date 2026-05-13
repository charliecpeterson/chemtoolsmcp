"""Server modes: capability-tag filtering for the MCP tool list.

Modes:
  analysis — pure parsing/drafting/planning. No NWChem executable, no scheduler.
  local    — NWChem runs as a foreground subprocess via a "direct" runner profile.
  hpc      — NWChem submitted to a scheduler via a "scheduler" runner profile.

Each registered tool carries a capability tag (set via the ``needs=`` kwarg on
``@_tool`` in ``chemtools/mcp/nwchem.py``). The active mode defines which tags
are exposed at ``tools/list`` time; blocked tools are also refused at
``tools/call`` time with an explanatory error.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

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

# Known program tags. "generic" is treated specially — generic tools are
# always visible regardless of the --programs filter.
KNOWN_PROGRAMS = ("nwchem", "molcas", "dirac", "grasp")

# Env var that selects mode explicitly (overrides auto-detect).
MODE_ENV = "CHEMTOOLS_MODE"
# Env var pointing at the user's runner profiles file (drives auto-detect).
PROFILES_ENV = "CHEMTOOLS_RUNNER_PROFILES"
# Env var that selects which programs to load (comma-separated).
PROGRAMS_ENV = "CHEMTOOLS_PROGRAMS"


def filter_tools(
    definitions: list[dict[str, Any]],
    capabilities: dict[str, str],
    mode: str,
    programs: Iterable[str] | None = None,
    program_tags: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return tool defs visible under *mode* and the optional *programs* filter.

    A tool is visible iff:
      1. Its capability tag is in MODE_CAPABILITIES[mode], AND
      2. Either:
         - programs is None (no filter — show all programs' tools), OR
         - the tool's program tag is "generic", OR
         - the tool's program tag is in the *programs* set.
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
        out.append(d)
    return out


def is_tool_allowed(
    name: str,
    capabilities: dict[str, str],
    mode: str,
    programs: Iterable[str] | None = None,
    program_tags: dict[str, str] | None = None,
) -> bool:
    if capabilities.get(name, "none") not in MODE_CAPABILITIES[mode]:
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

    kinds: set[str] = set()
    for key, profile in payload.items():
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
) -> dict[str, Any]:
    """Compact summary used by the get_server_mode tool and startup logging."""
    allowed = MODE_CAPABILITIES[mode]
    available = filter_tools(
        definitions, capabilities, mode,
        programs=programs, program_tags=program_tags,
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
