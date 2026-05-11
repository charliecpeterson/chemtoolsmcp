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

# Env var that selects mode explicitly (overrides auto-detect).
MODE_ENV = "CHEMTOOLS_MODE"
# Env var pointing at the user's runner profiles file (drives auto-detect).
PROFILES_ENV = "CHEMTOOLS_RUNNER_PROFILES"


def filter_tools(
    definitions: list[dict[str, Any]],
    capabilities: dict[str, str],
    mode: str,
) -> list[dict[str, Any]]:
    """Return only the tool defs whose capability tag is allowed in *mode*."""
    allowed = MODE_CAPABILITIES[mode]
    return [d for d in definitions if capabilities.get(d["name"], "none") in allowed]


def is_tool_allowed(name: str, capabilities: dict[str, str], mode: str) -> bool:
    return capabilities.get(name, "none") in MODE_CAPABILITIES[mode]


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
) -> dict[str, Any]:
    """Compact summary used by the get_server_mode tool and startup logging."""
    allowed = MODE_CAPABILITIES[mode]
    available = filter_tools(definitions, capabilities, mode)
    blocked_by_tag: dict[str, list[str]] = {}
    for name, tag in capabilities.items():
        if tag not in allowed:
            blocked_by_tag.setdefault(tag, []).append(name)
    for names in blocked_by_tag.values():
        names.sort()
    return {
        "mode": mode,
        "allowed_capability_tags": sorted(allowed),
        "available_tool_count": len(available),
        "total_tool_count": len(definitions),
        "blocked_tools_by_tag": blocked_by_tag,
    }
