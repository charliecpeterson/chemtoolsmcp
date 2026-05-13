"""Molcas launch helpers: safe-command builder + scratch isolation.

Born out of the May 2026 dogfood (see feedback_molcas_dogfood memory). Two
classes of issue this module guards against:

  * Some Molcas builds segfault in CASPT2's distributed-amplitude code path
    when run with ``-np > 1`` (Global Arrays / ARMCI binding mismatch with
    the host MPI). The runner profile carries
    ``execution.parallel_caspt2_supported`` (default True); when False and
    the input contains ``&CASPT2``, the launcher forces ``MOLCAS_NPROCS=1``
    and emits a warning.

  * Molcas writes scratch under ``/tmp/<Project>`` (or wherever
    ``MOLCAS_WORKDIR`` points). The runfile records the launch's nProcs.
    Re-running with a different ``-np`` aborts with ``RunHdr%nProcs/=nProcs``.
    To avoid this, the launcher sets a unique ``MOLCAS_PROJECT`` per launch
    (default = input-file stem; callers can override).

Public API:

    prepare_launch(input_path, profile, requested_np, job_name=None,
                   apptainer_sif=None) -> dict

The returned dict has keys:
    command            list[str] ready to pass to subprocess
    command_str        joined shell-friendly string
    env                env-var additions for the launch
    requested_np       integer the caller asked for
    effective_np       integer that will actually be used
    project            the MOLCAS_PROJECT value set
    has_caspt2         whether the input contains an &CASPT2 block
    parallel_caspt2_supported
                       boolean from the profile (default True)
    warnings           list of advisory strings (e.g. "forcing -np 1 because ...")
"""

from __future__ import annotations

import os

import re
import shlex
from pathlib import Path
from typing import Any


_CASPT2_HEADER_RE = re.compile(r"^\s*&CASPT2\b", re.M | re.I)


def detect_caspt2(input_text: str) -> bool:
    """True if the input has an &CASPT2 module block."""
    return bool(_CASPT2_HEADER_RE.search(input_text))


def prepare_launch(
    input_path: str | Path,
    profile: dict[str, Any] | None = None,
    requested_np: int = 1,
    *,
    job_name: str | None = None,
    apptainer_sif: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a safe pymolcas command + env dict for one Molcas launch.

    Parameters
    ----------
    input_path
        Path to the Molcas input file (.input).
    profile
        Optional runner-profile dict. The fields consulted:
          * ``execution.parallel_caspt2_supported`` (bool, default True)
          * ``execution.apptainer_sif`` (str, optional — path to a .sif image)
          * ``execution.pymolcas_command`` (str, default "pymolcas")
          * ``execution.env`` (dict, optional — extra env vars)
    requested_np
        MPI rank count the caller asked for (positive int). May be lowered
        to 1 if the input contains &CASPT2 and the profile disables parallel
        CASPT2.
    job_name
        Override for ``MOLCAS_PROJECT``. Default: input-file stem.
    apptainer_sif
        Path to a .sif container. If supplied (or set in the profile),
        the command is wrapped with ``apptainer exec``.
    extra_env
        Additional env vars to pass through.
    """
    if requested_np < 1:
        raise ValueError(f"requested_np must be >= 1; got {requested_np}")
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_text = input_path.read_text(encoding="utf-8", errors="replace")
    has_caspt2 = detect_caspt2(input_text)

    profile = profile or {}
    exec_cfg = (profile.get("execution") or {}) if isinstance(profile, dict) else {}
    parallel_caspt2_supported = bool(exec_cfg.get("parallel_caspt2_supported", True))

    warnings: list[str] = []
    effective_np = requested_np
    if has_caspt2 and not parallel_caspt2_supported and requested_np > 1:
        warnings.append(
            f"Input contains &CASPT2 and profile sets parallel_caspt2_supported=False; "
            f"forcing -np 1 (requested {requested_np}). Set parallel_caspt2_supported=True "
            "in the runner profile once your Molcas build is known to handle parallel CASPT2."
        )
        effective_np = 1

    project = job_name or input_path.stem
    pymolcas_cmd = exec_cfg.get("pymolcas_command", "pymolcas")
    # Container resolution: explicit arg > profile > env var.
    # Mirrors the GRASP pattern (CHEMTOOLS_GRASP_CONTAINER).
    sif = (
        apptainer_sif
        or exec_cfg.get("apptainer_sif")
        or os.environ.get("CHEMTOOLS_MOLCAS_CONTAINER")
    )
    if sif:
        sif = os.path.expanduser(sif)

    env: dict[str, str] = {
        "MOLCAS_PROJECT": project,
        "MOLCAS_NPROCS": str(effective_np),
    }
    # Don't double-set MOLCAS_MEM here — that's already in the >>> Export MOLCAS_MEM
    # line in the input file (drafter emits it). Profile env wins if specified.
    if isinstance(exec_cfg.get("env"), dict):
        env.update({str(k): str(v) for k, v in exec_cfg["env"].items()})
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})

    # Build the launch command. Two shapes:
    #   * Native:        pymolcas -np N <input>
    #   * Containerized: apptainer exec <sif> pymolcas -np N <input>
    inner_cmd = [pymolcas_cmd, "-np", str(effective_np), str(input_path)]
    if sif:
        command = ["apptainer", "exec", str(sif), *inner_cmd]
    else:
        command = inner_cmd

    return {
        "command": command,
        "command_str": " ".join(shlex.quote(c) for c in command),
        "env": env,
        "requested_np": requested_np,
        "effective_np": effective_np,
        "project": project,
        "has_caspt2": has_caspt2,
        "parallel_caspt2_supported": parallel_caspt2_supported,
        "apptainer_sif": str(sif) if sif else None,
        "warnings": warnings,
    }
