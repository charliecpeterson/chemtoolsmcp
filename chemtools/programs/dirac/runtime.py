"""DIRAC `pam-dirac` launcher.

Builds the command an agent should execute to run a DIRAC job. Does NOT
execute. Typical commands look like::

    apptainer exec $HOME/mycontainers/dirac-25.0.sif pam-dirac \
        --mpi=20 --inp=test.inp --mol=test.mol \
        --mw=1250 --nw=1250 --outcmo --get=DFACMO

For atomic-start chains, the molecule's launch carries an
``--copy="Elem1.h5 Elem2.h5 ..."`` flag that pulls the atomic .h5 files
into the molecule's scratch directory as starting orbitals.

The function reads from a runner profile when available (so users can
configure their apptainer SIF path + default MPI/memory) and falls
back to sensible defaults otherwise.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any


def pam_dirac_arguments(
    input_name: str,
    mol_name: str,
    *,
    mpi: int,
    mw: int | None = None,
    nw: int | None = None,
    copy_files: list[str] | None = None,
    put_files: list[str] | None = None,
    outcmo: bool = False,
    get_files: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    arguments = [
        f"--mpi={mpi}",
        f"--inp={input_name}",
        f"--mol={mol_name}",
    ]
    if mw is not None:
        arguments.append(f"--mw={mw}")
    if nw is not None:
        arguments.append(f"--nw={nw}")
    if copy_files:
        arguments.append(f"--copy={' '.join(copy_files)}")
    if put_files:
        arguments.append(f"--put={' '.join(put_files)}")
    if outcmo:
        arguments.append("--outcmo")
    if get_files:
        arguments.extend(f"--get={name}" for name in get_files)
    if extra_args:
        arguments.extend(extra_args)
    return arguments


def prepare_launch(
    input_file: str,
    mol_file: str,
    *,
    mpi: int | None = None,
    mw: int | None = None,
    nw: int | None = None,
    copy_files: list[str] | None = None,
    put_files: list[str] | None = None,
    outcmo: bool = False,
    get_files: list[str] | None = None,
    container_sif: str | None = None,
    pam_dirac_binary: str = "pam-dirac",
    apptainer_binary: str = "apptainer",
    work_dir: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Build a pam-dirac command line. Returns the command + environment.

    Parameters
    ----------
    input_file, mol_file
        Absolute (or work_dir-relative) paths to the .inp and .mol.
    mpi
        Number of MPI ranks. Defaults to 1 if not supplied.
    mw, nw
        Master / node memory in MB (DIRAC's ``--mw=`` / ``--nw=`` flags).
    copy_files
        List of files to copy into the run's scratch dir. For atomic
        start this is the list of per-element .h5 files. They must
        exist in ``work_dir`` (or be absolute paths the agent has
        staged).
    outcmo
        Add ``--outcmo`` (keep the converged MO coefficients in DFCOEF
        format in the output dir).
    get_files
        DIRAC artifact names to retrieve (``--get=NAME``). Common: DFACMO
        (active MO coefficients), DFCOEF (all coefficients), DFPCMO
        (positive-energy coefficients).
    container_sif
        Path to a DIRAC apptainer/singularity .sif. When provided, the
        command is prefixed with ``apptainer exec <sif> pam-dirac``.
        If None, pam-dirac is invoked directly (caller's PATH must
        contain it).
    work_dir
        Directory the command runs in. Resolves relative paths.

    Returns
    -------
    dict::

        command:      list of argv tokens (suitable for subprocess.run(shell=False))
        command_str:  shell-formatted string
        env:          {} (none required by pam-dirac; documented for future use)
        work_dir:     resolved working directory
        warnings:     list of agent-relevant warnings (e.g. missing copy_files)
    """
    if mpi is None:
        mpi = 1
    warnings: list[str] = []

    work = Path(work_dir).resolve() if work_dir else Path.cwd()

    # Validate file paths
    inp_path = _resolve_path(input_file, work)
    mol_path = _resolve_path(mol_file, work)
    if not Path(inp_path).exists():
        warnings.append(f"Input file not found yet: {inp_path}")
    if not Path(mol_path).exists():
        warnings.append(f"Mol file not found yet: {mol_path}")

    # copy_files staging check
    staged_copy: list[str] = []
    if copy_files:
        for f in copy_files:
            # If the path is bare basename, expect it in work_dir; absolute
            # paths are accepted as-is.
            p = Path(f) if Path(f).is_absolute() else work / f
            staged_copy.append(p.name)
            if not p.exists():
                warnings.append(f"--copy target staged?: {p}")

    # Container resolution: explicit arg > env var fallback.
    # Mirrors the GRASP pattern (CHEMTOOLS_GRASP_CONTAINER) so container
    # paths can be set once via MCP server env without threading them
    # through every tool call.
    if not container_sif:
        env_sif = os.environ.get("CHEMTOOLS_DIRAC_CONTAINER")
        if env_sif:
            container_sif = os.path.expanduser(env_sif)

    # Build the command
    cmd: list[str] = []
    if container_sif:
        cmd.extend([apptainer_binary, "exec", container_sif])
    cmd.append(pam_dirac_binary)
    cmd.extend(pam_dirac_arguments(
        Path(inp_path).name,
        Path(mol_path).name,
        mpi=mpi,
        mw=mw,
        nw=nw,
        copy_files=staged_copy,
        put_files=put_files,
        outcmo=outcmo,
        get_files=get_files,
        extra_args=extra_args,
    ))

    # pam-dirac output naming: <inp_stem>_<mol_stem>.{out,h5} unless the
    # two stems are identical, in which case it deduplicates to <stem>.{out,h5}.
    inp_stem = Path(inp_path).stem
    mol_stem = Path(mol_path).stem
    out_stem = inp_stem if inp_stem == mol_stem else f"{inp_stem}_{mol_stem}"

    return {
        "command":     cmd,
        "command_str": _shell_quote_list(cmd),
        "env":         {},
        "work_dir":    str(work),
        "warnings":    warnings,
        "expected_outputs": {
            "out": str(work / f"{out_stem}.out"),
            "h5":  str(work / f"{out_stem}.h5"),
        },
    }


def prepare_launch_from_profile(
    input_file: str,
    mol_file: str,
    profile: dict[str, Any],
    *,
    mpi: int | None = None,
    mw: int | None = None,
    nw: int | None = None,
    copy_files: list[str] | None = None,
    outcmo: bool = False,
    get_files: list[str] | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Same as ``prepare_launch`` but reads container/binary defaults from a
    DIRAC runner profile dict.

    Recognized profile keys::

        container_sif       path to dirac-XX.sif
        pam_dirac_binary    e.g. "pam-dirac" (default)
        apptainer_binary    e.g. "apptainer" or "singularity"
        default_mpi         int — fallback when ``mpi`` not specified
        default_mw          int — fallback for ``--mw=``
        default_nw          int — fallback for ``--nw=``
    """
    return prepare_launch(
        input_file=input_file,
        mol_file=mol_file,
        mpi=mpi if mpi is not None else profile.get("default_mpi"),
        mw=mw if mw is not None else profile.get("default_mw"),
        nw=nw if nw is not None else profile.get("default_nw"),
        copy_files=copy_files,
        outcmo=outcmo,
        get_files=get_files,
        container_sif=profile.get("container_sif"),
        pam_dirac_binary=profile.get("pam_dirac_binary", "pam-dirac"),
        apptainer_binary=profile.get("apptainer_binary", "apptainer"),
        work_dir=work_dir,
    )


def _resolve_path(p: str, work: Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((work / pp).resolve())


def _shell_quote_list(cmd: list[str]) -> str:
    return " ".join(shlex.quote(t) for t in cmd)


# ---------------------------------------------------------------------------
# Example runner profile — written to the user's profiles file by hand;
# documented here so the example file in the repo stays accurate.
# ---------------------------------------------------------------------------
EXAMPLE_PROFILE = {
    "dirac_local_apptainer": {
        "program":         "dirac",
        "container_sif":   os.path.expanduser("~/mycontainers/dirac-25.0.sif"),
        "pam_dirac_binary": "pam-dirac",
        "apptainer_binary": "apptainer",
        "default_mpi":     10,
        "default_mw":      1250,
        "default_nw":      1250,
    },
}
