"""GRASP workflow orchestrators.

Each orchestrator returns a structured plan: a sequence of
``{exe, stdin, args, post, description}`` steps the caller can execute via
``run_grasp_exe``. In ``local`` mode the MCP handlers actually execute the
steps; in ``analysis`` mode the plan is returned for the user to run manually.

Workflows
---------
* ``plan_dhf_workflow`` — straight Dirac-Fock SCF chain
  (rnucleus → rcsfgenerate → rangular → rwfnestimate → rmcdhf → rsave →
  jj2lsj → rlevels)
* ``plan_nonrel_limit_workflow`` — same chain but with ``c=2000`` au, used
  to verify the nonrelativistic limit
* ``plan_restart_from_workflow`` — start a new run with a previous run's
  ``*.w`` orbital file as the initial guess (great for n-shell expansion)
* ``plan_hf_bootstrap_workflow`` — run the (non-rel) ``hf`` code first,
  convert via ``rwfnmchfmcdf``, then feed those as the rwfnestimate
  starting orbitals (essential for high-Z atoms where Thomas-Fermi
  fails to converge — Cf, Bk, etc.)
"""

from __future__ import annotations

from typing import Any

from chemtools.programs.grasp.input.heredoc import (
    rnucleus_input,
    rcsfgenerate_input,
    rangular_input,
    rwfnestimate_input,
    rmcdhf_input,
    jj2lsj_input,
    hf_input,
    rwfnmchfmcdf_input,
    rsave_args,
)


def plan_dhf_workflow(
    *,
    z: int,
    a: int,
    nuclear_mass_amu: float | None = None,
    nuclear_spin: float = 0,
    dipole_moment: float = 0,
    quadrupole_moment: float = 0,
    core: int = 0,
    configurations: list[str],
    active_orbitals: str,
    twoj_min: int,
    twoj_max: int,
    excitations: int = 0,
    additional_generation_lists: list[dict[str, object]] | None = None,
    block_level_selections: list[str],
    expected_csf_blocks: list[dict[str, object]],
    orbitals_to_optimize: str | None = None,
    weighting: str = "5",
    spectroscopic_orbitals: str | None = None,
    max_scf_cycles: int = 100,
    name: str,
    speed_of_light_au: float | None = None,
    rwfnestimate_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Build the full DHF workflow plan as an ordered list of steps.

    ``name`` is the rsave prefix (e.g. ``2s_2p_DF``, ``5f10``). All output
    files will be ``<name>.{w,c,m,sum,alog,log,lsj.lbl}``.

    Set ``speed_of_light_au=2000`` for the non-relativistic limit (or use
    ``plan_nonrel_limit_workflow`` directly).
    """
    if rwfnestimate_sources is None:
        rwfnestimate_sources = ["2"]  # Thomas-Fermi default
    orbitals_to_optimize, spectroscopic_orbitals = _orbital_policy(
        excitations=excitations,
        additional_generation_lists=additional_generation_lists,
        orbitals_to_optimize=orbitals_to_optimize,
        spectroscopic_orbitals=spectroscopic_orbitals,
    )
    if len(block_level_selections) != len(expected_csf_blocks):
        raise ValueError(
            "block_level_selections and expected_csf_blocks must have the "
            "same length"
        )

    steps: list[dict[str, Any]] = [
        {
            "exe": "rnucleus",
            "stdin": rnucleus_input(
                z=z, a=a, nuclear_mass_amu=nuclear_mass_amu,
                nuclear_spin=nuclear_spin, dipole_moment=dipole_moment,
                quadrupole_moment=quadrupole_moment,
            ),
            "args": [],
            "post": [],
            "description": f"Build nuclear data for Z={z}, A={a}",
        },
        {
            "exe": "rcsfgenerate",
            "stdin": rcsfgenerate_input(
                core=core, configurations=configurations,
                active_orbitals=active_orbitals,
                twoj_min=twoj_min, twoj_max=twoj_max,
                excitations=excitations,
                additional_lists=additional_generation_lists,
            ),
            "args": [],
            "post": ["cp rcsf.out rcsf.inp"],
            "expected_csf_blocks": expected_csf_blocks,
            "description": f"Generate CSF list for {len(configurations)} configuration(s)",
        },
        {
            "exe": "rangular",
            "stdin": rangular_input(),
            "args": [],
            "post": [],
            "description": "Angular integration (mcp.30..mcp.39 files)",
        },
        {
            "exe": "rwfnestimate",
            "stdin": rwfnestimate_input(
                sources=rwfnestimate_sources,
                speed_of_light_au=speed_of_light_au,
                default_settings=(speed_of_light_au is None),
            ),
            "args": [],
            "post": [],
            "description": "Generate initial orbital estimates"
                           + (f" (c={speed_of_light_au} au, NON-REL LIMIT)"
                              if speed_of_light_au else ""),
        },
        {
            "exe": "rmcdhf",
            "stdin": rmcdhf_input(
                block_level_selections=block_level_selections,
                orbitals_to_optimize=orbitals_to_optimize,
                weighting=weighting,
                spectroscopic_orbitals=spectroscopic_orbitals,
                max_scf_cycles=max_scf_cycles,
                speed_of_light_au=speed_of_light_au,
                default_settings=(speed_of_light_au is None),
            ),
            "args": [],
            "post": [],
            "description": "Self-consistent Dirac-Fock SCF",
        },
        {
            "exe": "rsave",
            "stdin": "",
            "args": rsave_args(name),
            "post": [],
            "description": f"Save converged results as {name}.*",
        },
        {
            "exe": "jj2lsj",
            "stdin": jj2lsj_input(name=name, mixing_coefficients=False),
            "args": [],
            "post": [],
            "description": "Transform jj-coupled CSFs to LSJ representation",
        },
        {
            "exe": "rlevels",
            "stdin": "",
            "args": [f"{name}.m"],
            "post": [],
            "description": "Print energy levels with splittings",
        },
    ]

    return {
        "workflow": "dhf",
        "name": name,
        "speed_of_light_au": speed_of_light_au or 137.0359991390,
        "is_nonrel_limit": speed_of_light_au is not None and speed_of_light_au > 500,
        "orbital_policy": {
            "orbitals_to_optimize": orbitals_to_optimize,
            "spectroscopic_orbitals": spectroscopic_orbitals,
            "correlation_expansion": _has_correlation_expansion(
                excitations,
                additional_generation_lists,
            ),
        },
        "n_steps": len(steps),
        "steps": steps,
        "expected_outputs": [
            f"{name}.w (radial wavefunctions)",
            f"{name}.c (CSF list)",
            f"{name}.m (mixing coefficients)",
            f"{name}.sum (run summary)",
            f"{name}.lsj.lbl (LSJ-coupled compositions)",
            "rlevels stdout (energy table)",
        ],
        "next_actions": [
            "Run each step in order via run_grasp_exe (or call run_grasp_workflow)",
            f"Parse the rlevels stdout via parse_grasp_levels()",
            f"Parse {name}.sum via parse_grasp_sum() to verify SCF converged",
            f"Parse {name}.lsj.lbl via parse_grasp_lsjlbl() for LS composition",
        ],
    }


def plan_nonrel_limit_workflow(
    *,
    speed_of_light_au: float = 2000.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """DHF workflow with c set to a large value (default 2000 au).

    The relativistic Hamiltonian smoothly reduces to the non-relativistic
    Schrödinger limit as c → ∞. By setting c much larger than the physical
    137.036 au, all relativistic effects are suppressed, letting the user
    compare against non-rel calculations or isolate purely-relativistic
    contributions to a property.

    Pass the same arguments as ``plan_dhf_workflow`` (excluding the
    ``speed_of_light_au`` argument which is forced).
    """
    kwargs["speed_of_light_au"] = speed_of_light_au
    plan = plan_dhf_workflow(**kwargs)
    plan["workflow"] = "nonrel_limit"
    return plan


def plan_restart_from_workflow(
    *,
    previous_w_file: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """DHF workflow that uses a previous run's ``*.w`` as starting orbitals.

    Common pattern: optimize n=3 first, save as ``2s_3.w``; then restart
    n=4 calculation by copying ``2s_3.w`` to the new working directory as
    ``rwfn.inp``. rwfnestimate auto-detects the file source.

    The orchestrator emits a ``cp`` pre-step so the user knows to seed
    rwfn.inp before rwfnestimate runs.
    """
    plan = plan_dhf_workflow(rwfnestimate_sources=[f"file:{previous_w_file}", "2"], **kwargs)
    plan["workflow"] = "restart_from_w"
    plan["restart_source"] = previous_w_file
    plan["preamble"] = [
        f"cp {previous_w_file} rwfn.inp  # seed initial orbitals from previous run",
    ]
    return plan


def plan_hf_bootstrap_workflow(
    *,
    z: int,
    a: int,
    element_symbol: str,
    hf_orbital_list: str,
    hf_open_shell: str,
    nuclear_mass_amu: float | None = None,
    nuclear_spin: float = 0,
    dipole_moment: float = 0,
    quadrupole_moment: float = 0,
    core: int = 0,
    configurations: list[str],
    active_orbitals: str,
    twoj_min: int,
    twoj_max: int,
    excitations: int = 0,
    additional_generation_lists: list[dict[str, object]] | None = None,
    block_level_selections: list[str],
    expected_csf_blocks: list[dict[str, object]],
    orbitals_to_optimize: str | None = None,
    weighting: str = "5",
    spectroscopic_orbitals: str | None = None,
    max_scf_cycles: int = 100,
    name: str,
) -> dict[str, Any]:
    """Workflow: hf (non-rel) → rwfnmchfmcdf → DHF using hf orbitals as guess.

    Required for high-Z atoms (Cf, Bk, Es, etc.) where Thomas-Fermi alone
    fails to give a useful starting guess for DHF. The chain is:

      1. rnucleus
      2. rcsfgenerate + cp rcsf.out → rcsf.inp
      3. rangular
      4. hf  (non-relativistic Hartree-Fock — writes wfn.out)
      5. cp wfn.out → wfn.inp
      6. rwfnmchfmcdf (converts to rwfn.out)
      7. cp rwfn.out → rwfn.inp
      8. rwfnestimate (default settings find rwfn.inp automatically)
      9. rmcdhf
      10. rsave / jj2lsj / rlevels

    ``element_symbol`` is the chemical symbol (e.g. ``Cf``, ``Th``).
    ``hf_orbital_list`` is the closed-shell orbitals as a space-separated
    string (e.g. ``" 1s  2s  2p  3s  3p  3d  4s  4p  4d  4f  5s  5p  5d  6s  6p  7s"``).
    ``hf_open_shell`` is the open-shell occupation (e.g. ``"5f(10)"``).
    """
    orbitals_to_optimize, spectroscopic_orbitals = _orbital_policy(
        excitations=excitations,
        additional_generation_lists=additional_generation_lists,
        orbitals_to_optimize=orbitals_to_optimize,
        spectroscopic_orbitals=spectroscopic_orbitals,
    )
    if len(block_level_selections) != len(expected_csf_blocks):
        raise ValueError(
            "block_level_selections and expected_csf_blocks must have the "
            "same length"
        )
    steps: list[dict[str, Any]] = [
        {
            "exe": "rnucleus",
            "stdin": rnucleus_input(
                z=z, a=a, nuclear_mass_amu=nuclear_mass_amu,
                nuclear_spin=nuclear_spin, dipole_moment=dipole_moment,
                quadrupole_moment=quadrupole_moment,
            ),
            "args": [],
            "post": [],
            "description": f"Build nuclear data for Z={z}, A={a}",
        },
        {
            "exe": "rcsfgenerate",
            "stdin": rcsfgenerate_input(
                core=core, configurations=configurations,
                active_orbitals=active_orbitals,
                twoj_min=twoj_min, twoj_max=twoj_max,
                excitations=excitations,
                additional_lists=additional_generation_lists,
            ),
            "args": [],
            "post": ["cp rcsf.out rcsf.inp"],
            "expected_csf_blocks": expected_csf_blocks,
            "description": f"Generate CSF list for {len(configurations)} configuration(s)",
        },
        {
            "exe": "rangular",
            "stdin": rangular_input(),
            "args": [],
            "post": [],
            "description": "Angular integration",
        },
        {
            "exe": "hf",
            "stdin": hf_input(
                element_av_z=f"{element_symbol},AV,{z}",
                orbital_list=hf_orbital_list,
                open_shell=hf_open_shell,
            ),
            "args": [],
            "post": ["cp wfn.out wfn.inp"],
            "description": "Non-relativistic Hartree-Fock (high-Z guess)",
        },
        {
            "exe": "rwfnmchfmcdf",
            "stdin": rwfnmchfmcdf_input(),
            "args": [],
            "post": ["cp rwfn.out rwfn.inp"],
            "description": "Convert hf wfn.inp → grasp rwfn.out",
        },
        {
            "exe": "rwfnestimate",
            "stdin": rwfnestimate_input(sources=["2"]),  # rwfn.inp auto-picked
            "args": [],
            "post": [],
            "description": "Read hf orbitals from rwfn.inp + Thomas-Fermi for any unfilled subshells",
        },
        {
            "exe": "rmcdhf",
            "stdin": rmcdhf_input(
                block_level_selections=block_level_selections,
                orbitals_to_optimize=orbitals_to_optimize,
                weighting=weighting,
                spectroscopic_orbitals=spectroscopic_orbitals,
                max_scf_cycles=max_scf_cycles,
            ),
            "args": [],
            "post": [],
            "description": "Dirac-Fock SCF using hf-bootstrapped orbitals",
        },
        {
            "exe": "rsave",
            "stdin": "",
            "args": rsave_args(name),
            "post": [],
            "description": f"Save as {name}.*",
        },
        {
            "exe": "jj2lsj",
            "stdin": jj2lsj_input(name=name, mixing_coefficients=False),
            "args": [],
            "post": [],
            "description": "jj → LSJ transformation",
        },
        {
            "exe": "rlevels",
            "stdin": "",
            "args": [f"{name}.m"],
            "post": [],
            "description": "Print energy levels with splittings",
        },
    ]

    return {
        "workflow": "hf_bootstrap",
        "name": name,
        "element": element_symbol,
        "z": z,
        "orbital_policy": {
            "orbitals_to_optimize": orbitals_to_optimize,
            "spectroscopic_orbitals": spectroscopic_orbitals,
            "correlation_expansion": _has_correlation_expansion(
                excitations,
                additional_generation_lists,
            ),
        },
        "n_steps": len(steps),
        "steps": steps,
        "expected_outputs": [
            "wfn.out (hf non-rel orbitals)",
            "rwfn.out (converted to grasp format)",
            f"{name}.{{w,c,m,sum,lsj.lbl}}",
            "rlevels stdout (energy table)",
        ],
        "next_actions": [
            f"Recommended for high-Z atoms (Z>=80) where Thomas-Fermi diverges",
            f"Parse {name}.sum to verify SCF converged",
            f"Compare to plain DHF if you want to quantify the bootstrap improvement",
        ],
    }


def _orbital_policy(
    *,
    excitations: int,
    additional_generation_lists: list[dict[str, object]] | None,
    orbitals_to_optimize: str | None,
    spectroscopic_orbitals: str | None,
) -> tuple[str, str]:
    correlation_expansion = _has_correlation_expansion(
        excitations,
        additional_generation_lists,
    )
    if correlation_expansion and (
        orbitals_to_optimize is None or spectroscopic_orbitals is None
    ):
        raise ValueError(
            "correlation workflows require explicit orbitals_to_optimize "
            "and spectroscopic_orbitals; for a new layer, vary only that "
            "layer and use a blank spectroscopic selection"
        )
    return orbitals_to_optimize or "*", (
        "*" if spectroscopic_orbitals is None else spectroscopic_orbitals
    )


def _has_correlation_expansion(
    excitations: int,
    additional_generation_lists: list[dict[str, object]] | None,
) -> bool:
    if excitations != 0:
        return True
    return any(
        specification.get("excitations", 0) != 0
        for specification in additional_generation_lists or ()
    )
