"""Type-safe stdin-heredoc builders for each GRASP executable.

Each builder returns a list of strings (one per stdin line) that can be fed
verbatim to ``run_grasp_exe(stdin_lines=...)``. The list form is preferred
so the runner can join with ``\n`` and the session log can show one
"prompt answer" per line.

These builders encode the prompt sequence each binary expects. The prompts
themselves are documented in the GRASP2018 manual chapter 7.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RcsfGenerationList:
    """One independently parameterized list in an rcsfgenerate union."""

    configurations: tuple[str, ...]
    active_orbitals: str
    twoj_min: int
    twoj_max: int
    excitations: int = 0


def rnucleus_input(
    *,
    z: int,
    a: int,
    point_source: bool = False,
    revise_radius: bool = False,
    nuclear_mass_amu: float | None = None,
    nuclear_spin: float = 0,
    dipole_moment: float = 0,
    quadrupole_moment: float = 0,
) -> list[str]:
    """Build rnucleus stdin.

    Prompts (in order):
      1. Atomic number Z
      2. Mass number A (0 for point nucleus)
      3. Revise rms radius / skin thickness? (n/y)
      4. Mass of neutral atom in amu (0 for static nucleus; defaults to A)
      5. Nuclear spin I (h/2π)
      6. Nuclear dipole moment (nuclear magnetons)
      7. Nuclear quadrupole moment (barns)
    """
    mass = nuclear_mass_amu if nuclear_mass_amu is not None else float(a)
    return [
        str(z),
        "0" if point_source else str(a),
        "y" if revise_radius else "n",
        str(mass),
        str(nuclear_spin),
        str(dipole_moment),
        str(quadrupole_moment),
    ]


def rcsfgenerate_input(
    *,
    core: int = 0,
    configurations: list[str],
    active_orbitals: str,
    twoj_min: int,
    twoj_max: int,
    excitations: int = 0,
    ordering: str = "*",
    generate_more: bool = False,
    additional_lists: Sequence[
        RcsfGenerationList | Mapping[str, object]
    ] | None = None,
) -> list[str]:
    """Build rcsfgenerate stdin for one or more generation lists.

    Prompts:
      1. Ordering (* / r / s / u)
      2. Core selection (0–6: 0=none, 1=He, 2=Ne, ..., 6=Rn)
      3. Configuration 1
         Configuration 2
         ...
         <blank>           # end of list
      4. Active orbitals (e.g. ``7s,6p,5d,5f``)
      5. 2*J range (low, high) → e.g. ``0,16``
      6. Number of excitations (0 = no excitations, negative = always doubly occ)
      7. Generate more lists? (y/n). If yes, repeat prompts 3 through 7.

    Put configurations sharing one active set, J range, and excitation policy
    in ``configurations``. Use ``additional_lists`` when those policies differ.
    For example, ordinary SD substitutions and a second ``-2`` list that
    requires correlation orbitals to be doubly occupied are separate lists.

    ``generate_more=True`` by itself previously emitted an incomplete answer
    stream. It is now rejected; supply the following lists explicitly.
    """
    if generate_more:
        raise ValueError(
            "generate_more=True leaves rcsfgenerate waiting for another "
            "list; provide that list with additional_lists"
        )
    if ordering not in {"*", "r", "s", "u"}:
        raise ValueError("ordering must be one of '*', 'r', 's', or 'u'")
    if not 0 <= core <= 6:
        raise ValueError("core must be between 0 and 6")

    generation_lists = [
        _generation_list(
            configurations=configurations,
            active_orbitals=active_orbitals,
            twoj_min=twoj_min,
            twoj_max=twoj_max,
            excitations=excitations,
        )
    ]
    generation_lists.extend(
        _coerce_generation_list(specification)
        for specification in additional_lists or ()
    )

    lines: list[str] = [ordering, str(core)]
    for index, specification in enumerate(generation_lists):
        lines.extend(specification.configurations)
        lines.append("")
        lines.append(specification.active_orbitals)
        lines.append(f"{specification.twoj_min},{specification.twoj_max}")
        lines.append(str(specification.excitations))
        lines.append("y" if index < len(generation_lists) - 1 else "n")
    return lines


def _coerce_generation_list(
    value: RcsfGenerationList | Mapping[str, object],
) -> RcsfGenerationList:
    if isinstance(value, RcsfGenerationList):
        return _generation_list(
            configurations=value.configurations,
            active_orbitals=value.active_orbitals,
            twoj_min=value.twoj_min,
            twoj_max=value.twoj_max,
            excitations=value.excitations,
        )
    required = {
        "configurations",
        "active_orbitals",
        "twoj_min",
        "twoj_max",
        "excitations",
    }
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(
            "each additional rcsfgenerate list requires exactly "
            f"{sorted(required)}; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    configurations = value["configurations"]
    if not isinstance(configurations, (list, tuple)) or not all(
        isinstance(configuration, str) for configuration in configurations
    ):
        raise ValueError("additional-list configurations must be strings")
    active_orbitals = value["active_orbitals"]
    if not isinstance(active_orbitals, str):
        raise ValueError("additional-list active_orbitals must be a string")
    try:
        twoj_min = int(value["twoj_min"])
        twoj_max = int(value["twoj_max"])
        excitations = int(value["excitations"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "additional-list J bounds and excitations must be integers"
        ) from error
    return _generation_list(
        configurations=configurations,
        active_orbitals=active_orbitals,
        twoj_min=twoj_min,
        twoj_max=twoj_max,
        excitations=excitations,
    )


def _generation_list(
    *,
    configurations: Sequence[str],
    active_orbitals: str,
    twoj_min: int,
    twoj_max: int,
    excitations: int,
) -> RcsfGenerationList:
    if not all(
        isinstance(configuration, str) for configuration in configurations
    ):
        raise ValueError("each rcsfgenerate configuration must be a string")
    if not isinstance(active_orbitals, str):
        raise ValueError("active_orbitals must be a string")
    cleaned = tuple(configuration.strip() for configuration in configurations)
    if not cleaned or any(not configuration for configuration in cleaned):
        raise ValueError("each rcsfgenerate list requires configurations")
    if not active_orbitals.strip():
        raise ValueError("each rcsfgenerate list requires active_orbitals")
    if twoj_min < 0 or twoj_max < twoj_min:
        raise ValueError("2J bounds must satisfy 0 <= twoj_min <= twoj_max")
    return RcsfGenerationList(
        configurations=cleaned,
        active_orbitals=active_orbitals.strip(),
        twoj_min=twoj_min,
        twoj_max=twoj_max,
        excitations=excitations,
    )


def rangular_input(*, default_settings: bool = True) -> list[str]:
    """Build rangular stdin.

    Prompts:
      1. Default settings? (y/n)
    """
    return ["y" if default_settings else "n"]


def rwfnestimate_input(
    *,
    default_settings: bool = True,
    speed_of_light_au: float | None = None,
    revise_grid: bool = False,
    sources: list[str] | None = None,
) -> list[str]:
    """Build rwfnestimate stdin.

    The non-default-settings branch is used to override either speed of light
    or radial grid. ``sources`` is the list of source numbers in the order
    the user wants to try:

      1: File (e.g. rwfn.inp from rwfnmchfmcdf)
      2: Thomas-Fermi
      3: Screened hydrogenic

    The common pattern after each source is to be prompted for a subshell
    pattern; passing ``*`` selects all remaining subshells.

    Examples
    --------
    Standard (Thomas-Fermi for all subshells):
        >>> rwfnestimate_input(sources=["2"])
        Default? y → source 2 → subshells *

    hf bootstrap (file source first, then Thomas-Fermi for the rest):
        >>> rwfnestimate_input(sources=["1", "rwfn.inp", "2"])
        Default? y → source 1 → file path → subshells * → source 2 → subshells *

    Non-rel limit (c = 2000):
        >>> rwfnestimate_input(default_settings=False, speed_of_light_au=2000, sources=["2"])
    """
    if sources is None:
        sources = ["2"]  # Thomas-Fermi by default

    lines: list[str] = []
    if default_settings and speed_of_light_au is None and not revise_grid:
        lines.append("y")
    else:
        # Non-default branch (Th_NR canonical pattern):
        #   Default settings ?                         → n
        #   Generate debug printout?                   → n
        #   File erwf.sum will be created [alt name?]  → * (= default)
        #   Change speed of light or radial grid?      → y
        #   Revise this value? (speed of light)        → y / n
        #   Enter the revised value:                   → <c value>
        #   Revise these values? (radial grid)         → n
        lines.append("n")
        lines.append("n")
        lines.append("*")
        lines.append("y")
        if speed_of_light_au is not None:
            lines.append("y")
            lines.append(str(speed_of_light_au))
        else:
            lines.append("n")
        lines.append("n")
    # Source iteration: rwfnestimate keeps prompting until all subshells are
    # estimated. Pattern: source_num, [file_path,] subshell_pattern, ...
    # No "continue?" between sources — the binary just loops if subshells remain.
    source_index = 0
    while source_index < len(sources):
        src = sources[source_index]
        if src.startswith("file:"):
            # Custom syntax: "file:/path/to/rwfn.inp" → source 1 + file path
            lines.append("1")
            lines.append(src[5:])
        elif src == "1":
            if source_index + 1 >= len(sources):
                raise ValueError(
                    "rwfnestimate file source 1 must be followed by a path"
                )
            source_index += 1
            lines.append("1")
            lines.append(sources[source_index])
        elif src in {"2", "3"}:
            lines.append(src)
        else:
            raise ValueError(
                "rwfnestimate sources must use '1', path, '2', '3', or "
                "the 'file:path' shorthand"
            )
        lines.append("*")  # all remaining subshells
        source_index += 1
    # Non-default branch: trailing "Revise any of these estimates?" prompt
    if not (default_settings and speed_of_light_au is None and not revise_grid):
        lines.append("n")
    return lines


def rmcdhf_input(
    *,
    default_settings: bool = True,
    speed_of_light_au: float | None = None,
    block_level_selections: list[str] | None = None,
    orbitals_to_optimize: str = "*",
    weighting: str = "5",
    spectroscopic_orbitals: str = "*",
    max_scf_cycles: int = 100,
    revise_orbital_initial: bool = False,
) -> list[str]:
    """Build rmcdhf stdin.

    Default flow:
      1. Default settings? (y/n)
      2. [if 'n'] revise speed of light? + value
      3. For each CSF block: levels to optimize (e.g. "1-2" or "1")
      4. If more than one ASF was selected, level weighting
         (5 = statistical weight 2J+1)
      5. Orbitals to optimize (e.g. "*" for all, "5*" for n=5)
      6. Which varied orbitals are spectroscopic ("*" = all, blank = none)
      7. Max SCF cycles (e.g. 100)

    ``block_level_selections`` is a per-block list — Si has 5 blocks so
    you'd pass ``["1-2", "1", "1-2", "1", "1"]`` if you want all levels in
    each block.
    """
    lines: list[str] = []
    if default_settings and speed_of_light_au is None:
        lines.append("y")
    else:
        # Non-default branch (Th_NR canonical pattern):
        #   Default settings?                          → n
        #   Generate debug printout?                   → n
        #   Change speed of light or radial grid?      → y
        #   Revise this value? (speed of light)        → y / n
        #   Enter the revised value:                   → <c value>
        #   Revise these values? (radial grid)         → n
        #   Revise default <something else>?           → n
        lines.append("n")
        lines.append("n")
        lines.append("y")
        if speed_of_light_au is not None:
            lines.append("y")
            lines.append(str(speed_of_light_au))
        else:
            lines.append("n")
        lines.append("n")
        lines.append("n")

    if block_level_selections is None:
        selections = ["1"]
    elif not block_level_selections:
        raise ValueError("block_level_selections must not be empty")
    else:
        selections = block_level_selections
    selected_asfs = sum(_asf_selection_count(selection) for selection in selections)
    if selected_asfs == 0:
        raise ValueError("rmcdhf requires at least one selected ASF")
    lines.extend(selections)
    if selected_asfs > 1:
        lines.append(weighting)
    lines.append(orbitals_to_optimize)
    lines.append(spectroscopic_orbitals)
    lines.append(str(max_scf_cycles))
    # Non-default branch: tail prompts after SCF cycles
    #   Some default to revise?                        → n
    #   <something>?                                   → 1
    if not (default_settings and speed_of_light_au is None):
        lines.append("n")
        lines.append("1")
    return lines


def _asf_selection_count(selection: str) -> int:
    tokens = [
        token for token in re.split(r"[\s,]+", selection.strip()) if token
    ]
    selected: set[int] = set()
    for token in tokens:
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise ValueError(f"invalid ASF selection token: {token!r}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first < 1 or last < first:
            raise ValueError(f"invalid ASF selection range: {token!r}")
        overlap = selected.intersection(range(first, last + 1))
        if overlap:
            raise ValueError(
                f"ASF selection contains duplicate indices: {sorted(overlap)}"
            )
        selected.update(range(first, last + 1))
    return len(selected)


def hf_input(
    *,
    element_av_z: str,
    orbital_list: str,
    open_shell: str,
    estimate_orbitals: str = "ALL",
    full_breit: bool = True,
    relativistic_corrections: bool = True,
    qed_corrections: bool = False,
    finite_nucleus: bool = False,
) -> list[str]:
    """Build hf (non-relativistic Hartree-Fock) stdin.

    The ``hf`` program ships with GRASP and produces a wfn.out file that
    can be converted to rwfn.out via ``rwfnmchfmcdf``. It's invaluable for
    high-Z atoms where the standard Thomas-Fermi or screened-hydrogenic
    initial guesses don't converge.

    Prompts (typical):
      1. ``Element,AV,Z`` (e.g. ``Cf,AV,98``)
      2. Orbital list (closed orbitals, space- or comma-separated)
      3. Open shell occupation (e.g. ``5f(10)``)
      4. Estimate orbital wavefunctions: ALL / SOME / NONE
      5. Y/N for: relativistic corrections, full Breit, QED, finite-nucleus

    Defaults match the Cf run_Cf.sh script.
    """
    return [
        element_av_z,
        orbital_list,
        open_shell,
        estimate_orbitals,
        "Y" if relativistic_corrections else "N",
        "Y" if full_breit else "N",
        "N" if not qed_corrections else "Y",
        "N" if not finite_nucleus else "Y",
    ]


def rwfnmchfmcdf_input() -> list[str]:
    """rwfnmchfmcdf has no prompts — it reads wfn.inp and writes rwfn.out."""
    return []


def rsave_args(prefix: str) -> list[str]:
    """rsave takes the prefix as a positional argument, not stdin."""
    return [prefix]


def jj2lsj_input(
    *,
    name: str,
    mixing_coefficients: bool = False,
    unique_labeling: bool = True,
    default_settings: bool = True,
) -> list[str]:
    """Build jj2lsj stdin.

    Prompts:
      1. Name of state (e.g. ``5f10``, ``2s_2p_DF``)
      2. Mixing coefficients from a CI calc? (y/n)
      3. Need unique labeling? (y/n)
      4. Default settings? (y/n)
    """
    return [
        name,
        "y" if mixing_coefficients else "n",
        "y" if unique_labeling else "n",
        "y" if default_settings else "n",
    ]


def rci_input(
    *,
    name: str,
    transverse: bool = True,
    photon_freq_scale: float = 1e-6,
    vacuum_polarization: bool = True,
    normal_mass_shift: bool = False,
    specific_mass_shift: bool = False,
    self_energy: bool = True,
    max_n_self_energy: int = 3,
    block_level_selections: list[str] | None = None,
    default_settings: bool = True,
) -> list[str]:
    """Build rci stdin (relativistic CI with Breit + QED).

    Prompts:
      1. Default settings? (y/n)
      2. Name of state
      3. Contribution of H (Transverse)?
      4. Modify photon frequencies? + scale factor
      5. Vacuum polarization?
      6. Normal mass shift?
      7. Specific mass shift?
      8. Self energy? + Max n for self energy
      9. Per-block level selections (ASF serial numbers)
    """
    lines = [
        "y" if default_settings else "n",
        name,
        "y" if transverse else "n",
    ]
    if transverse:
        lines.append("y")  # modify photon freq
        lines.append(str(photon_freq_scale))
    lines.extend([
        "y" if vacuum_polarization else "n",
        "y" if normal_mass_shift else "n",
        "y" if specific_mass_shift else "n",
        "y" if self_energy else "n",
    ])
    if self_energy:
        lines.append(str(max_n_self_energy))
    if block_level_selections:
        lines.extend(block_level_selections)
    else:
        lines.append("1")
    return lines


def rhfs_input(
    *,
    name: str,
    ci_mixing: bool = False,
    default_settings: bool = True,
) -> list[str]:
    """Build rhfs stdin (hyperfine A/B constants + Landé g_J).

    Reads isodata (nuclear spin + moments) + name.c/.w/.(c)m; writes name.(c)h.

    Prompts:
      1. Default settings? (y/n)
      2. Name of state
      3. Mixing coefficients from a CI calc.? (y if from rci .cm, n if rmcdhf .m)
    """
    return [
        "y" if default_settings else "n",
        name,
        "y" if ci_mixing else "n",
    ]


def rhfs_lsj_input(
    *,
    name: str,
    ci_mixing: bool = False,
    energy_sorted: bool = True,
) -> list[str]:
    """Build rhfs_lsj stdin — relabel rhfs output (name.(c)h) with LSJ terms.

    Prompts:
      1. Name of state
      2. Hfs data from a CI calc? (y/n)
      3. Energy sorted output? (y/n)
    """
    return [
        name,
        "y" if ci_mixing else "n",
        "y" if energy_sorted else "n",
    ]


def ris4_input(
    *,
    name: str,
    ci_mixing: bool = False,
    higher_order_field_shift: bool = False,
    save_angular: bool = False,
    default_settings: bool = True,
) -> list[str]:
    """Build ris4 stdin (isotope-shift electronic factors).

    Computes normal + specific mass-shift parameters and the electron density at
    the nucleus (first-order field-shift factor); writes name.i. The factors are
    isotope-independent, so any isotope (including spin-0) works.

    Prompts:
      1. Default settings? (y/n)
      2. Name of state
      3. Mixing coefficients from a CI calc.? (y/n)
      4. Compute higher order field shift electronic factors? (y/n)
      5. Save ang. coefficients of one- and two-body op.? (y/n)
    """
    return [
        "y" if default_settings else "n",
        name,
        "y" if ci_mixing else "n",
        "y" if higher_order_field_shift else "n",
        "y" if save_angular else "n",
    ]


def rbiotransform_input(
    *,
    initial: str,
    final: str,
    ci_mixing: bool = False,
    all_symmetries: bool = True,
    default_settings: bool = True,
) -> list[str]:
    """Build rbiotransform stdin — biorthogonalise two states before rtransition.

    Prompts:
      1. Default settings? (y/n)
      2. Input from a CI calculation? (y/n)
      3. Name of the Initial state
      4. Name of the Final state
      5. Transformation of all J symmetries? (y/n)
    """
    return [
        "y" if default_settings else "n",
        "y" if ci_mixing else "n",
        initial,
        final,
        "y" if all_symmetries else "n",
    ]


def rtransition_input(
    *,
    initial: str,
    final: str,
    transition_types: str = "E1",
    ci_mixing: bool = False,
    default_settings: bool = True,
) -> list[str]:
    """Build rtransition stdin — radiative transition rates between two states.

    Run rbiotransform on the same pair first. ``transition_types`` is a GRASP
    spec string, e.g. ``"E1"`` or ``"E1,M2"``.

    Prompts:
      1. Default settings? (y/n)
      2. Input from a CI calculation? (y/n)
      3. Name of the Initial state
      4. Name of the Final state
      5. Transition specifications (e.g. E1 or E1,M2)
    """
    return [
        "y" if default_settings else "n",
        "y" if ci_mixing else "n",
        initial,
        final,
        transition_types,
    ]
