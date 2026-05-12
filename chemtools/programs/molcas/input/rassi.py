"""Render the &RASSI input block.

Format (from the ZnO singlets+triplets+SOC reference):

    &RASSI &END
    NrOfJobiphs
    2 3 3
     1 2 3
     1 2 3
    IphNames
     JOB001
     JOB002
    EJob
    SpinOrbit
    MEES
    MESO
    Properties
     3
     'MLTPL  1' 1   'MLTPL  1' 2   'MLTPL  1' 3
    End of input
"""

from __future__ import annotations

from typing import Iterable


def render_rassi_block(
    *,
    jobiph_groups: list[dict],
    title: str | None = None,
    e_job: bool = True,
    spin_orbit: bool = False,
    print_matrix_elements_eigenstates: bool = True,
    print_matrix_elements_so: bool = True,
    print_expectation_values_eigenstates: bool = True,
    print_expectation_values_so: bool = True,
    natural_orbitals: int | None = None,
    properties: list[tuple[str, int]] | None = None,
    so_properties: list[tuple[str, int]] | None = None,
    threshold: float | None = None,
    extra_keywords: list[str] | None = None,
) -> str:
    """Build a &RASSI input block.

    jobiph_groups is a list of dicts describing each JobIph file:
        [
            {"name": "JOB001", "n_states": 3, "states": [1, 2, 3]},  # or omit states → [1..n]
            {"name": "JOB002", "n_states": 3, "states": [1, 2, 3]},
        ]

    properties / so_properties are lists of (label, component_index) tuples
    like ("MLTPL  1", 1) for dipole-X. The label string must be exactly 8
    characters (Molcas convention).
    """
    body: list[str] = ["&RASSI &END"]
    # NOTE: Molcas RASSI does NOT accept a `Title` keyword — input parsing
    # errors with "TITLE was not understood". Emit titles as `*` comments
    # instead.
    if title:
        body.append(f"* {title}")

    # NrOfJobiphs line: <num_jobiphs> <n_states_jobiph_1> <n_states_jobiph_2> ...
    n_jobs = len(jobiph_groups)
    n_states_list = [g.get("n_states", len(g.get("states", []))) for g in jobiph_groups]
    body.append("NrOfJobiphs")
    body.append(f" {n_jobs} " + " ".join(str(n) for n in n_states_list))
    # Per-jobiph state list
    for g in jobiph_groups:
        states = g.get("states") or list(range(1, g.get("n_states", 1) + 1))
        body.append(" " + " ".join(str(s) for s in states))

    if any("name" in g for g in jobiph_groups):
        body.append("IphNames")
        for g in jobiph_groups:
            body.append(f" {g.get('name', 'JOBIPH')}")

    if e_job:
        body.append("EJob")
    if spin_orbit:
        body.append("SpinOrbit")

    if natural_orbitals is not None:
        body.append("NatOrb")
        body.append(f" {int(natural_orbitals)}")

    if print_matrix_elements_eigenstates:
        body.append("MEES")
    if print_matrix_elements_so:
        body.append("MESO")
    if print_expectation_values_eigenstates:
        body.append("XVES")
    if print_expectation_values_so:
        body.append("XVSO")

    if threshold is not None:
        body.append("Thrs")
        body.append(f" {threshold}")

    if properties:
        body.append("Properties")
        body.append(f" {len(properties)}")
        body.append(" " + "   ".join(f"'{lbl}' {comp}" for lbl, comp in properties))

    if so_properties:
        body.append("SOProperty")
        body.append(f" {len(so_properties)}")
        body.append(" " + "   ".join(f"'{lbl}' {comp}" for lbl, comp in so_properties))

    if extra_keywords:
        body.extend(extra_keywords)

    body.append("End of input")
    return "\n".join(body) + "\n"


def render_jobiph_copy(jobiph_name: str, project_name: str = "$Project") -> str:
    """EMIL command: copy the working Project's JobIph to a stashed name for RASSI.

    Emits:  >>COPY $Project.JobIph JOB001
    """
    return f">>COPY {project_name}.JobIph {jobiph_name}\n"
