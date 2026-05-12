"""SEWARD / GATEWAY input block builder.

Renders the integral-driver block: title, optional symmetry generators,
per-element basis-set blocks with coordinates, optional reaction-field /
RICD / Cholesky / extras.

Coordinates are written in the unit the user provides; if `geometry_units`
is "angstrom", an `Angstrom` keyword is emitted so SEWARD does the conversion.
"""

from __future__ import annotations

from typing import Any, Iterable

from chemtools.programs.molcas.input._utils import (
    auto_label,
    group_atoms_by_element,
    normalize_atoms,
)
from chemtools.programs.molcas.input.basis_library import (
    resolve_basis_assignments,
    get_inline_basis_block,
)


def render_seward_block(
    *,
    atoms: list[dict],
    basis: str | dict[str, str],
    title: str | None = None,
    symmetry: str | None = None,
    geometry_units: str = "bohr",
    pkthrs: float | None = None,
    cholesky: bool = False,
    ricd: bool = False,
    expert: bool = False,
    extra_keywords: list[str] | None = None,
    use_gateway: bool = False,
    inline_basis: bool = False,
    gateway_extras: list[str] | None = None,
) -> str:
    """Build the SEWARD (or GATEWAY+SEWARD) input block.

    Parameters
    ----------
    atoms
        List of dicts with keys 'symbol', 'x', 'y', 'z', optional 'label'.
    basis
        Basis spec, either a global name like "ANO-S" or a per-element dict.
        See basis_library.resolve_basis_assignments.
    symmetry
        Symmetry generators string (e.g. "X XY" for C2v, "X" for Cs); None = C1.
    geometry_units
        "bohr" (Molcas default) or "angstrom".
    pkthrs
        Optional integral pre-screening threshold (PkThrs).
    cholesky
        If True, emit `Cholesky` to enable Cholesky decomposition of integrals.
    ricd
        If True, emit `RICD` to enable resolution-of-identity Cholesky decomposition.
    expert
        If True, emit `Expert` (allows non-recommended options to silence warnings).
    extra_keywords
        Free-form lines to append before End of input (e.g. "RF-Input ...").
    use_gateway
        If True, emit a separate `&GATEWAY ... End of input` block before
        `&SEWARD ... End of input` (cleaner for opt loops; recommended).
    inline_basis
        If True, splice each element's full primitive basis block (read from
        the bundled basis library) into the input. The result is portable
        across Molcas builds with different bundled libraries. Default False
        emits the library-reference form ``H.ANO-S...4s3p.``.
    gateway_extras
        Extra lines to append inside the GATEWAY block (after basis blocks,
        before "End of input"). Use this to inject Constraint blocks for
        constrained optimizations / scans. Only honored when use_gateway=True.
    """
    atoms_norm = auto_label(normalize_atoms(atoms))
    elements = list(dict.fromkeys(a["symbol"] for a in atoms_norm))
    basis_assignments = resolve_basis_assignments(basis, elements)

    # Convert Angstrom → bohr internally so we always emit coordinates in
    # Molcas's default unit. Avoids a SEWARD keyword-ordering bug: when
    # Symmetry is present, the top-level `Angstrom` keyword is rejected
    # with "ANGSTROM is not a keyword!". Bohr coordinates are universally
    # accepted.
    if geometry_units.lower() == "angstrom":
        _BOHR_PER_ANGSTROM = 1.8897261245650618
        atoms_norm = [
            {**a,
             "x": a["x"] * _BOHR_PER_ANGSTROM,
             "y": a["y"] * _BOHR_PER_ANGSTROM,
             "z": a["z"] * _BOHR_PER_ANGSTROM}
            for a in atoms_norm
        ]
    grouped = group_atoms_by_element(atoms_norm)

    coord_block_lines: list[str] = []
    if title:
        coord_block_lines.append("Title")
        coord_block_lines.append(f" {title}")
    if symmetry:
        coord_block_lines.append("Symmetry")
        coord_block_lines.append(f" {symmetry}")
    if expert:
        coord_block_lines.append("Expert")
    for element, group in grouped.items():
        coord_block_lines.append("Basis set")
        if inline_basis:
            # Parse the basis_assignments label (`H.ANO-S...4s3p.`) into
            # (basis_name, contraction) and pull the inline primitives.
            label = basis_assignments[element]
            parts = label.rstrip(".").split(".")
            # Format is `<E>.<lib>...<contraction>` — `parts` may have
            # variable-length middle from author/primitive names. Skip the
            # element + library prefix and take the LAST token as contraction.
            basis_name = parts[1] if len(parts) > 1 else label
            contraction = parts[-1] if len(parts) > 2 else None
            inline_block = get_inline_basis_block(basis_name, element, contraction)
            # Emit the inline block (already includes element label + primitives)
            coord_block_lines.append(inline_block.rstrip())
        else:
            coord_block_lines.append(basis_assignments[element])
        for atom in group:
            atom_label = atom.get("label") or element
            coord_block_lines.append(
                f"{atom_label:<6s} {atom['x']:18.10f} {atom['y']:18.10f} {atom['z']:18.10f}"
            )
        coord_block_lines.append("End of basis")

    seward_extras: list[str] = []
    if pkthrs is not None:
        seward_extras.append("PkThrs")
        seward_extras.append(f" {pkthrs:.1E}")
    if cholesky:
        seward_extras.append("Cholesky")
    if ricd:
        seward_extras.append("RICD")
    if extra_keywords:
        seward_extras.extend(extra_keywords)

    if use_gateway:
        gateway_lines = [*coord_block_lines]
        if gateway_extras:
            gateway_lines.extend(gateway_extras)
        gateway = ["&GATEWAY", *gateway_lines, "End of input", ""]
        seward = ["&SEWARD"]
        if seward_extras:
            seward.extend(seward_extras)
        seward.append("End of input")
        return "\n".join(gateway + seward) + "\n"
    # Single SEWARD block (legacy / dimethylcarbene-style)
    body = ["&SEWARD &END", *coord_block_lines]
    body.extend(seward_extras)
    body.append("End of input")
    return "\n".join(body) + "\n"
