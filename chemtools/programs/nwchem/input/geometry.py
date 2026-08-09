"""NWChem geometry-handling input helpers.

Two entry points:

  * extract_nwchem_geometry        Extract a converged or trajectory frame
                                   from an NWChem output and render an
                                   XYZ file or NWChem geometry block.
                                   Picks the best frame via
                                   _select_best_optimization_frame when
                                   the run didn't fully converge.

  * draft_initial_geometry         Build a starting geometry XYZ file
                                   from an element list using covalent
                                   radii estimates. Returns a payload
                                   ready to feed into the DFT workflow
                                   drafter.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any

from chemtools.core.common import read_text, ELEMENT_TO_Z, COVALENT_RADII
from chemtools.programs.nwchem.parse.input import (
    extract_nwchem_geometry_block,
    render_nwchem_geometry_block,
)
from chemtools.programs.nwchem.parse.freq import parse_trajectory as _parse_trajectory_raw
from chemtools.programs.nwchem.input.opt_followup import _select_best_optimization_frame


def extract_nwchem_geometry(
    output_path: str,
    frame: "str | int" = "best",
    input_path: "str | None" = None,
) -> dict:
    """Extract a geometry from a NWChem optimization output as XYZ and NWChem block text.

    Works without the original input file.  When ``input_path`` is provided the
    NWChem geometry block preserves the original header (units, symmetry, etc.) and
    atom labels; otherwise a plain ``geometry units angstrom`` block is emitted.

    Parameters
    ----------
    output_path:
        Path to the NWChem ``.out`` file (must contain an optimization trajectory).
    frame:
        Which geometry to extract.  One of:

        - ``"best"``  — smart selection: converged/incomplete → last frame;
          failed → min-energy frame if divergence > 1 mHa, else last.
        - ``"last"``  — last frame regardless of status.
        - ``"first"`` — first frame (useful as a before/after comparison).
        - ``"min_energy"`` — frame with the lowest energy.
        - ``int``     — specific frame index (0-based).
    input_path:
        Optional path to the original ``.nw`` input; used to preserve geometry
        header/directives and atom labels.

    Returns
    -------
    dict with keys:
        xyz_text, nwchem_geometry_block, selected_frame (step/energy/metrics),
        selection_reason, optimization_status, frame_count, all_frames_summary.
    """
    import re as _re
    contents = read_text(output_path)
    trajectory = _parse_trajectory_raw(output_path, contents, include_positions=True)
    frames = trajectory["frames"]

    if not frames:
        return {
            "available": False,
            "reason": "No optimization geometry frames found in output.",
            "optimization_status": trajectory.get("optimization_status", "unknown"),
        }

    opt_status = trajectory["optimization_status"]

    # --- Frame selection ---
    if frame == "best":
        chosen, reason = _select_best_optimization_frame(frames, opt_status)
    elif frame == "last":
        chosen, reason = frames[-1], "last_frame_requested"
    elif frame == "first":
        chosen, reason = frames[0], "first_frame_requested"
    elif frame == "min_energy":
        frames_with_e = [f for f in frames if f.get("energy_hartree") is not None]
        if frames_with_e:
            chosen = min(frames_with_e, key=lambda f: f["energy_hartree"])
            reason = f"min_energy_frame_at_step_{chosen['step']}"
        else:
            chosen, reason = frames[-1], "min_energy_requested_but_no_energy_data_using_last"
    elif isinstance(frame, int):
        if 0 <= frame < len(frames):
            chosen, reason = frames[frame], f"frame_index_{frame}_requested"
        else:
            raise ValueError(f"frame index {frame} out of range (0–{len(frames)-1})")
    else:
        raise ValueError(f"frame must be 'best', 'last', 'first', 'min_energy', or an int; got {frame!r}")

    positions = chosen.get("positions_angstrom")
    labels = chosen.get("labels", [])
    if not positions:
        return {
            "available": False,
            "reason": f"Frame {chosen.get('step')} has no position data (re-parse with include_positions=True).",
            "optimization_status": opt_status,
        }

    # Strip trailing digits from labels for element symbols (e.g. "C1" → "C")
    def _label_to_element(lbl: str) -> str:
        return _re.sub(r"\d+$", "", lbl).capitalize()

    elements = [_label_to_element(lbl) for lbl in labels]

    # --- XYZ text ---
    xyz_lines = [str(len(positions)), f"step={chosen.get('step')} E={chosen.get('energy_hartree')} ({reason})"]
    for elem, (x, y, z) in zip(elements, positions):
        xyz_lines.append(f"{elem:4s} {x:15.8f} {y:15.8f} {z:15.8f}")
    xyz_text = "\n".join(xyz_lines) + "\n"

    # --- NWChem geometry block ---
    # If input_path provided: preserve header+directives+original labels
    if input_path:
        try:
            orig_geom = extract_nwchem_geometry_block(input_path)
            orig_atoms = orig_geom.get("atoms", [])
            if len(orig_atoms) == len(positions):
                atom_dicts = [
                    {"label": a["label"], "element": a["element"],
                     "x": pos[0], "y": pos[1], "z": pos[2]}
                    for a, pos in zip(orig_atoms, positions)
                ]
                nw_block = render_nwchem_geometry_block(
                    orig_geom["header_line"], atom_dicts, directives=orig_geom["directives"]
                )
            else:
                input_path = None  # atom count mismatch, fall through to plain block
        except Exception:  # input file may be missing or have unreadable geometry block
            input_path = None

    if not input_path:
        nw_lines = ["geometry units angstrom"]
        for lbl, (x, y, z) in zip(labels, positions):
            nw_lines.append(f"  {lbl:6s} {x:15.8f} {y:15.8f} {z:15.8f}")
        nw_lines.append("end")
        nw_block = "\n".join(nw_lines)

    # --- All frames summary (without positions, for context) ---
    frames_summary = [
        {
            "index": f["index"],
            "step": f["step"],
            "energy_hartree": f.get("energy_hartree"),
            "gmax": f.get("gmax"),
            "grms": f.get("grms"),
        }
        for f in frames
    ]

    atoms_out = [
        {"element": elem, "label": lbl, "x": pos[0], "y": pos[1], "z": pos[2]}
        for elem, lbl, pos in zip(elements, labels, positions)
    ]

    return {
        "available": True,
        "optimization_status": opt_status,
        "frame_count": len(frames),
        "selected_frame": {
            "index": chosen["index"],
            "step": chosen["step"],
            "energy_hartree": chosen.get("energy_hartree"),
            "gmax": chosen.get("gmax"),
            "grms": chosen.get("grms"),
            "xrms": chosen.get("xrms"),
            "xmax": chosen.get("xmax"),
        },
        "selection_reason": reason,
        "xyz_text": xyz_text,
        "nwchem_geometry_block": nw_block,
        "atom_count": len(positions),
        "elements": elements,
        "atoms": atoms_out,
        "all_frames_summary": frames_summary,
    }





# SCF recovery and property-check drafters moved to
# programs/nwchem/input/scf_recovery.py. Re-exported below for back-compat
# with mcp/nwchem.py and existing callers.
from chemtools.programs.nwchem.input.scf_recovery import (  # noqa: F401
    draft_nwchem_vectors_swap_input,
    draft_nwchem_property_check_input,
    draft_nwchem_scf_stabilization_input,
)




# MCSCF drafters moved to programs/nwchem/input/mcscf.py.
from chemtools.programs.nwchem.input.mcscf import (  # noqa: F401
    draft_nwchem_mcscf_input,
    draft_nwchem_mcscf_retry_input,
)


# Cube drafters moved to programs/nwchem/input/cube.py. Re-exported here for
# compatibility with mcp/nwchem.py and existing Python callers.
from chemtools.programs.nwchem.input.cube import (  # noqa: F401
    draft_nwchem_cube_input,
    draft_nwchem_frontier_cube_input,
)



def draft_initial_geometry(
    atoms: list[str],
    output_path: str,
    comment: str | None = None,
    central_atom: str | None = None,
) -> dict[str, Any]:
    """Generate a plausible initial geometry XYZ file from an atom list.

    Uses covalent radii sums for bond length estimates. Handles diatomics,
    MXn complexes (n=1..6) with symmetric placement, and linear chains.
    Never requires the caller to know bond lengths or 3-D coordinates.

    Parameters
    ----------
    atoms:
        Flat list of element symbols, e.g. ``["Fe", "Cl"]`` or
        ``["Fe", "Cl", "Cl", "Cl", "Cl"]``.  Repeats allowed.
    output_path:
        Where to write the XYZ file.
    comment:
        Optional comment line (second line of XYZ).  Auto-generated if None.
    central_atom:
        Hint for which element is the central/metal atom when building MXn
        geometry.  Inferred automatically if None (the element that appears
        fewest times, or the heaviest unique element).
    """
    _FALLBACK_R = 1.20  # Å, used when element not in table

    def _r(elem: str) -> float:
        return COVALENT_RADII.get(elem, _FALLBACK_R)

    def _bond(a: str, b: str) -> float:
        return _r(a) + _r(b)

    n = len(atoms)
    if n == 0:
        raise ValueError("atoms list must not be empty")

    positions: list[tuple[float, float, float]] = []

    if n == 1:
        positions = [(0.0, 0.0, 0.0)]

    elif n == 2:
        # Diatomic along z
        bl = _bond(atoms[0], atoms[1])
        positions = [(0.0, 0.0, 0.0), (0.0, 0.0, bl)]

    else:
        # Try to identify central atom for MXn placement
        from collections import Counter
        counts = Counter(atoms)

        # Determine central atom: explicit hint, or fewest occurrences, or heaviest element
        from chemtools.core.common import ELEMENT_TO_Z
        if central_atom and central_atom in counts:
            center = central_atom
        elif len(counts) > 1:
            min_count = min(counts.values())
            candidates = [e for e, c in counts.items() if c == min_count]
            # Among candidates pick heaviest (most likely the metal)
            center = max(candidates, key=lambda e: ELEMENT_TO_Z.get(e, 0))
        else:
            center = atoms[0]

        ligands = [a for a in atoms if a != center]
        n_lig = len(ligands)

        # Unique ligand bond length (use first ligand type for MXn)
        r_ml = _bond(center, ligands[0]) if ligands else 2.0

        # Symmetric ligand positions around center at origin
        # Elements with lone pairs that produce bent AX2 geometries
        _BENT_ANGLE: dict[str, float] = {
            "O": 104.5, "S": 92.0, "Se": 91.0, "Te": 90.0,
            "N": 107.0, "P": 93.5, "As": 91.8, "Sb": 91.6,
            "C": 109.5, "Si": 109.5, "Ge": 91.0,
        }
        two_pi = 2.0 * math.pi
        if n_lig == 1:
            lig_coords = [(0.0, 0.0, r_ml)]
        elif n_lig == 2:
            if center in _BENT_ANGLE:
                angle_deg = _BENT_ANGLE[center]
                half = math.radians(angle_deg / 2)
                lig_coords = [
                    ( r_ml * math.sin(half),  r_ml * math.cos(half), 0.0),
                    (-r_ml * math.sin(half),  r_ml * math.cos(half), 0.0),
                ]
            else:
                lig_coords = [(0.0, 0.0, -r_ml), (0.0, 0.0, r_ml)]
        elif n_lig == 3:
            # Elements with lone pairs that produce pyramidal AX3 geometries
            _PYRAMIDAL_CENTERS = {"N", "P", "As", "Sb", "Bi", "S", "Se", "Te"}
            if center in _PYRAMIDAL_CENTERS:
                # Pyramidal: ligands below the central atom plane
                # Use ~107° bond angle for N, ~93° for heavier pnictogens
                pyr_angle_deg = {"N": 107.0, "P": 93.5, "As": 91.8, "Sb": 91.6, "Bi": 90.5}.get(center, 95.0)
                half_angle = math.radians(pyr_angle_deg / 2)
                # Place ligands in a cone below center
                z_lig = -r_ml * math.cos(half_angle)
                r_xy = r_ml * math.sin(half_angle)
                lig_coords = [
                    (r_xy * math.cos(k * two_pi / 3), r_xy * math.sin(k * two_pi / 3), z_lig)
                    for k in range(3)
                ]
            else:
                # Trigonal planar (e.g. BH3, AlCl3)
                lig_coords = [
                    (r_ml * math.cos(k * two_pi / 3), r_ml * math.sin(k * two_pi / 3), 0.0)
                    for k in range(3)
                ]
        elif n_lig == 4:
            s = r_ml / math.sqrt(3)
            lig_coords = [(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]
        elif n_lig == 5:
            # Square pyramidal
            lig_coords = [
                (r_ml, 0.0, 0.0), (-r_ml, 0.0, 0.0),
                (0.0, r_ml, 0.0), (0.0, -r_ml, 0.0),
                (0.0, 0.0, r_ml),
            ]
        elif n_lig == 6:
            lig_coords = [
                (r_ml, 0.0, 0.0), (-r_ml, 0.0, 0.0),
                (0.0, r_ml, 0.0), (0.0, -r_ml, 0.0),
                (0.0, 0.0, r_ml), (0.0, 0.0, -r_ml),
            ]
        else:
            # Linear chain along z for anything larger
            lig_coords = [(0.0, 0.0, (k + 1) * r_ml) for k in range(n_lig)]

        # Reassemble: center first, then ligands in original order
        center_pos = (0.0, 0.0, 0.0)
        ordered_atoms: list[str] = [center]
        ordered_pos: list[tuple[float, float, float]] = [center_pos]
        lig_iter = iter(lig_coords)
        for a in atoms:
            if a == center and center not in ordered_atoms[1:]:
                continue  # already placed center
            else:
                ordered_atoms.append(a)
                ordered_pos.append(next(lig_iter))
        atoms = ordered_atoms
        positions = ordered_pos

    # Build XYZ content
    auto_comment = comment or f"Initial geometry guess — {' '.join(atoms)}"
    lines = [str(len(atoms)), auto_comment]
    bond_summary: list[dict[str, Any]] = []
    for elem, (x, y, z) in zip(atoms, positions):
        lines.append(f"{elem:4s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")

    # Summarise bond lengths for the return value
    if len(atoms) == 2:
        bond_summary.append({
            "atoms": f"{atoms[0]}-{atoms[1]}",
            "length_angstrom": round(
                math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[0], positions[1]))), 4
            ),
            "source": "covalent_radii_sum",
        })

    xyz_text = "\n".join(lines) + "\n"
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xyz_text, encoding="utf-8")

    return {
        "output_file": str(out.resolve()),
        "n_atoms": len(atoms),
        "atoms": atoms,
        "positions_angstrom": [list(p) for p in positions],
        "bond_lengths_used": bond_summary,
        "comment": auto_comment,
        "note": (
            "Geometry is a covalent-radii guess — suitable for initial optimization only. "
            "Always run geometry optimization before any correlated calculation."
        ),
        "next_steps": [
            f"Call create_nwchem_dft_workflow_input with geometry_file='{out.resolve()}' "
            "to build the optimization input.",
            "Run lint_nwchem_input on the generated input before launching.",
        ],
    }



__all__ = [
    "extract_nwchem_geometry",
    "draft_initial_geometry",
]
