"""DIRAC HDF5 checkpoint reader.

DIRAC ≥ 22 writes per-run HDF5 files alongside the text output. The schema
this module reads (verified against DIRAC 22 / 25 on real fixtures):

    input/
      aobasis/<center>/   angular, contractions, exponents, n_ao, n_cont,
                          n_prim, n_shells, orbmom, center (xyz × n_atoms)
      molecule/           geometry (3×n_atoms), n_atoms, nuc_charge, symbols
      dirac_inp           echoed .inp lines  (S100 array)
      molecule_inp        echoed .mol lines (S100 array)
    result/
      execution/status
      operators/ao_matrices/   OVERLAP TFFT, BETAMAT FFFT, MOLFIELDTFFT
      symmetry/                inversion, isymax (6,), maxopr
      wavefunctions/scf/
        energy            scalar (Hartree)
        mobasis/
          eigenvalues     (n_mo_total,)   orbital energies (Ha)
          occupations     (n_mo_total,)   per-orbital occupation
          orbitals        (n_mo_total × n_basis_total × nz,)   MO coefficients
          shell_id        (n_mo_total,)   shell classification index
          symmetry        (n_mo_total,)   per-MO irrep index (1..8 for D2h)
          n_basis         (n_fsym,)       basis dim per fermion symmetry
          n_mo            (n_fsym,)       MO count per fermion symmetry
          n_po            (n_fsym,)       positive-energy MO count per fsym
          n_fsym          scalar          number of fermion symmetries (2)
          nz              scalar          quaternion factor (1 real, 2 ARH, 4 full)

Negative-energy orbitals (the positronic / Dirac sea side of the 4c
spectrum) are tagged with negative symmetry / shell_id values. Routine
chemistry analysis filters them out via ``include_negative=False`` (the
default for the orbital summary functions).
"""

from __future__ import annotations

from typing import Any

try:
    import h5py  # type: ignore
    H5PY_AVAILABLE = True
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore
    H5PY_AVAILABLE = False


def _require_h5py() -> None:
    if not H5PY_AVAILABLE:
        raise ImportError(
            "Reading DIRAC .h5 checkpoints requires the h5py package. "
            "Install with `pip install chemtools[dirac]` or "
            "`pip install h5py`."
        )


def read_metadata(path: str) -> dict[str, Any]:
    """Read top-level DIRAC HDF5 metadata: version, n_atoms, symmetry, MO counts."""
    _require_h5py()
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        version = attrs.get("DIRAC_VERSION")
        if isinstance(version, bytes):
            version = version.decode().strip()

        mol = f["input/molecule"]
        sym = f.get("result/symmetry")
        scf_e = None
        mb = None
        if "result/wavefunctions/scf" in f:
            scf = f["result/wavefunctions/scf"]
            scf_e = float(scf["energy"][0])
            mb = scf["mobasis"]

        meta: dict[str, Any] = {
            "path": path,
            "version": version,
            "n_atoms": int(mol["n_atoms"][0]),
            "scf_energy_hartree": scf_e,
        }
        if sym is not None:
            meta["inversion_symmetry"] = bool(sym["inversion"][0] != 0)
            meta["isymax"] = sym["isymax"][:].tolist()
            meta["max_operators"] = int(sym["maxopr"][0])
        if mb is not None:
            meta["n_fermion_symmetries"] = int(mb["n_fsym"][0])
            meta["n_mo_per_fsym"] = mb["n_mo"][:].tolist()
            meta["n_basis_per_fsym"] = mb["n_basis"][:].tolist()
            meta["n_pos_energy_per_fsym"] = mb["n_po"][:].tolist()
            meta["nz"] = int(mb["nz"][0])
        return meta


def read_geometry(path: str) -> dict[str, Any]:
    """Read molecule geometry from a DIRAC .h5. Coordinates are returned in angstrom.

    DIRAC stores ``input/molecule/geometry`` in angstrom regardless of the
    .mol file's input units flag — verified against a 25.0 run where the
    .mol declared bohr coords and the h5 stored the bohr→Å-converted values.
    """
    _require_h5py()
    with h5py.File(path, "r") as f:
        mol = f["input/molecule"]
        geom = mol["geometry"][:]      # shape (3 × n_atoms,) flattened
        nuc = mol["nuc_charge"][:]
        symbols = mol["symbols"][:]
        n_atoms = int(mol["n_atoms"][0])

        # geometry is stored as (3*n_atoms,) — reshape to (n_atoms, 3).
        # The actual layout in DIRAC is x_a, y_a, z_a, x_b, y_b, z_b, ...
        coords = geom.reshape(n_atoms, 3) if len(geom) == 3 * n_atoms else geom.reshape(-1, 3)

        atoms: list[dict[str, Any]] = []
        for i in range(n_atoms):
            sym_raw = symbols[i] if i < len(symbols) else (
                symbols[0] if len(symbols) else b""
            )
            label = (sym_raw.decode() if isinstance(sym_raw, bytes) else str(sym_raw)).strip()
            z = float(nuc[i]) if i < len(nuc) else float(nuc[0])
            atoms.append({
                "label": label,
                "element": _z_to_element(int(z)) or label,
                "nuclear_charge": z,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
            })

        return {
            "path": path,
            "n_atoms": n_atoms,
            "atoms": atoms,
            "units": "angstrom",
        }


def read_total_energy(path: str) -> float | None:
    """Read the converged SCF energy from a DIRAC .h5, in Hartree."""
    _require_h5py()
    with h5py.File(path, "r") as f:
        if "result/wavefunctions/scf/energy" not in f:
            return None
        return float(f["result/wavefunctions/scf/energy"][0])


def read_orbital_summary(
    path: str,
    *,
    include_negative_energy: bool = False,
    only_occupied: bool = False,
    fractional_only: bool = False,
) -> list[dict[str, Any]]:
    """Read per-MO data: index, fermion symmetry, irrep, energy, occupation, shell class.

    Parameters
    ----------
    include_negative_energy:
        DIRAC's 4-component spectrum includes negative-energy (positronic /
        "Dirac sea") orbitals. They are tagged with negative symmetry /
        shell_id. Default False — drop them for chemistry analysis.
    only_occupied:
        Drop unoccupied / virtual orbitals (occupation ≤ 1e-6).
    fractional_only:
        Return only orbitals with fractional occupation (the AOC open-shell
        signature). Useful for "which orbitals carry the open electrons?"
        questions.
    """
    _require_h5py()
    with h5py.File(path, "r") as f:
        if "result/wavefunctions/scf/mobasis" not in f:
            return []
        mb = f["result/wavefunctions/scf/mobasis"]
        energies = mb["eigenvalues"][:]
        occs = mb["occupations"][:]
        sym = mb["symmetry"][:]
        shell = mb["shell_id"][:]
        n_fsym = int(mb["n_fsym"][0])
        n_mo_per_fsym = mb["n_mo"][:].tolist()
        n_po_per_fsym = mb["n_po"][:].tolist()

        # The flat orbital array is ordered: fsym 1 (all MOs), fsym 2 (all MOs), ...
        # Within each fsym, negative-energy (positronic / Dirac-sea) orbitals
        # appear first when DIRAC tracks them separately (n_po < n_mo).
        # However, for atomic runs (and other cases) DIRAC writes n_po=0
        # for ALL fermion symmetries even though every orbital is electronic.
        # Physics-based fallback: anything with eigenvalue < -2c² (≈ -37500 Ha
        # in a.u.) is positronic; everything above is electronic.
        # Speed-of-light squared in atomic units:
        _NEG_ENERGY_THRESHOLD = -2.0 * (137.035999084 ** 2)  # ≈ -37557.7 Ha

        results: list[dict[str, Any]] = []
        global_idx = 0
        for fs_idx in range(n_fsym):
            n_mo = n_mo_per_fsym[fs_idx]
            n_po = n_po_per_fsym[fs_idx]
            n_neg_index = n_mo - n_po if n_po > 0 else 0
            pos_counter = 0
            for local in range(n_mo):
                e = float(energies[global_idx])
                o = float(occs[global_idx])
                s = int(sym[global_idx])
                sh = int(shell[global_idx])

                # Primary classifier: index-based when DIRAC reports n_po > 0.
                # Fallback: energy threshold for atomic-style runs where n_po=0.
                if n_po > 0:
                    is_negative = local < n_neg_index
                else:
                    is_negative = e < _NEG_ENERGY_THRESHOLD

                if not is_negative:
                    pos_counter += 1
                    pos_index = pos_counter
                else:
                    pos_index = None

                global_idx += 1

                if is_negative and not include_negative_energy:
                    continue
                if only_occupied and o <= 1e-6:
                    continue
                if fractional_only and not _is_fractional(o):
                    continue

                results.append({
                    "global_index": global_idx - 1,
                    "fermion_symmetry": fs_idx + 1,
                    "positive_energy_index": pos_index,
                    "irrep": s,
                    "shell_class": _classify_shell(o, is_negative),
                    "shell_id_raw": sh,
                    "energy_hartree": e,
                    "occupation": o,
                    "is_negative_energy": is_negative,
                })
        return results


def read_mo_coefficients(
    path: str,
    *,
    mo_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Read MO coefficients from a DIRAC .h5.

    Returns the full coefficient array (shape ``(n_mo_total, n_basis_total, nz)``)
    by default, or a slice if ``mo_indices`` is provided. The flat layout in
    HDF5 has nz "quaternion" components stacked on the basis axis — chemistry
    callers usually want the real part only (``coeffs[..., 0]``).
    """
    _require_h5py()
    with h5py.File(path, "r") as f:
        mb = f["result/wavefunctions/scf/mobasis"]
        flat = mb["orbitals"][:]
        n_mo_per_fsym = mb["n_mo"][:].tolist()
        n_basis_per_fsym = mb["n_basis"][:].tolist()
        nz = int(mb["nz"][0])
        n_mo_total = sum(n_mo_per_fsym)
        n_basis_total = sum(n_basis_per_fsym)

        # DIRAC stores per-fermion-symmetry blocks; full coefficient array is
        # the union, shape (n_mo, n_basis, nz). Layout assumes fsym1 then fsym2.
        # We expose it flat; callers that need block-diagonal structure can
        # slice using the per-fsym counts also returned here.
        coeffs = flat.reshape(n_mo_total, n_basis_total, nz) \
            if len(flat) == n_mo_total * n_basis_total * nz else flat

        if mo_indices is not None:
            coeffs = coeffs[mo_indices]

        return {
            "path": path,
            "coefficients": coeffs,
            "n_mo": n_mo_per_fsym,
            "n_basis": n_basis_per_fsym,
            "nz": nz,
            "mo_indices": mo_indices,
        }


def read_aobasis_info(path: str) -> dict[str, Any]:
    """Read AO basis primitive + contraction info per atomic center.

    Returns a dict keyed by atomic-center index (1-based) with each entry's
    angular, contractions, exponents, orbmom, n_ao / n_cont / n_prim /
    n_shells arrays. Useful for advanced analyses (e.g. AO-character
    decomposition); not loaded by routine summary calls.
    """
    _require_h5py()
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        if "input/aobasis" not in f:
            return out
        for key in f["input/aobasis"].keys():
            try:
                idx = int(key)
            except ValueError:
                continue
            grp = f[f"input/aobasis/{key}"]
            entry: dict[str, Any] = {}
            for fld in (
                "angular", "contractions", "exponents", "orbmom",
                "n_ao", "n_cont", "n_prim", "n_shells", "center", "aobasis_id",
            ):
                if fld in grp:
                    arr = grp[fld][:]
                    entry[fld] = arr.tolist() if arr.size <= 200 else f"<array shape={arr.shape}>"
            out[idx] = entry
    return out


def _classify_shell(occupation: float, is_negative_energy: bool) -> str:
    """Translate a DIRAC orbital's occupation + negative-energy flag into a
    chemistry-meaningful class label.

    The 4-component spectrum splits into positive-energy (electronic) and
    negative-energy (positronic / Dirac sea) blocks; negative-energy
    orbitals always count as ``negative_energy`` regardless of occupation.
    Within positive-energy: occupation ≈ 1.0 (Kramers-pair scaled) →
    ``closed``, fractional → ``open`` (AOC), near-zero → ``virtual``.
    """
    if is_negative_energy:
        return "negative_energy"
    if occupation > 1.0 - 1e-4:
        return "closed"
    if occupation > 1e-4:
        return "open"
    return "virtual"


def _is_fractional(occ: float) -> bool:
    return 1e-4 < occ < 1.0 - 1e-4


def _z_to_element(z: int) -> str | None:
    from chemtools.core.common import ATOMIC_SYMBOLS
    return ATOMIC_SYMBOLS.get(z)
