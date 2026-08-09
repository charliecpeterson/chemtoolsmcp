"""Resolve bundled NWChem data paths and supported environment overrides."""

from __future__ import annotations

import os

from chemtools.programs.nwchem.input.basis_library import (
    bundled_basis_library_path,
)


DEFAULT_BASIS_LIBRARY = bundled_basis_library_path()


def basis_library_path(path: str | None = None) -> str:
    if path:
        return path
    return os.environ.get(
        "CHEMTOOLS_BASIS_LIBRARY",
        str(DEFAULT_BASIS_LIBRARY),
    )
