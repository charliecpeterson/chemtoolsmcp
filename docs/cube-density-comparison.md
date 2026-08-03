# Same-grid electron-density comparison

`compare_cube_densities` compares two caller-declared electron-density CUBE
files. It is a field-comparison step for a deliberately matched PySCF and
production-code calculation, not a calculation runner and not a correctness
verdict.

The tool requires each density-value unit because the CUBE grid-coordinate
unit does not establish the scalar field unit. It accepts
`electron_per_bohr3` and `electron_per_angstrom3`, converts both fields to
electrons per cubic angstrom, then reports integrated electrons, integrated
electron difference, L1 and relative-L1 difference, L2 difference, RMS
density difference, and maximum absolute density difference. Integral-derived
values use a uniform trapezoidal rule because CUBE grids include box-boundary
points. They are grid-resolution evidence, not independently validated
electron counts.

It compares values only when the two inputs have the same grid shape, origin,
and voxel vectors within fixed floating-point tolerances, and the same nuclear
geometry within 1e-6 angstrom. It does not interpolate or resample. An
explicitly identified orbital, potential, or spin-density CUBE is refused. A
CUBE whose header does not identify its field type can still be compared only
because the caller declared it as density; the returned warning preserves that
uncertainty.

The CUBE atom-record charge column is not used to establish nuclear geometry:
NWChem writes nuclear charges there while PySCF writes zeroes. Matching atomic
numbers and positions are required; a differing charge column is retained as a
warning.

The initial 2,000,000-voxel ceiling keeps the in-memory comparison bounded.
Use the same coordinate box and grid for both source programs. Set
`density_cube_grid_points` on `run_pyscf_single_point` to write a fixed-name
total-density CUBE after a converged PySCF SCF. The result records its path,
SHA-256, grid dimensions, and `electron_per_bohr3` value unit. The existing
NWChem `draft_nwchem_cube_input` helper can draft a total-density DPLOT CUBE.
`run_nwchem_pyscf_matched_reference` now composes NWChem/PySCF settings and
energy evidence and can include density CUBE evidence when the caller provides
both `reference_density_cube` and `density_cube_grid_points`. The NWChem CUBE
must already exist. The operation passes the pair to this comparator and keeps
an incompatible grid as `not_comparable`; it does not resample it.

For a practical matched grid, PySCF's current molecular CUBE writer uses a
3 bohr margin around the atom-coordinate extrema and treats the requested
grid value as endpoint-inclusive point counts. NWChem `dplot limitxyz` uses
spacings, so a matching NWChem input needs one fewer spacing per direction and
the corresponding geometry-derived bohr bounds. Pass the same point count as
`pyscf_compatible_grid_points` to `draft_nwchem_cube_input` or
`draft_nwchem_frontier_cube_input`; it generates those bounds from one
explicit-unit Cartesian geometry and returns the box as `cube_grid` evidence.
The draft adds `nocenter`, `noautosym`, and `noautoz` to retain that input
frame in NWChem's CUBE header.
