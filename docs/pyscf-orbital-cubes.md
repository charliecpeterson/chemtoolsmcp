# Selected PySCF orbital CUBEs

`run_pyscf_single_point` can write up to eight selected molecular-orbital
CUBEs after converged SCF. Supply `orbital_cube_grid_points` from 20 through
120 and an `orbital_cube_requests` list. Each selector has a zero-based
PySCF `orbital_index` and a spin channel.

RHF and RKS requests must select `restricted` orbitals. UHF and UKS requests
must select `alpha` or `beta` orbitals. The runner rejects duplicate selectors
and derives every output filename from the request and selector. It never
accepts a caller-chosen artifact path.

Each written artifact records its path, SHA-256, spin, index, derived label,
MO energy, occupation, grid size, and the CUBE value unit
`bohr_to_minus_three_halves`. An out-of-range index or a write failure becomes
an artifact-level failure while preserving the completed SCF result. An
unconverged SCF records requested artifacts as `not_written`.

The generated artifact can be supplied directly to
`compare_pyscf_reference_calculation` as `pyscf_orbital_cube`, after choosing
the reference orbital CUBE and label. Selectors across a shared grid can also
be passed to `compare_cube_orbital_subspaces` for principal-angle comparison.
The tools do not infer correspondence, occupation equivalence, or physical
degeneracy from MO indices.
