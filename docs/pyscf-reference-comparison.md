# PySCF reference-calculation comparison

`compare_pyscf_reference_calculation` compares a completed
`run_pyscf_single_point` result with one caller-declared reference record. It
does not parse a production output file or infer correspondence from filenames.
The caller states the reference label, Cartesian geometry, method, basis, XC,
density-fitting choice, charge, multiplicity, electron count, SCF outcome, and
total energy in Hartree.

The report compares the supplied geometry without translation or rotation,
using a `1e-6` angstrom coordinate tolerance and atom-order-independent element
matching. It separately reports every calculation-setting comparison, the SCF
outcome, electron count, and `PySCF − reference` energy in Hartree and
kcal/mol. A difference is evidence to inspect, not a correctness verdict.

When the completed PySCF result contains a written density CUBE and the
reference record declares a density CUBE and its scalar unit, the report calls
`compare_cube_densities`. A caller may also provide one written PySCF
`orbital_cubes` artifact and one reference orbital CUBE with explicit labels;
the report then calls `compare_cube_orbitals`. Both field comparisons retain their
strict same-grid and geometry requirements. Missing artifacts remain
`not_compared`, rather than being treated as field agreement.

The first scope is one molecular single point against one reference record.
It does not establish that two methods are equivalent, select a preferred
energy, compare a degenerate orbital subspace, or create/run a production
calculation.
