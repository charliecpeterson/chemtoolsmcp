# Matched NWChem-to-PySCF single points

`run_nwchem_pyscf_matched_reference` is the local-execution completion of the
NWChem/PySCF comparison path. It performs three fixed steps:

1. Build a NWChem reference draft from one input/output pair.
2. Run the existing bounded PySCF single-point operation with the draft's
   normalized geometry and declared settings.
3. Return `compare_pyscf_reference_calculation` evidence for the two results.

The tool does not start PySCF when the NWChem reference is incomplete. A
single Cartesian geometry block with explicit units, one unambiguous library
basis, charge, multiplicity, a converged NWChem SCF result, and its final
energy are required.

The caller must declare `pyscf_method`, `density_fit`, and `electron_total`.
For RKS and UKS, `pyscf_xc` is required too. NWChem's `xc` text is retained as
source evidence, but is never assumed to be a semantically identical PySCF
functional.

```json
{
  "input_file": "h2.nw",
  "output_file": "h2.out",
  "working_directory": "/path/to/chemtools-scratch/h2-pyscf",
  "pyscf_method": "rhf",
  "density_fit": false,
  "electron_total": 2
}
```

Use `dry_run: true` to validate the reference and render the fixed PySCF
launch without executing it. The completed result records the NWChem draft,
the PySCF launch/result, and the comparison report. It does not choose a
scientifically correct result from the energy difference.

To include total-density evidence, provide both `reference_density_cube` and
`density_cube_grid_points`. The former is the already-written NWChem total
density CUBE and its scalar unit; the latter asks PySCF to write its own total
density CUBE. The completed comparison attaches the existing same-grid density
report. It reports `not_comparable` rather than resampling when the CUBE box,
point grid, voxel vectors, or nuclear geometry differ. Orbital CUBE comparison
remains an explicit follow-up call because orbital correspondence must still be
caller-declared.

The current PySCF writer fixes its box from the molecular coordinates with a
3 bohr margin on every side, and its `density_cube_grid_points` value is the
number of points including both endpoints. Set
`pyscf_compatible_grid_points` to the same value on
`draft_nwchem_cube_input` (or `draft_nwchem_frontier_cube_input`) to generate
the matching NWChem `dplot limitxyz units bohr` box. The drafter requires one
Cartesian geometry block with explicit units, uses the same 3 bohr margin, and
writes one fewer NWChem spacings in each direction. It also adds NWChem's
`nocenter`, `noautosym`, and `noautoz` controls so the CUBE retains the input
Cartesian frame. Generate the NWChem CUBE from that draft before starting the
matched-reference operation.
