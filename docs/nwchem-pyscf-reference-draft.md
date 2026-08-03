# NWChem-to-PySCF reference drafts

`draft_nwchem_pyscf_reference` prepares evidence for
`compare_pyscf_reference_calculation`. It reads one NWChem input and, when
available, its output. It does not run either program.

The result has `status: "drafted"`, a `reference_draft`, field-level source
records, and `missing_required_fields`. Pass `reference_draft` to the PySCF
comparison only when `comparison_ready` is true.

## Extracted evidence

- Cartesian geometry only when the input contains exactly one geometry block
  and declares `units angstrom` or a Bohr spelling. The draft normalizes the
  coordinates to angstrom.
- A basis only when all parsed `library` declarations name one basis. Manual
  definitions and multiple names remain unresolved because the bounded PySCF
  runner currently accepts one basis string.
- Charge and multiplicity from the input parser.
- The raw `xc` declaration from the selected DFT block as source evidence.
- SCF convergence and total energy only when the supplied output reaches a
  completed SCF record. A failed or incomplete calculation keeps the energy
  unresolved.

## Required declarations

NWChem's `scf` and `dft` task labels do not identify the matching PySCF SCF
flavour. Likewise, NWChem fitting directives cannot safely be treated as the
PySCF `density_fit` setting. The adapter requires the caller to supply:

- `pyscf_method`: `rhf`, `uhf`, `rks`, or `uks`.
- `pyscf_xc`: the PySCF functional for RKS or UKS. The NWChem `xc` line is
  retained for review but is not assumed to be semantically identical.
- `density_fit`: the Boolean setting for the intended PySCF run.
- `electron_total`: the effective electron count to compare.

The adapter deliberately does not calculate the last value from elements and
charge: ECPs and nonstandard center charges can change the count. It is an
evidence adapter, not an automatic method-conversion or correctness tool.

```json
{
  "input_file": "h2.nw",
  "output_file": "h2.out",
  "pyscf_method": "rhf",
  "pyscf_xc": null,
  "density_fit": false,
  "electron_total": 2
}
```

With a declared Cartesian input, one `sto-3g` library basis, and converged
output, this yields a `reference_draft` that is ready to supply as the
`reference` argument to `compare_pyscf_reference_calculation`.
