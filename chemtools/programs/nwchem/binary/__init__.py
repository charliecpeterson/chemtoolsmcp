"""NWChem binary / on-disk file readers.

Modules here read NWChem-produced files that aren't the main output text:

  * hessian.py     `.hess` (ASCII lower-triangle Hessian, Eh/bohr^2)
  * (planned) movecs.py     `.movecs` (Fortran-unformatted MO file —
                              currently lives in parse/tce.py)
  * (planned) drv_hessian.py `.drv.hess` (binary driver Hessian)
  * (planned) fdrst.py        `.fdrst` (frequency restart bookkeeping)

These are routed through the Program plugin's `binary` sub-protocol
(see chemtools/core/program.py).
"""
