"""NWChem binary / on-disk file readers.

Modules here read NWChem-produced files that aren't the main output text:

  * hessian.py reads ASCII lower-triangle `.hess` files.
  * movecs.py reads supported Fortran-unformatted `.movecs` files.

These are routed through the NWChem backend's `binary` provider
(see chemtools/core/program.py).
"""
