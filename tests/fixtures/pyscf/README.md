# PySCF companion-runtime fixtures

This is a small, reviewed corpus for the bounded molecular single-point
operation. Each entry retains the exact request, a deliberately limited set
of expected result facts, and the purpose of the case. It is not a benchmark
suite or a substitute for a method-appropriate production calculation.

The reference outcomes were recorded with PySCF 2.13.1 on Python 3.12.13 in
the `chemtools-science` environment on linux-4090. Energies are regression
evidence for the stated geometry, method, basis, and implementation version.
They must not be compared with results from a different Hamiltonian, basis,
functional, relativistic treatment, or program as though they establish a
correctness winner.

`h2_rhf_sto3g`, `o2_uhf_triplet_sto3g`, `h2_rks_pbe_sto3g`, and
`h_uks_pbe_sto3g` exercise all four supported SCF entry points. The remaining
cases make a separate distinction between a completed process with an
unconverged SCF and a PySCF runtime refusal caused by an inconsistent
electron-count and spin request.
