# PySCF single-point fixture curation

The bounded `run_pyscf_single_point` operation accepts molecular RHF, UHF,
RKS, and UKS single points with typed Cartesian atoms. Its response records
process execution separately from the SCF result. This document records the
evidence used to promote the two initial PySCF knowledge cards.

The reviewed fixture corpus is
[`tests/fixtures/pyscf/single_point_cases.json`](../tests/fixtures/pyscf/single_point_cases.json).
It was recorded using PySCF 2.13.1 on Python 3.12.13 in the
`chemtools-science` environment on linux-4090.

## Completion and convergence are separate facts

`h2_stretched_rhf_cycle_limit` intentionally limits a stretched H2 RHF/STO-3G
calculation to one SCF cycle. The companion runner completes normally and
returns a finite energy, but reports `scf.converged: false` and the
`scf_not_converged` warning. A completed process therefore cannot make that
energy eligible as a converged SCF result.

This evidence covers only the companion runner's bounded molecular
single-point contract. It does not decide whether a converged result uses an
appropriate model chemistry, basis, geometry, or electronic state.

## Electron count and spin must be compatible

`h2_uhf_doublet_electron_spin_inconsistent` is a typed UHF request for neutral
H2 with doublet multiplicity. It passes request-boundary validation because an
unrestricted method can request a non-singlet multiplicity, then PySCF refuses
to build the molecule because two electrons cannot have spin 1. The response
is a `runtime_error`, distinct from both request validation and an
unconverged SCF.

This is an electronic-consistency check, not a method-selection rule. A
request that passes it may still have the wrong charge, multiplicity, spin
state, or reference for the scientific problem.
