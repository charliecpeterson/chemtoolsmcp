# ORCA experimental inputs

These inputs build the first ORCA 6.1.1 reference corpus. They are small on
purpose: each one adds a distinct output or execution behavior that Chemtools
may need to identify and parse.

`h2_hf_def2_svp.inp` follows the ORCA 6.1 quick-start example and is the serial
installation smoke test. `water_r2scan3c_opt_freq.inp` adds a geometry
optimization followed by frequencies. `o2_triplet_pbe0.inp` adds unrestricted
open-shell SCF output. `formaldehyde_wb97x_d4.inp` adds a small organic
range-separated-hybrid case. `cucl4_doublet_pbe0.inp` adds an idealized
square-planar open-shell transition-metal complex. `uranyl_singlet_pbe0_zora.inp`
adds an idealized linear f-block complex with scalar relativity and an explicit
element-specific SARC basis assignment. `cucl4_doublet_interrupted.inp` caps
the Cu calculation at three SCF iterations to produce a controlled restart
artifact and nonconvergence output. `cucl4_doublet_restart.inp` reads that
artifact under a different calculation basename and completes the SCF.
`fe_macrocycle_sextet_b3lyp_diis.inp` ports a larger Fe model whose analogous
NWChem calculation failed to converge, and bounds an ORCA DIIS/SOSCF baseline
with automatic TRAH disabled. `fe_macrocycle_sextet_b3lyp_trah.inp` runs the
same model with manual TRAH from a fresh guess for a controlled convergence
comparison. `water_pentamer_qmmm.inp` treats the central water with PBEh-3c
and four complete neighboring waters with a small fixed-charge MM file. It has
no covalent QM/MM boundary or link atoms. `nacl_crystalprep.inp` builds the
10x10x10 NaCl embedding used by `nacl_ionic_crystal_qmmm.inp`; the latter uses
one Na and its six nearest Cl neighbors as the -5 QM core, one SDD cECP layer,
and self-consistent crystal charges.

`alpha_glycine_crystalprep.inp` builds a 5x5x5 molecular-crystal embedding
from the reduced COD 2310002 structure. `alpha_glycine_mol_crystal_qmmm.inp`
treats one neutral glycine molecule with PBEh-3c and embeds it in 1,140 MM
point charges whose values are converged against the QM CHELPG charges. This
path requires xTB 6.7.1 or later as `otool_xtb` in the ORCA directory.

`n2_stretched_casscf_nevpt2.inp` uses CASSCF(6,6) followed by strongly
contracted NEVPT2 on N2 at 1.8 angstrom. `formaldehyde_mrci.inp` follows the
ORCA MRCI tutorial and correlates three CAS(2,2)-based symmetry and
multiplicity blocks with a Davidson correction. `formaldehyde_pbe0_tddft.inp`
requests six singlet and six triplet TD-DFT roots plus natural transition
orbitals. `formaldehyde_eom_ccsd.inp` follows the EOM-CCSD tutorial with four
roots. `n2_excited_casscf_caspt2.inp` uses a three-root state average followed
by FIC-CASPT2. Its run completed all roots, but retained five active-orbital
constraint warnings; that makes it useful for checking that normal termination
does not erase method-level warnings.

The NaCl preparation step must find `orca_mm` by basename, so the ORCA
directory must be on `PATH` when `orca_crystalprep` runs. The preparation
creates the PDB and `ORCAFF.prms` files referenced by the calculation input.
The successful run and generated side files are pinned under
`/home/charlie/input_examples/orca/13_nacl_ionic_crystal_qmmm`.

ORCA 6.1.1 CrystalPrep generated an incomplete molecular-crystal template for
alpha-glycine: `NUnitCellAtoms` and `QMAtoms` were placeholders, and its XYZ
file lacked the required comment line. The checked-in calculation input fixes
the first two fields. The corrected XYZ, original generated files, successful
output, and converged charges are pinned under
`/home/charlie/input_examples/orca/14_alpha_glycine_mol_crystal_qmmm`.

The multireference and excited-state runs are pinned under numbered directories
15 through 19 in `/home/charlie/input_examples/orca`.

The formaldehyde ORCA_ESD suite adds harmonic vibronic absorption,
fluorescence, phosphorescence, resonance Raman, and radiative-rate examples.
Its required run order, Hessian dependencies, observed spectra, and model
limits are in `formaldehyde_esd_README.md`. The completed files are pinned in
`/home/charlie/input_examples/orca/20_formaldehyde_esd`.

These method and geometry choices are parser and execution experiments, not
general method recommendations. The transition-metal and f-block cases remain
exploratory until their scientific setup and output have been reviewed. The
water MM parameters are an illustrative fixed-charge model for exercising
ORCA's additive interface, not a production solvent model.
