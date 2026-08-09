# ORCA integration checklist

Status: experimental

ORCA support starts from real ORCA 6.1.1 inputs and outputs. Parser fields,
input advice, and launch behavior stay provisional until the corresponding
case has run and its evidence has been reviewed.

## Runtime baseline

- [x] Replace the ARM64 download with the x86-64 ORCA 6.1.1 distribution for
      linux-4090.
- [x] Confirm that the serial `orca` driver is an x86-64 executable and that
      its direct shared-library dependencies resolve.
- [x] Run the documented H2 HF/DEF2-SVP serial smoke case.
- [x] Record the exact executable, input and output hashes, return code,
      termination marker, final energy, and generated artifacts.
- [ ] Provide OpenMPI 4.1.8 before testing parallel ORCA. The selected system
      launcher is currently Intel MPI 2021.11 and must not launch this ORCA
      distribution.
- [ ] Repeat one cheap case with ORCA parallelism after the OpenMPI runtime is
      available. Invoke the ORCA driver directly with its full path, never
      through `mpirun`.

## Case ladder

- [x] Closed-shell single point: H2 manual smoke case.
- [x] Small-molecule geometry optimization: use water as the compact molecule
      and verify energy and geometry convergence.
- [x] Small-molecule frequency calculation: verify completion, mode count,
      imaginary-mode handling, and thermochemistry fields.
- [x] Functional-family checks: one composite method, one global hybrid, and
      one range-separated hybrid. Candidate methods are B97-3C or R2SCAN-3C,
      PBE0 or B3LYP, and WB97X-D4.
- [x] Closed-shell post-HF check: run DLPNO-CCSD(T) with an explicit AuxC
      basis and preserve the coupled-cluster and triples energy components.
- [x] Closed-shell RIJCOSX check: request the approximation and def2/J
      explicitly, then confirm effective activation from the output.
- [x] Open-shell check: use triplet O2 and record the selected
      wavefunction, multiplicity, spin expectation, and spin populations.
- [x] Transition-metal baseline: idealized square-planar `[CuCl4]2-` tests
      doublet SCF convergence, spin evidence, def2-TZVP coverage, controlled
      SCF failure, and explicit MOREAD recovery from partial orbitals.
- [x] F-block check: use linear `UO2^2+`. Test one
      bounded method first, then an explicitly matched scalar-relativistic
      Hamiltonian and light-atom/heavy-atom basis combination.
- [x] Larger-system check: exercise output volume and runtime behavior without
      using it as the first parser fixture.
- [x] Ground-state multireference check: run CASSCF followed by SC-NEVPT2 and
      preserve the active-space dimensions, reference energy, correction, and
      correlated total energy.
- [x] MRCI excited-state check: preserve state symmetry, multiplicity,
      excitation energy, absolute energy, and reference weight.
- [x] Single-reference excited-state checks: run TD-DFT and EOM-CCSD, retain
      root-resolved energies, and keep the EOM singles character.
- [x] Multireference excited-state check: run state-averaged CASSCF followed by
      FIC-CASPT2 and retain per-root convergence, reference weights, energy
      denominators, and active-orbital warnings.
- [x] ORCA_ESD check: run harmonic vibronic absorption, fluorescence,
      phosphorescence, and resonance Raman examples; retain the spectrum
      artifact, rate constants, FC/HT decomposition, and all three triplet SOC
      sublevels.

The transition-metal and f-block systems and methods require scientific
selection before inputs are drafted. Representative functional families are
more useful here than a catalog of many nearly equivalent functionals.

## Chemtools work

- [x] Pin nineteen observed ORCA cases as exploratory or regression-failure
      external references; scientific promotion remains a separate review step.
- [x] Add program detection from observed ORCA 6.1.1 output markers.
- [x] Add the smallest parser slice needed for termination, version, energies,
      SCF status, explicit error termination, MOREAD restart evidence, geometry
      optimization, and frequencies.
- [ ] Add input review only for rules supported by the reviewed cases and ORCA
      documentation.
- [ ] Add serial launch preparation through the existing execution service.
- [ ] Add parallel launch preparation only after the compatible OpenMPI path
      has been tested.
- [x] Expose ORCA through the shared guided tools without adding a second
      execution
      or configuration mechanism.
- [x] Add focused program and application tests, then run the full suite.

Exit criterion: Chemtools can identify, inspect, review, and prepare the
validated ORCA workflows while unsupported methods remain explicit.

## Recorded runs

### H2 serial smoke, 2026-08-08

- ORCA: 6.1.1 release, no-DMRG x86-64 build.
- Executable: `/home/charlie/apps/orca/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca`.
- Executable SHA-256: `38b5f057452fef275c0a1b98d270ad03d0411dab70453820d1c678bd732f6c83`.
- Corpus case: `/home/charlie/input_examples/orca/00_h2_serial_smoke`.
- Input SHA-256: `802d7216c242617df23ee15d08379bd51e6c4fed5b44df280893078353345342`.
- Output SHA-256: `d70dcdb80a18ba5c3119fa4e61c796534b6b12dc1083fdd195d112f4f60af1a4`.
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty).
- Exit code: 0.
- SCF: converged in 7 cycles.
- Final single-point energy: `-1.128893619388 E_h`.
- Termination: `ORCA TERMINATED NORMALLY`.
- ORCA-reported runtime: 0.221 seconds.
- Generated artifacts: `.gbw`, `.bibtex`, `.densities`,
  `.densitiesinfo`, and `.property.txt`.

The blank warning section is not classified as a warning. This case validates
serial installation, input parsing, SCF completion, primary energy extraction,
termination detection, and artifact discovery. It does not validate DFT,
geometry optimization, frequencies, open-shell behavior, relativistic methods,
or MPI execution.

### Water R2SCAN-3C optimization and frequency, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/01_water_r2scan3c_opt_freq`.
- Input SHA-256: `9237c4282a1dbaa0b4719925ed753f7bd1cdb580aca8f363e9080f5ce2c6ceef`.
- Output SHA-256: `52005367134dffa51f73aca977cb6841b2de1d7f1ae32f4c262d19af78bcd58a`.
- Exit code: 0; stderr is empty.
- Optimization: converged after 4 cycles.
- Optimized O-H distances: 0.9618 angstrom; H-O-H angle: 104.05 degrees.
- Stationary-point energy: `-76.418938720848 E_h`.
- Vibrations: 3 molecular modes at 1653.27, 3813.56, and 3932.69
  inverse centimeters; no imaginary molecular mode.
- Thermochemistry section: present at 298.15 K and 1 atm.
- Termination: normal; ORCA-reported runtime: 3.399 seconds.

The output includes informational warnings that r2SCAN uses the libXC variant
and that geometry optimization disables AutoStart. Neither warning indicates
calculation failure. This case also establishes that an optimization output
contains several intermediate `FINAL SINGLE POINT ENERGY` records before the
stationary-point evaluation.

### Triplet O2 PBE0/DEF2-SVP single point, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/02_o2_triplet_pbe0`.
- Input SHA-256: `03e388bee1ae677d8343d3049f534fd14564f6b04ce2640d7207e48e91234035`.
- Output SHA-256: `c8f196b97dbe7c549aa616ec5fc1a0d5fb7d7c9118b79e1700aea7914c41b7f1`.
- Exit code: 0; stderr is empty.
- SCF: converged after 9 cycles.
- Final energy: `-150.051687658399 E_h`.
- Wavefunction behavior: ORCA warns that the initial RKS choice is
  incompatible with the open-shell multiplicity and switches to UKS.
- Spin expectation: 2.007223 versus ideal triplet value 2.000000.
- Mulliken spin populations: 1.000000 on each oxygen, sum 2.000000.
- Termination: normal; ORCA-reported runtime: 0.591 seconds.

The DFT spin-contamination paragraph says the reported Hartree-Fock-style
spin expectation has limited theoretical relevance. Chemtools should preserve
that context instead of turning the numerical deviation into an automatic
quality verdict.

### Formaldehyde WB97X-D4/DEF2-SVP single point, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/03_formaldehyde_wb97x_d4`.
- Input SHA-256: `b1c08309d97362367ffa5451f10991d5e994834bac6abe833174b2c2d989a385`.
- Output SHA-256: `eddd7271759dac3aa284e63c29d587aff7796e2f15a1d0813f74fc252ad51018`.
- Exit code: 0; stderr is empty.
- SCF: converged after 10 cycles.
- Final energy: `-114.442230839406 E_h`.
- Wavefunction: RHF/RKS path, charge 0, multiplicity 1.
- Termination: normal; ORCA-reported runtime: 1.278 seconds.

The output contains ORCA's informational automatic Def2/J assignment warning.
Together with the water R2SCAN-3C and O2 PBE0 runs, this completes the initial
composite, global-hybrid, and range-separated-hybrid parser slice.

### `[CuCl4]2-` doublet PBE0/DEF2-TZVP single point, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/04_cucl4_doublet_pbe0`.
- Input SHA-256: `715bb0cd975b3c2a4958624d2c4378c262fd003162e6028dad4b5619292036d1`.
- Output SHA-256: `5d0a12ef66938e80e03fcc123dac37cf2fb3be49938e6b279793178ed350dba2`.
- Exit code: 0; stderr is empty.
- Model: idealized square-planar geometry with 2.25 angstrom Cu-Cl distances,
  charge -2, and multiplicity 2.
- SCF: ORCA selected UHF/UKS and converged after 15 cycles.
- Final energy: `-3480.711549601409 E_h`.
- Spin expectation: 0.753458 versus ideal doublet value 0.750000.
- Mulliken spin sum: 1.000000, with 0.522643 on Cu and 0.119339 on
  each chloride.
- Termination: normal; ORCA-reported runtime: 27.718 seconds.

This establishes the first transition-metal parser case. Its geometry and
method remain exploratory, and it does not supply restart or SCF-recovery
evidence because the baseline converged without intervention.

### `UO2^2+` singlet PBE0/ZORA single point, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/05_uranyl_singlet_pbe0_zora`.
- Input SHA-256: `969e844dbcef5c64ac5e7e57541d727ed3f690156b1785f2eeadf87f6d7497fc`.
- Output SHA-256: `c8ddb1d164335a4d75ae48b014fce1b21ab5fa0deb58be6899ee9c729a72fefc`.
- Exit code: 0; stderr is empty.
- Model: idealized linear geometry with 1.76 angstrom U-O distances, charge
  +2, and multiplicity 1.
- Relativity: all-electron ZORA(MP), with ZORA-def2-TZVP on oxygen,
  SARC-ZORA-TZVP on uranium, and SARC/J for Coulomb fitting.
- Size: 106 electrons and 223 orbital basis functions.
- SCF: RHF/RKS, RIJCOSX, converged after 23 cycles.
- Final energy: `-29564.019120058209 E_h`.
- Termination: normal; ORCA-reported runtime: 68.812 seconds.

The output confirms both intended orbital basis families and contains no ECP.
Its picture-change warnings are preserved as calculation context. This case
validates basis composition and scalar-relativistic markers, not the scientific
adequacy of the chosen geometry or method for production uranyl work.

### Controlled `[CuCl4]2-` interruption and MOREAD restart, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/06_cucl4_scf_interrupted`.
- Interrupted input SHA-256:
  `16d1ba4eeb528bfb00e34925e3995290650dc28376876aba328d5dd5a4ea4081`.
- Interrupted output SHA-256:
  `38bb269de34c1f38665fc65d4d81c94bc0a132960bc110bc4a0c308319a06208`.
- Partial GBW SHA-256:
  `11601d7c9dee41fea230712230bfbd237bdcd5d37cc0fe06fcef79e47802bbff`.
- The `%scf MaxIter 3` run printed `SCF NOT CONVERGED`, followed by error
  termination in LEANSCF.
- The ORCA driver returned code 0 despite the explicit calculation failure.
- Restart input SHA-256:
  `020b6e112e336fd4c074f79c1b882ec58368e366fe6e175b9970a574bfd153df`.
- Restart output SHA-256:
  `9ad9b83708b23bbe77deeddc4e32e03d12b350045ea196796e931fb7f12a6a6b`.
- The differently named restart read `cucl4_interrupted.gbw` with MOREAD,
  converged after 12 cycles, and terminated normally at
  `-3480.711547683096 E_h`.

This case establishes that ORCA status must be read from output content. A
zero process return code does not prove SCF success. It also supplies a pinned
partial wavefunction and a successful explicit restart pair.

### Sextet Fe macrocycle difficult-SCF comparison, 2026-08-08

- DIIS/SOSCF corpus case:
  `/home/charlie/input_examples/orca/07_fe_macrocycle_sextet_diis`.
- Manual-TRAH corpus case:
  `/home/charlie/input_examples/orca/08_fe_macrocycle_sextet_trah`.
- Staged-recovery corpus case:
  `/home/charlie/input_examples/orca/09_fe_macrocycle_staged_recovery`.
- The geometry, charge +1, multiplicity 6, B3LYP, and def2-SVP model were
  ported from `nwchem/train/failed/hexaaquairon.nw`, which had failed to
  converge in NWChem.
- ORCA size: 155 electrons and 353 orbital basis functions.
- DIIS/SOSCF: converged after 58 cycles in 464.101 seconds at
  `-2092.531209401490 E_h`; `<S^2> = 8.992045` versus 8.75 ideal.
- Its path included SOSCF activation, seven clipped steps, and two approximate
  Hessian resets.
- Fresh-guess TRAH: converged after 89 counted cycles in 529.211 seconds at
  `-2092.517247358560 E_h`; `<S^2> = 10.001358`.
- The TRAH solution is `0.013962042930 E_h` (8.7613 kcal/mol) above the
  DIIS/SOSCF solution and has greater spin contamination.

TRAH resolved its starting negative gaps and entered the Newton regime, but it
did not recover the lower-energy DIIS/SOSCF solution. Recovery guidance must
therefore treat the SCF basin, orbital character, and spin evidence separately
from the binary fact of convergence.

The ORCA manual's three-stage starting-orbital procedure was then tested on
the same model:

- Stage-one input SHA-256:
  `aa37051cf2bfc5e9bc94f05c39def06b052605fabcbd38cacdd7cbed750a0ae1`;
  output SHA-256:
  `76162b27ce320c7555ec3a9eb84973eb43560883420404cbe7dad6a99be5b326`.
- Stage-two input SHA-256:
  `bc1e1be8818fa141a5b248f34f283c81ad04c7cceb86db4deb40c29126c41dc1`;
  output SHA-256:
  `874d1347d3a73b930098e78f55cbc695e8921572f921d5ed17876a8efb7ff7d2`.
- Final-stage input SHA-256:
  `caf2c083687fed42e9579ff327f9f0832c8bbf89acd6f5b4b6fbfbf616d25168`;
  output SHA-256:
  `e3c61fd14ce96a8065a0d883f093664c9d691446992174f7cd25bb63efabb2ce`.
- BP86/SV with RI, `LooseSCF`, and `VerySlowConv` converged after 91 counted
  cycles in 88.425 seconds at `-2093.081865106228 E_h`; AutoTRAH intervened
  after the damped iterations stalled. The run used 211 basis functions and
  gave `<S^2> = 8.901737`.
- The resulting orbitals were read with MOREAD and projected using
  `GuessMode CMatrix` into the 353-function def2-SVP basis. BP86/def2-SVP
  converged after 68 counted cycles in 120.743 seconds at
  `-2093.361414317764 E_h`; AutoTRAH intervened at iteration 50.
- B3LYP/def2-SVP read the converged target-basis BP86 orbitals and converged
  after 21 cycles in 163.563 seconds at `-2092.531206645561 E_h`, with
  `<S^2> = 8.992117`.

The staged B3LYP result is only `0.000002755929 E_h` (0.00173 kcal/mol) above
the fresh DIIS/SOSCF result and has the same spin character at the precision
reported here. The final stage used 21 rather than 58 cycles. All three stages
together took 372.731 seconds, about 19.7 percent less than the 464.101-second
fresh B3LYP run on this machine. These timings are experimental observations,
not a general performance guarantee.

This case supports staged orbitals as a recovery option, but it also shows that
the preparatory pure-GGA calculations may themselves need AutoTRAH. A recovery
recipe should preserve the intermediate method, basis, projection mode, and
final-state comparison instead of presenting MOREAD as a single opaque retry.

### Water DLPNO-CCSD(T)/cc-pVTZ TightPNO, 2026-08-08

- Corpus case:
  `/home/charlie/input_examples/orca/10_water_dlpno_ccsdt_tightpno`.
- Input SHA-256:
  `462781692c94ee6d402cd780f4a2c4934bb9c47d29f64635ef7a5b5eb23287c5`.
- Output SHA-256:
  `0c5c3cf7a90739d554dcd08570ad11fe7497c4850d9f36debc4278da469a86bb`.
- Geometry: the final water coordinates from the earlier R2SCAN-3C
  optimization case.
- Method: ORCA input keyword `DLPNO-CCSD(T)`, cc-pVTZ orbital basis,
  cc-pVTZ/C correlation-fitting basis, `TightPNO`, and `TightSCF`.
- Size: 10 electrons and 64 orbital basis functions.
- Reference SCF: RHF, converged after 11 cycles.
- Coupled-cluster iterations: converged after eight updates beyond iteration
  zero.
- T1 diagnostic: 0.006583174. Chemtools records the value without imposing an
  automatic scientific threshold.
- CCSD energy: `-76.324684273 E_h`.
- Semi-canonical triples correction: `-0.007429935 E_h`.
- Printed `E(CCSD(T))`: `-76.332114208 E_h`; final full-precision energy:
  `-76.332114208318 E_h`.
- Termination: normal; ORCA-reported runtime: 5.276 seconds.

ORCA 6.1 uses `DLPNO-CCSD(T)` for its semi-canonical triples approximation;
the manual notes that papers may call the same approximation T0. The iterative
triples input keyword is `DLPNO-CCSD(T1)`, which this case does not test.

### Water PBE0/def2-TZVP RIJCOSX, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/11_water_pbe0_rijcosx`.
- Input SHA-256:
  `4db836c00efe9f541f0e7b188b7c5d22502e6c860e395822fa802cfea32f5e89`.
- Output SHA-256:
  `c7a39472c9a5ba1840c8fe96ef456f332b78040b2262752f3cce19a8a3950fc6`.
- Geometry: the final water coordinates from the earlier R2SCAN-3C
  optimization case.
- Method: PBE0/def2-TZVP with explicit `RIJCOSX`, def2/J, `TightSCF`, and
  `NoAutoStart`.
- Size: 10 electrons and 43 orbital basis functions.
- Output confirmation: the RI Coulomb term is on and RIJ-COSX is on.
- SCF: RHF/RKS, converged after 9 cycles.
- The final GridX recomputation changed the exchange energy by
  `0.000035990 E_h` before the final energy evaluation.
- Final energy: `-76.377445212252 E_h`.
- Termination: normal; ORCA-reported runtime: 0.740 seconds.

This is an explicit-input and parser example. Water is too small to make a
meaningful RIJCOSX performance claim, and the run is not a timing comparison
against conventional exchange or RI-JK.

### Water pentamer additive QM/MM, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/12_water_pentamer_qmmm`.
- Input SHA-256:
  `86aec0a15686ed5beeb9d643ef9c6a63ecd61910a9e41c7d6612fd78b16eec66`.
- Output SHA-256:
  `e5ab7dc2bcd6c8047e015a10bbc3c4bc98ed4a58f7eb9407cab62d560d77bc83`.
- Model: additive QM/MM with electrostatic embedding, a three-atom central
  water at PBEh-3c, and twelve MM atoms in four complete neighboring waters.
- Boundary: zero link atoms. No covalent bond crosses the QM/MM partition.
- MM contribution: `0.019670168498 E_h`.
- Combined final energy: `-76.200270108260 E_h`.
- SCF: converged after 11 cycles.
- Termination: normal; ORCA-reported runtime: 0.478 seconds.

The checked-in `ORCAFF.prms` uses an illustrative fixed-charge water model.
The case validates ORCA's additive coupling and Chemtools' combined-energy
selection. It is not a solvent-model recommendation.

### NaCl6 ionic Crystal-QMMM, 2026-08-08

- Corpus case:
  `/home/charlie/input_examples/orca/13_nacl_ionic_crystal_qmmm`.
- Input SHA-256:
  `b4e23e684dfffad35af86191746c7fb4e741679886d7f6c7c0bf949ba23753b0`.
- Output SHA-256:
  `74a999f907249f91689d487105e1996be6140cfd5999c46548681b1214c52b85`.
- Preparation: `orca_crystalprep` expanded a 10x10x10 NaCl supercell. Its
  child `orca_mm` process required the ORCA directory on `PATH`.
- QM core: one Na and its six nearest Cl neighbors, charge -5 and singlet,
  matching the composition of the ORCA manual's minimal NaCl6 example.
- Boundary and embedding: 18 SDD cECP centers and 2,620 MM point charges in a
  2,645-site model.
- Method: PBE0/def2-SVP with def2/J, RIJCOSX, `TightSCF`, and `SlowConv`.
- Charge convergence: maximum charge change decreased through
  `0.630, 0.309, 0.141, 0.067, 0.031, 0.015, 0.007`; threshold `0.010`.
- Final combined energy: `-2926.983813378450 E_h`.
- Termination: normal; ORCA-reported runtime: 129.127 seconds.
- Restart artifact: ORCA wrote
  `nacl_ionic_crystal_qmmm.convCharges.ORCAFF.prms` with the converged charges.

The compact layer-selected NaCl core is retained only in scratch evidence. It
had a 6 Na/13 Cl imbalance and developed a negative HOMO-LUMO gap under PBE.
It is not presented as a runnable example.

### Alpha-glycine molecular Crystal-QMMM, 2026-08-08

- Corpus case:
  `/home/charlie/input_examples/orca/14_alpha_glycine_mol_crystal_qmmm`.
- Input SHA-256:
  `89966de683fcee55e21764e3c0e564d93bc3e41b49a80b975efe677ebf017617`.
- Output SHA-256:
  `94118f9cf5cbd6b82354ecfcf689fc81b3e4ba74a022bb23942cf55113534a93`.
- xTB: official Linux x86-64 xTB 6.7.1 release, installed separately and
  linked into the ORCA 6.1.1 directory as `otool_xtb`.
- Preparation: CrystalPrep expanded COD 2310002 to a 5x5x5 embedding. A
  reduced checked-in CIF generates coordinates identical to the full CIF.
- CrystalPrep corrections: the generated template left `NUnitCellAtoms` and
  `QMAtoms` as placeholders, and the generated XYZ lacked its comment line.
  The runnable input uses 10 unit-cell atoms, QM atoms 0 through 9, and a
  corrected XYZ file.
- Model: one neutral alpha-glycine molecule at PBEh-3c embedded in 1,140 MM
  point charges, 1,150 atoms total.
- Charge convergence: maximum charge change decreased through
  `0.685, 0.114, 0.012, 0.002`; threshold `0.010`.
- MM contribution: `0.347913570073 E_h`.
- Combined final energy: `-283.517417817239 E_h`.
- Termination: normal; ORCA-reported runtime: 24.703 seconds.
- Restart artifact: ORCA wrote
  `alpha_glycine_mol_crystal_qmmm.convCharges.ORCAFF.prms`.

This case is a working parser and execution reference. Its finite embedding,
one-molecule QM region, and PBEh-3c choice remain exploratory pending a
scientific review.

### Stretched N2 CASSCF(6,6)/SC-NEVPT2, 2026-08-08

- Corpus case:
  `/home/charlie/input_examples/orca/15_n2_stretched_casscf_nevpt2`.
- Input SHA-256:
  `cb9ce1fffc39be054408cb2211f255f14b97c4cd9512faddaa748612ccfb8572`.
- Output SHA-256:
  `714b566e2182249755fb2161e160250e24a631e3e3174c30eb2044dd1b4dbf2c`.
- Model: neutral singlet N2 at 1.8 angstrom with six active electrons in six
  active orbitals and no frozen core.
- Final CASSCF reference energy: `-108.707127699817 E_h`.
- SC-NEVPT2 correction: `-0.16407720165333 E_h`.
- SC-NEVPT2 total energy: `-108.87120490147032 E_h`.
- Termination: normal; ORCA-reported runtime: 0.724 seconds.

This is a compact bond-stretching and parser example. The basis and one-point
geometry do not establish a production dissociation treatment.

### Formaldehyde CAS(2,2)-MRCI excited states, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/16_formaldehyde_mrci`.
- Input SHA-256:
  `4bd6901ffa6b9215a6fbcbda97a7bc51ce9ec288dc37ed17d5cdbc98119d5094`.
- Output SHA-256:
  `c366eac67241cb7e1ac1f6d3a0ae68a9da0fd2d5c3a9060fdca8075bbb1930ba`.
- Setup: the ORCA MRCI tutorial's three CAS(2,2)-based reference blocks for
  the ground `1 A1`, excited `1 A2`, and `3 A2` states, with Davidson1 and
  the FullMP2 unselected-space estimate.
- Ground-state energy: `-114.113096218 E_h`; reference weight: 0.9124.
- Triplet `3 A2`: 3.649 eV; reference weight: 0.9002.
- Singlet `1 A2`: 4.041 eV; reference weight: 0.8883.
- Termination: normal; ORCA-reported runtime: 2.358 seconds.

This reproduces a syntax and output tutorial. The compact basis, selection
thresholds, and active space still require method-specific review before use
as a quantitative benchmark.

### Formaldehyde PBE0 TD-DFT, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/17_formaldehyde_pbe0_tddft`.
- Input SHA-256:
  `076b687b664a00f5ee4191c487a95b80f558dce8ffb6271d14a2d61d50d85bff`.
- Output SHA-256:
  `100b417bf07aa25461ad2f8139e6d722e51dd74089aad58e0fc708d4aadeae27`.
- Setup: PBE0/def2-TZVP with def2/J and RIJCOSX, six singlet roots, six
  triplet roots, and natural transition orbital generation.
- Lowest singlet: 3.927 eV, dominated by `7a -> 8a` with printed weight
  0.995631.
- Lowest triplet: 3.185 eV, dominated by `7a -> 8a` with printed weight
  0.993096.
- Termination: normal; ORCA-reported runtime: 3.350 seconds.

The case checks spin-resolved TD-DFT root parsing and NTO generation. It is a
vertical-excitation example, not a statement about the best functional for
formaldehyde spectroscopy.

### Formaldehyde EOM-CCSD, 2026-08-08

- Corpus case: `/home/charlie/input_examples/orca/18_formaldehyde_eom_ccsd`.
- Input SHA-256:
  `9ba5465c5380fcae52677e39cfb2532472f56b82ed9c617621211ef420b41491`.
- Output SHA-256:
  `69ce366c1ef726cd2586b7303b3942b148349b5671f5c51726d1c8e839a00541`.
- Setup: the ORCA EOM tutorial's RHF EOM-CCSD/cc-pVDZ model with four roots.
- CCSD ground-state energy: `-114.208715775 E_h`.
- Excitation energies: 4.023, 8.548, 9.356, and 9.910 eV.
- Singles character: 92.30, 90.42, 92.05, and 87.22 percent.
- Termination: normal; ORCA-reported runtime: 2.596 seconds.

ORCA repeats the roots in an approximate left-state section. Chemtools keeps
the right-state result table once, preventing duplicated roots. The printed
final single-point value is the first excited-state absolute energy, so the
parser also exposes the CCSD ground-state energy separately.

### N2 state-averaged CASSCF(6,6)/FIC-CASPT2, 2026-08-08

- Corpus case:
  `/home/charlie/input_examples/orca/19_n2_excited_casscf_caspt2`.
- Input SHA-256:
  `473efba26d60b0856221248e80e2b99a3d01d3210755a134bf2a7a8f4e0d545b`.
- Output SHA-256:
  `5ca73eb8699affbed0f11be0c80d6bdf9928efd0c12023cee8cab37b71b65d30`.
- Setup: neutral singlet N2 at 1.10 angstrom, three equally weighted singlet
  roots in CASSCF(6,6), followed by zero-shift FIC-CASPT2.
- CASSCF excitation energies: 10.569 and 11.358 eV.
- CASPT2 excitation energies: 10.008 and 10.519 eV.
- CASPT2 convergence: 9, 10, and 10 iterations for roots 0, 1, and 2.
- Reference weights: 0.9527201710, 0.9426592446, and 0.9341686828.
- Minimum printed denominators: 0.803090021, 0.184577601, and 0.188948668
  `E_h`.
- Warnings: five attempts failed to constrain the active orbitals during
  orbital fitting, and the CASSCF path also discarded several badly
  conditioned DIIS vectors.
- Termination: all three CASPT2 roots converged and ORCA terminated normally;
  ORCA-reported runtime: 1.208 seconds.

The warning history remains part of the parsed diagnostics. The reference
weights and denominators do not show an obvious intruder-state failure in this
run, but the active-orbital tracking warnings require orbital inspection before
the state assignments are treated as scientifically settled.

### Formaldehyde ORCA_ESD spectra and radiative rates, 2026-08-08

- Corpus case:
  `/home/charlie/input_examples/orca/20_formaldehyde_esd`.
- Model: PBE0/def2-SVP with def2/J and RIJCOSX. Absorption, fluorescence, and
  resonance Raman use the vertical-gradient model with Herzberg-Teller
  derivatives.
- Ground state: optimized energy `-114.284008969306 E_h`; six positive
  molecular frequencies.
- Triplet recovery: the exactly planar T1 optimization found a saddle with a
  `-673.64 cm-1` out-of-plane mode. An out-of-plane displaced restart reached
  a nonplanar T1 minimum at `-114.177681338356 E_h`, with frequencies 769.77,
  844.21, 1247.62, 1453.58, 2947.98, and 3062.84 cm-1.
- Absorption: 0-0 energy `30844.79 cm-1`; maximum at 312.07 nm.
- Fluorescence: 10 cm-1 linewidth; maximum at 337.34 nm; rate
  `1.830239e5 s-1`, equivalent to a 5.46 microsecond radiative lifetime.
- Resonance Raman: default laser energy `30844.79 cm-1`; strongest computed
  band at `2399.90 cm-1`. ORCA 6.1.1 accepts `RRINTENS`; it rejects the
  `RRINTES` spelling shown in the manual's short narrative example.
- Phosphorescence: adiabatic gap `23336.22 cm-1` without ZPE; 0-0 energy
  `22615.33 cm-1`. The three SOC-sublevel rates are 1.247343, 0.5652320, and
  48.10585 s-1. Their sum is 49.91843 s-1, equivalent to a 20.03 ms radiative
  lifetime. The dominant spectrum is from root 3 near 442.18 nm.
- Every spectrum and emission-rate run printed
  `ORCA ESD FINISHED WITHOUT ERROR`, followed by normal ORCA termination.

The rates are radiative-only values from a harmonic gas-phase model. They omit
internal conversion, intersystem crossing, solvent, and other nonradiative
channels. The FC and HT percentages are retained exactly as ORCA prints them;
destructive interference can make an individual component negative or larger
than 100 percent.

## Implementation evidence, 2026-08-08

- Backend registry: ORCA is the seventh built-in backend and is selected by
  content from 31 real outputs across nineteen pinned cases.
- Capabilities: input parsing, output parsing and task indexing, last-geometry
  extraction, and frequency extraction. ORCA output evidence now includes
  orbital and auxiliary basis sets, basis-function and electron counts, and
  the scalar-relativistic method when present. Explicit SCF error termination
  is classified as failure even when the ORCA driver returns zero. MOREAD is
  preserved as initial-guess evidence, and input parsing preserves CMatrix or
  FMatrix basis projection when explicitly selected. DLPNO output preserves
  coupled-cluster convergence, the T1 diagnostic, correlation and CCSD
  energies, the triples correction, and the printed CCSD(T) energy. ORCA's
  output-side RIJ-COSX activation marker is preserved separately from the
  requested simple-input keyword. QM/MM parsing preserves the model, coupling
  and embedding schemes, subsystem sizes, MM point-charge and cECP counts,
  charge-convergence history, MM energy, and combined QM/MM energy.
  Multireference and excited-state parsing preserves CASSCF active spaces and
  roots, NEVPT2 and CASPT2 root energies and corrections, CASPT2 convergence,
  reference weights and denominators, MRCI states, TD-DFT singlet and triplet
  roots, and EOM-CCSD roots with singles character. ORCA_ESD parsing preserves
  process, model, linewidth, temperature, energy gaps, laser energy, spectrum
  filename, module completion, radiative rate, and FC/HT decomposition.
- Shared interface: the ORCA program filter exposes the eleven guided tools;
  real ORCA inspection and exploratory-reference search succeed from an
  installed wheel.
- Test suite: 1,891 tests passed with
  `CHEMTOOLS_REFERENCE_CORPUS=/home/charlie/input_examples`.
- Installed wheel SHA-256:
  `f15e69c1d24e559b0a472b6aa192aacd95c3db1442649772f0d6da025eec123d`.
- Installation:
  `/home/charlie/mcps/chemtoolsmcp/venv/lib64/python3.11/site-packages/chemtools`.

The ORCA-specific MCP module advertises no duplicate low-level tools. ORCA
uses the existing shared inspection and reference-case operations. Input
linting, drafting, recovery advice, serial launch preparation, and parallel
launch preparation remain unavailable until their corresponding cases and
runtime choices are validated.
