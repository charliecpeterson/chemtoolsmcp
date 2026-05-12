orphan  

# 26.0

New features:

- Basis set library has been extended with 5z basis sets for s- and p-block elements: New dyall.(v5z/cv5z/av5z/ae5z/acv5z/aae5z) basis sets.  
  - Contributor: K. Dyall.
  - Zenodo links: [s-block
    elements](https://zenodo.org/records/17088050) and [p-block
    elements](https://zenodo.org/records/18509352).
  - Reference: [M. L. Reitsma, E. H. Prinsen, J. D. Polet, A.
    Borschevsky, and K. G. Dyall, Relativistic quintuple-zeta basis sets
    for the s block Available to Purchase, J. Chem. Phys. 163, 144116
    (2025).](https://doi.org/10.1063/5.0297009)

- Merge of the DIRAC_CASPT2 program published by Masuda and coworkers into DIRAC.  
  - This adds the following features:  
    - IVO calculation (Huzinaga1970).
    - CASCI/CASPT2 calculation (Anderson1990 but the reference function
      is CASCI not CASSCF).
    - RASCI/RASPT2 calculation.

  - Contributors: K. Noda, S. Iwamuro, Yasuto Masuda and M. Abe, Tokyo
    University of Agriculture and Technology,iUniversity of Hiroshima,
    Japan.

  - Reference: [Yasuto Masuda, Kohei Noda, Sumika Iwamuro, Masahiko
    Hada, Naoki Nakatani, and Minori Abe, Relativistic CASPT2/RASPT2
    Program along with DIRAC Software, J. Chem. Theor. Comput. 21, 1249
    (2025).](https://doi.org/10.1021/acs.jctc.4c01589)

- Schiff moment operator:  
  - Contributor: K. Gaul.
  - Reference: [Konstantin Gaul, Nicholas Hutzler, Phelan Yu, Andrew
    Jayich, Miroslav Ilias and Anastasia Borschevsky, CP-violation
    sensitivity of closed-shell radium-containing polyatomic molecular
    ions, Phys. Rev. A 109, 042819
    (2024)](https://doi.org/10.1103/PhysRevA.109.042819).

- Calculation of parity-violating EFG tensors. Activated with ".PVCEFG"  
  - Contributor: Juan J. Aucar.
  - Reference: [Juan J. Aucar and Alejandro F. Maldonado, Parity
    Violation Effects on the Electric Field Gradient, Phys. Chem. Chem.
    Phys. 27, 7594 (2025)](https://doi.org/10.1039/D4CP04840G)
  - Manual: PVCEFG
    <https://diracprogram.org/doc/release-26/manual/properties.html#pvcefg>.

- ExaCorr: new functionality  
  - Support via the TAPP API to tensor libraries (e.g. the TBLIS library for single-node CPU architectures), in addition to TAL-SH (single-node CPU or GPU architectures) and ExaTENSOR (distributed memory CPU or GPU architectures). The choice of tensor library is made on compilation.  
    - Contributors: J. Brandejs, A. S. P. Gomes, L. Visscher
    - Reference: [J. Brandejs, N. Hörnblad, E. F. Valeev, A.
      Heinecke, J. Hammond, D. Matthews, P. Bientinesi, Tensor Algebra
      Processing Primitives (TAPP): Towards a Standard for Tensor
      Operations arXiv:2601.07827
      (2026)](https://arxiv.org/abs/2601.07827)
    - Manual:
      <https://diracprogram.org/doc/release-26/manual/wave_function/exacorr.html#executor>
      and
      <https://diracprogram.org/doc/release-26/installation/tensor_libraries.html>

  - Orbital-unrelaxed quadratic response coupled cluster molecular properties.  
    - Contributors: X. Yuan, L. Halbert, L. Visscher, A. S. P. Gomes
    - Reference: [X. Yuan, L. Halbert, L. Visscher, A. S. P. Gomes, A
      Comparison of Relativistic Coupled Cluster and Equation of Motion
      Coupled Cluster Quadratic Response Theory J. Phys. Chem. A 129,
      11695-11712 (2025)](https://doi.org/10.1021/acs.jpca.5c04475)
    - Manual:
      <https://diracprogram.org/doc/release-26/manual/wave_function/exacorr.html#ccciqr>.

  - Ground-state expectation values with higher-order coupled cluster (CCSDT, CCSDTQ).  
    - Contributors: G. Fabbro, J. Brandejs, T. Saue
    - Reference: [G Fabbro, J Brandejs, T Saue, Highly Accurate
      Expectation Values Using High-Order Relativistic Coupled Cluster
      Theory, J. Phys. Chem. A 129, 6942
      (2025)](https://doi.org/10.1021/acs.jpca.5c02844)
    - Manual:
      <https://diracprogram.org/doc/release-26/manual/wave_function/exacorr.html#wavefunctions-for-iterative-methods>

  - CIS and CIS(D) excitation energies.  
    - Contributors: X. Yuan, T. Lejeune, A. S. P. Gomes
    - Manual:
      <https://diracprogram.org/doc/release-26/manual/wave_function/exacorr.html#excited-state-calculation-options>

  - Virtual frozen natural orbitals for response properties.  
    - Contributors: M. Le, X. Yuan, A. S. P. Gomes
    - Manual:
      <https://diracprogram.org/doc/release-26/manual/wave_function/exacorr.html#id10>

- VISUAL: new functionality  
  - Improved data handling and more calculation options  
    - The real-space data can be exported in TXT/CSV/CUBE/HDF5 formats
    - MPI parallelization
    - Possibility to calculate various densities in the same run on the
      same or different grids
    - Improved user input: more flexibility to define grids and
      densities ("grid functions")
    - Selected tutorials are translated to jupyter notebooks (added to
      <span class="title-ref">demo_notebooks/</span>)

  - Contributor: Gosia Olejniczak

  - Manual: <https://diracprogram.org/doc/release-26/manual/visual.html>

- RELCCSD: print out largest amplitudes (for analysis) at print level 2.  
  - Contributor: Lucas Visscher.
  - Manual:
    <https://diracprogram.org/doc/release-26/manual/wave_function/relccsd.html#print>

- TREXIO-converter: Jupyter notebook to convert (non-relativistic) wave function information to the TREXIO format.  
  - Contributor: Lucas Visscher.
  - Manual: <https://diracprogram.org/doc/release-26/manual/data.html>

------------------------------------------------------------------------

# 25.0

New features:

- Effective QED potentials:  
  - Contributors: A. Sunaga and T. Saue.
  - Reference: [Ayaki Sunaga, Maen Salman and Trond Saue, 4-component
    relativistic Hamiltonian with effective QED potentials for molecular
    calculations, J. Chem. Phys. 157, 164101
    (2022)](https://doi.org/10.1063/5.0116140).
  - Vacuum polarization: Uehling.
  - Electron self-energy: Flambaum-Ginges and Pyykkö-Zhao.
  - Tutorial:
    <http://www.diracprogram.org/doc/release-25/tutorials/qed/effqed.html>.

- Atomic shift operator:  
  - Contributor: T. Saue.
  - Reference: [Ayaki Sunaga, Maen Salman and Trond Saue, 4-component
    relativistic Hamiltonian with effective QED potentials for molecular
    calculations, J. Chem. Phys. 157, 164101
    (2022)](https://doi.org/10.1063/5.0116140).
  - Manual:
    [ASHIFT](http://www.diracprogram.org/doc/release-25/manual/hamiltonian.html#ashift).

- New ESR/EPR features:  
  - Contributor: H. J. Aa. Jensen.
  - .ESR CI flexible option for defining CI for ESR/EPR calculations.
  - .G10PPM for \*ESR : request g-shifts to 10 ppm accuracy instead of
    default of +/-1 ppt = 1000 ppm accuracy.
  - Added .STATE option for \*ESR, defining first CI state belonging to
    ESR multiplet (for example if a "singlet" state is below the
    "triplet" multiplet).
  - .MAXMK2 for \*ESR.
  - .KR CONJ to generate the second state for doublets by Kramers
    conjugation.

- KRCI module:  
  - Contributor: H. J. Aa. Jensen.
  - '.CIROOTS' input has been made easier, description in manual has
    been updated.
  - The KRCI module now again works with both LUCIAREL and GASCIP (a fix
    from 2021 fixed symmetry specification for LUCIAREL but made it not
    work for GASCIP).
  - GASCIP code optimized, much faster now.

- DIRAC calculator for the Atomistic Simulation Environment (ASE) is available.  
  - Contributors: Carlos M. R. Rocha and Lucas Visscher.
  - Reference (DIRAC calculator): <https://qc2.readthedocs.io>
  - Reference (ASE): <https://wiki.fysik.dtu.dk/ase/>

- Approximations to EOM-CCSD (Partitioned EOM (P-EOM-CCSD); second-order perturbation theory-based EOM (EOM-MBPT(2) and P-EOM-MBPT(2)) are available in the RELCCSD module for excitation and ionization energies.  
  - Contributors: L. Halbert and A. Gomes.
  - Reference: [L. Halbert and A. S. P. Gomes, The performance of
    approximate equation of motion coupled cluster for valence and core
    states of heavy element systems, Mol. Phys. e2246592
    (2023)](https://doi.org/10.1080/00268976.2023.2246592).

Infrastructure changes and fixes:

- Better cmake structure for compiling DIRAC. Now utility programs are small and not the same size as dirac.x.  
  - Contributor: H. J. Aa. Jensen.

- Changed all references to ifort etc. to ifx etc from intel oneAPI; ifort is not supported from oneAPI 2025.  
  - Contributor: H. J. Aa. Jensen and J. J. Aucar.

- Made all necessary changes for Nvidia's NVHPC compilers (previously known as PGI compilers).  
  - Contributor: H. J. Aa. Jensen.

- Include support in ExaCorr for NVHPC, and added hints for compilation with OneAPI and NVHPC, and a way to control via environment variable CUDA_SM_ARCH which NVIDIA GPU architeture to compile for. Please note that calculations beyond iterative triples (e.g. CCSDTQ) are not supported with NVHPC.  
  - Contributor: A. Gomes.

- Quit if LUCIAREL is requested for sequential dirac.x (LUCIAREL only works if compiled with MPI).  
  - Contributor: H. J. Aa. Jensen.

Basis set library:

- Basis set library has been extended with "relativistic prolapse-free Gaussian basis sets" RPF-2Z, RPF-3Z, aug-RPF-2Z, and aug-RPF-3Z.  
  - Contributor: H. J. Aa. Jensen.
  - Reference: [J. S. Sousa, E. F. Gusmão, A. K. N. R. Dias,
    and R. L. A. Haiduke, Relativistic Prolapse-Free Gaussian Basis Sets
    of Double- and Triple-ζ Quality for s- and p-Block Elements:
    (aug-)RPF-2Z and (aug-)RPF-3Z. J. Chem. Theor. Comput. 20, 9991
    (2024).](https://doi.org/10.1021/acs.jctc.4c01211)

- Basis set library has been extended with ECP basis sets cc-pVnZ-PP and cc-pwCVnZ-PP, for n = D,T,Q,5, and the augmented versions aug-cc-pVnZ-PP and aug-cc-pwCVnZ-PP  
  - Contributor: Kirk Peterson.

- Transferred fixes of Dalton basis sets from Dalton:  
  - Corrected some references in correlation consistent basis sets (as
    e.g. aug-cc-pVTZ).
  - Fixed Scandium in aug-cc-pVTZ-J (two lines missing, would have given
    error if used).
  - Contributor: H. J. Aa. Jensen.

- Added special basis sets for spin-spin coupling and hyperfine coupling from Frank Jensen group (Aarhus University):  
  - pcJ, ccJ and pcH sets.
  - Copied from Dalton basis set directory.
  - Contributor: H. J. Aa. Jensen.

- Corrections to Dyall basis sets:  
  - Fixed incorrect definitions for the number of exponents for Zr/Nb/Ru
    in dyall.acv4z.

  - Fixed missing exponents:  
    - 1x f for Lu in cv2z/ae2z
    - 1x d for K in acv4z
    - 3x f, 2x g, 1x h for Rb/Sr in acv4z
    - 1x g for Ra in acv4z/aae4z
    - 1x f for Rf--Cn in aae4z

  - Fixed incorrect exponent values:  
    - 1x s, 1x p, 1x d for Zr in av3z/acv3z/aae3z
    - 2x g, 1x h for Pb--Rn in acv4z/aae4z
    - 1x g for Tl--Rn in ae4z
    - 1x g for Lr in v4z

  - Fixed 150+ instances of small rounding errors.

  - Contributor: Lukas F. Pasteka.

Bugfixes:

- Fix of symmetry problem for KRMCSCF with LUCIAREL and odd number of electrons.  
  - Contributor: H. J. Aa. Jensen.

- Changes and corrections in the definition of one-electron operators $`\boldsymbol{\beta\gamma}^5`$ and $`i \boldsymbol{\beta\alpha}`$.  
  - One-electron operator type `ZiBETAAL` was not defined properly in
    previous code versions. From now on, it is replaced by `ZBETAALP`
    and represents the operator
    $`i \boldsymbol{\beta} \boldsymbol{\alpha}_z`$.
  - In previous code versions, the one-electron operator types
    `XiBETAAL` and `YiBETAAL` represented
    $`i \boldsymbol{\beta} \boldsymbol{\alpha}_x`$ and
    $`i \boldsymbol{\beta} \boldsymbol{\alpha}_y`$, respectively. From
    now on, they are replaced by the operator types `XBETAALP` and
    `YBETAALP`, respectively.
  - New one-electron operator type `BETAGAMMA5` available, representing
    the time-symmetric operator $`\boldsymbol{\beta\gamma}^5`$. Please
    note that the operator type `iBETAGAMMA5` was already implemented in
    DIRAC and it also points to $`\boldsymbol{\beta\gamma}^5`$, but with
    the assignments of time-reversal antisymmetry.
  - For more details, please check the manual: `one_electron_operators`.
  - Contributor: I. A. Aucar.

- Set the correct sign of CC linear and quadratic response functions in the case of more than one time-antisymmetric perturbation operator  
  - Contributor: A. Gomes.

New defaults:

- The CODATA2022 set of physical constants is now used as default (taken from <http://physics.nist.gov/constants> on March 6, 2025).  
  - Contributor: I. A. Aucar.

- The inverse fine-structure constant is now extracted directly from CODATA, rather than being calculated from other fundamental constants.  
  - Contributor: I. A. Aucar.

- Planck's constant is taken from CODATA (the reduced Planck constant is now calculated internally).  
  - Contributor: I. A. Aucar.

- Tenpi-generated code is now the default choice for automatically generated T equation codes.  
  - Contributor: A. Gomes.

- Default CI gradient convergence threshold for \*KRCI is now 3.d-3 for energy calculations and 1.d-4 for property calculations. The 3.d-3 value is sufficient for total energies, accurate to ca. 0.01 mHartree, and saves a lot of unnecessary CPU time, but it will only give 2-3 digits in properties, which have linear accuracy in the CI wavefunction.  
  - Contributor: H. J. Aa. Jensen.

------------------------------------------------------------------------

# 24.0

New features:

New functionality with ExaCorr:  
- Calculation of linear response properties with CCSD wavefunctions.
- Contributors: Xiang Yuan, Loic Halbert, Johann Pototschnig, Anastasios
  Papadopoulos, Lucas Visscher and Andre Gomes.
- Reference: Xiang Yuan, Loic Halbert, Johann Valentin Pototschnig,
  Anastasios Papadopoulos, Sonia Coriani, Lucas Visscher, and Andre
  Severo Pereira Gomes, Formulation and Implementation of
  Frequency-Dependent Linear Response Properties with Relativistic
  Coupled Cluster Theory for GPU-Accelerated Computer Architectures, J.
  Chem. Theory Comput. 20, 677 (2024)
  <https://doi.org/10.1021/acs.jctc.3c00812>
- Manual: CCCILR
  <https://diracprogram.org/doc/release-24/manual/wave_function/exacorr.html#cccilr>
- Tutorial: not yet available; please have a look at examples in the
  testset.
- Calculation of quadratic response properties CCSD wavefunctions.
- Contributors: Xiang Yuan, Loic Halbert, Lucas Visscher and Andre
  Gomes.
- Reference: Xiang Yuan, Loic Halbert, Lucas Visscher, and Andre Severo
  Pereira Gomes, Frequency-Dependent Quadratic Response Properties and
  Two-Photon Absorption from Relativistic Equation-of-Motion Coupled
  Cluster Theory, J. Chem. Theory Comput. 19, 9248 (2023)
  <https://doi.org/10.1021/acs.jctc.3c01011>
- Manual: CCCIQR
  <https://diracprogram.org/doc/release-24/manual/wave_function/exacorr.html#ccciqr>
- Tutorial: not yet available; please have a look at examples in the
  testset.
- Calculation of energies with equation of motion coupled cluster for
  excitation (EOM-EE-CCSD), ionization (EOM-IP-CCSD), and electron
  affinity (EOM-EA-CCSD) models.
- Contributors: Xiang Yuan, Loic Halbert, Lucas Visscher and Andre
  Gomes.
- Reference: Xiang Yuan, PhD thesis (2024),
  <https://doi.org/10.5463/thesis.505>
- Manual: EXEOM
  <https://diracprogram.org/doc/release-24/manual/wave_function/exacorr.html#exeom>
  see also
  <https://www.diracprogram.org/doc/release-24/manual/wave_function/exacorr.html#id10>
- Tutorial: not yet available; please have a look at examples in the
  testset.
- Two-body reduced density matrix for CCSD ground-state wavefunctions.
- Contributors: Loic Halbert, Andre Gomes
- Manual: GS2RDM
  <https://diracprogram.org/doc/release-24/manual/wave_function/exacorr.html#gs2rdm>
- Tutorial: not yet available; please have a look at examples in the
  testset.

Revised features:

- Restart from density matrices of CC for first-order properties.  
  - Contributors: Trond Saue, Andre Gomes
  - Manual: RDCCDM
    <https://diracprogram.org/doc/release-24/manual/properties.html#rdccdm>

- Update of DIRAC schema which defines the information stored on CHECKPOINT.h5.  
  - Essential symmetry information added.
  - MO coefficients for C\<sub\>1\</sub\> symmetry optionally stored for
    restarts without use of symmetry.
  - Occupation of spinors in SCF added.
  - Non-default property integrals stored on separate file to reduce the
    size of the checkpoint file.
  - Original input is preserved in the CHECKPOINT file.
  - Contributors: Lucas Visscher, Trond Saue

Improvements:

- Support for AMD GPUs with ExaCorr (TAL-SH, work in progress to enable distributed calculations with ExaTensor), in addition to the NVIDIA GPU support (TAL-SH and ExaTensor).  
  - Contributors: Johann Pototschnig, Jan Brandejs and Andre Gomes.
  - For further information on ExaTensor and TAL-SH as used in DIRAC,
    see <https://github.com/RelMBdev/ExaTENSOR/tree/dirac>
  - Note : ROCM versions 5.7+ should be used. Earlier ROCM versions may
    result in code that compiles but may show runtime errors (e.g.
    HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, or values different than
    reference ones in the tests). A potential workaround for earlier
    versions is to reduce hipcc optimization level to -O1.

Bugfixes:

- Fixed orbital analysis with .VECPRI for X2C, issue \#79 (H. J. Aa.
  Jensen)
- Fixed Fock-space coupled cluster calculations with finite fields.

------------------------------------------------------------------------

# 23.0

New features:

- Calculation of parity-violating nuclear spin-rotation tensors. Activated with ".PVCSR"  
  - Contributor: I. Agustín Aucar.
  - Reference: [Ignacio Agustín Aucar and Anastasia Borschevsky,
    Relativistic study of parity-violating nuclear spin-rotation
    tensors. J. Chem. Phys. 155, 134307
    (2021)](https://doi.org/10.1063/5.0065487)
  - Manual:
    [PVCSR](http://www.diracprogram.org/doc/release-23/manual/properties.html#pvcsr)
  - Tutorial:
    <http://www.diracprogram.org/doc/release-23/tutorials/pvcsr/tutorial.html>

- Calculation of oscillator strenghts beyond the electric dipole approximation in the generalized velocity representation  
  - Contributors: Martin van Horn, Trond Saue, and Nanna Holmgaard List.
  - Reference: [M. van Horn, T. Saue, N. H. List. Probing Chirality
    across the Electromagnetic Spectrum with the Full Semi-Classical
    Light-Matter Interaction. J. Chem. Phys. 156, 054113
    (2022)](https://doi.org/10.1063/5.0077502)
  - Manual:
    [VPOL](http://www.diracprogram.org/doc/release-23/manual/properties/excitation.html#vpole)
  - Tutorial:
    <http://www.diracprogram.org/doc/release-23/tutorials/x_ray/BED_Ba/tutorial.html>

- (extended) atomic-mean-field two-electron scalar and spin-orbit picture-change corrections for two-component Hamiltonian calculations  
  - Contributors: Stefan Knecht, Michal Repsiky, Hans Joergen Aa. Jensen
    and Trond Saue.
  - Reference: [Stefan Knecht, Michal Repisky, Hans Jørgen Aagaard
    Jensen, and Trond Saue, Exact two-component Hamiltonians for
    relativistic quantum chemistry: Two-electron picture-change
    corrections made simple, J. Chem. Phys. 157, 114106
    (2022)](https://doi.org/10.1063/5.0095112)
  - Manual:
    [XAMFI](http://www.diracprogram.org/doc/release-23/manual/xamfi.html)
  - Tutorial:
    <http://www.diracprogram.org/doc/release-23/tutorials/two_component_hamiltonians/xamfX2C.html>

- ExaCorr module works now also with contracted basis functions.  
  - Contributor: Lucas Visscher.

- Basis set library has been extended with basis sets with diffuse functions for s and d blocks, dyall.aXNz basis sets.  
  - Contributor: Ken Dyall.
  - Reference: [K. G. Dyall, P. Tecmer, and A. Sunaga, Diffuse basis
    functions for relativistic s and d block Gaussian basis sets, J.
    Chem. Theor. Comput. 19, 198
    (2023).](https://doi.org/10.1021/acs.jctc.2c01050)

Infrastructure changes and fixes:

- HDF5 checkpoint file is now default for data handling  
  - Contributor: Lucas Visscher
  - User manual: [Data storage in
    DIRAC](http://www.diracprogram.org/doc/release-23/manual/data.html)
  - Developer manual: [The CHECKPOINT.h5
    file](http://www.diracprogram.org/doc/release-23/development/checkpoint.html)

Revised features:

- Added a CI gradient convergence threshold option '.THRGCI' to '\*KRCICALC' (could not be changed by user previously).  
  - Contributor: H. J. Aa. Jensen.

Performance improvements:

- Improved code in luciarel CI program. Note that KRMC and KRCI by default uses luciarel.  
  - Contributor: Andreas Nyvang.
  - numerical stability has been improved in the complex parallel
    diagonalization routine
  - convergence algorithm has also been changed, with considerable
    improvement in convergence rate. In particular, the truncation of
    the reduced space in the Davidson iterations is now done in a much
    better way, leading to the improvement in convergence rate. Both for
    real and complex parallel cases.

New defaults:

- Default of MOLTRA changed from scheme 6 to scheme 4 (A. Sunaga).

Bugfix:

- Default for .MVO has been corrected, so it now behaves as expected:
  modified virtual orbitals based on the Fock potential from the
  doubly-occupied molecular orbitals after an open-shell SCF calculation
  (H. J. Aa. Jensen).

------------------------------------------------------------------------

# 22.0 (2022-02-08)

New features:

- New licence: GNU Lesser General Public License v2.1

- Electronic circular dichroism (ECD) beyond lowest-order -- full light-matter interaction  
  - Contributors: Martin van Horn, Trond Saue, and Nanna Holmgaard List
  - Reference: [M. van Horn, T. Saue, N. H. List., Probing Chirality
    across the Electromagnetic Spectrum with the Full Semiclassical
    Light-Matter Interaction. J. Chem. Phys. (in press)
    (2022)](https://doi.org/10.1063/5.0077502),
    [ChemRxiv](https://doi.org/10.33774/chemrxiv-2021-j1hcz-v2)

- MP2 frozen virtual natural orbitals via the ExaCorr module  
  - Contributors: Xiang Yuan, Lucas Visscher, André Severo Pereira
    Gomes.
  - Reference: [X. Yuan, L. Visscher, A. S. P. Gomes., Assessing MP2
    frozen natural orbitals in relativistic correlated electronic
    structure calculations.](https://arxiv.org/abs/2202.01146)
  - Manual:
    [.MP2NO](http://www.diracprogram.org/doc/release-22/manual/wave_function/exacorr.html#mp2no)
    and
    [.DONATORB](http://www.diracprogram.org/doc/release-22/manual/wave_function/exacorr.html#donatorb)

- Polarizable Embedding with Complex Polarization Propagator  
  - Contributors: Joel Creutzberg, Erik D. Hedegård
  - Reference: [Polarizable embedding complex polarization propagator in
    four- and two-component
    frameworks.](https://arxiv.org/abs/2112.07721)

Infrastructure changes and fixes:

- HDF5 checkpoint file and DIRAC data scheme, python utilities to extract data  
  - Contributor: Lucas Visscher
  - Manual: [The CHECKPOINT.h5
    file](http://www.diracprogram.org/doc/release-22/development/checkpoint.html)

Improvements:

- DIIS replaced by [CROP](https://pubs.acs.org/doi/10.1021/ct501114q) algorithm in ExaCorr module.  
  - Contributors: Chima Chibueze and Lucas Visscher.
  - Reference: [Pototschnig et al., J. Chem. Theory Comput. 17 (2021)
    5509−5529.](https://doi.org/10.1021/acs.jctc.1c00260)
  - Manual:
    [EXACC](http://www.diracprogram.org/doc/release-22/manual/wave_function/exacorr.html)

Bugfix:

- There was an error in the construction of the projection operator for
  secondary orbitals using density matrices for the very special case of
  4-component relatistic average-of-configuration HF calculations on
  systems with inversion symmetry and no occupation in ungerade
  symmetry.

------------------------------------------------------------------------

# 21.1 (2021-09-16)

Revised features:

- Enhanced pam script: now it also looks in job directory for .diracrc,
  and it looks there first. pam looks for .diracrc three places: 1) job
  directory ("local .diracrc"), 2) home directory ("global
  .diracrc"), 3) pam script directory ("system .diracrc). The first
  .diracrc found is used. (H. J. Aa. Jensen).
  - added ENABLE_EXATENSOR to the options you can change with 'ccmake
    --' (H. J. Aa. Jensen).
  - enable talsh only with unsupported mpi libraries (Andre Gomes)
  - improved exatensor MPI identification (Johann Pototschnig)

Error corrections:

- Fixed pam script so it can find dirac.x and basis set directories in
  an installed version (i.e. after make install or e.g. "cmake --install
  build --prefix \<installation root dir\>"). H.J.Aa.Jensen.
- Fixed memory management in LUCIA (Lucas Visscher)
- atomic supersymmetry now works with X2Cmmf (Ayaki Sunaga, Stefan
  Knecht, Trond Saue)

------------------------------------------------------------------------

# 21.0 (2021-05-28)

New features:

- New parallel coupled cluster implementation ExaCorr. Contributors: J.V. Pototschnig, A. Papadopoulos,  
  D. I. Lyakh, M. Repisky, L.Halbert, A.S.P. Gomes, H.J.Aa. Jensen,
  and L. Visscher. To be published.

- Improved start potential with far fewer functions and better accuracy. Contributor: Lucas Visscher.  
  S. Lehtola, L. Visscher and E. Engel. Efficient implementation of the
  superposition of atomic potentials initial guess for electronic
  structure calculations in Gaussian basis sets.

  10. Chem. Phys. 152, 144105 (2020);
      <http://dx.doi.org/10.1063/5.0004046> doi: 10.1063/5.0004046

- Calculation of Rotational Molecular g-tensors. Activated with ".ROTG". Contributor: I. Agustín Aucar.  
  I. A. Aucar, S. S. Gómez, C. G. Giribet and M. C. Ruiz de Azúa.
  Theoretical study of the relativistic molecular rotational g-tensor.

  10. Chem. Phys. 141, 194103 (2014);
      <http://dx.doi.org/10.1063/1.4901422> doi: 10.1063/1.4901422

- Improved reading of basis sets to be able to read all EMSL basis sets
  of type DALTON. Contributor: Hans Joergen Aa. Jensen.

- "make latex" for making a latex version of the documenation in
  "build/latex", used e.g. pdflatex to make a pdf version of the manual.
  Contributor: Hans Joergen Aa. Jensen

- Efficient parallel diagonalization of quaternion Fock matrices using
  openMP. Contributor: Hans Joergen Aa. Jensen.

- Utility (dirac_data.py) to export DIRAC data into a Python dictionary.
  Contributor: Lucas Visscher.

Revised features:

- Gauge origin (.GAUGEO and .GO ANG) now can only be set under
  \*\*HAMILTONIAN.
- Phase and dipole origins (.PHASEO and .DIPORG, respectively) now can
  only be set under \*\*HAMILTONIAN.
- EOM-CC excitation energies collected for all symmetries and summarized
  at the end.

New defaults:

- One-electron operator ANGMOM's origin was moved from the gauge-origin
  to the molecular center of mass.
- The CODATA2018 set of physical constants is now used as default (taken
  from <http://physics.nist.gov/constants>). Before this version,
  CODATA1998 was default.
- Non-relativistic and relativistic effective core-potential
  calculations (".NONREL" and ".ECP") now have nuclear point nucleus as
  default. For non-relativistic calculations this makes it easier to
  compare to other codes, and for RECP this is the intended behavior
  which by mistake was not activated.

------------------------------------------------------------------------

# 19.0 (2019-12-12)

New features:

- Calculation of Nuclear Spin-Rotation tensors. Activated with ".SPINRO". Contributors: Trond Saue and I. Agustín Aucar.  
  I. A. Aucar, S. S. Gómez, M. C. Ruiz de Azúa, and C. G. Giribet.
  Theoretical study of the nuclear spin-molecular rotation coupling for
  relativistic electrons and non-relativistic nuclei.

  10. Chem. Phys. 136, 204119 (2012);
      <http://dx.doi.org/10.1063/1.4721627> doi: 10.1063/1.4721627

- Interface with the Quantum Computing package OpenFermion. <https://github.com/bsenjean/Openfermion-Dirac>. Contributors: Bruno Senjean and Luuk Visscher.  
  T. E. O'Brien, B. Senjean, R. Sagastizabal, X. Bonet-Monroig, A.
  Dutkiewicz, F. Buda, L. DiCarlo and L. Visscher Calculating energy
  derivatives for quantum chemistry on a quantum computer. Accepted to
  npj: Quantum Information. <https://arxiv.org/abs/1905.03742>

- Expectation value of Nuclear Magnetic-Quadrupole-Moment interaction constant using ".OPERATOR" and ".NMQM" keyword in KRCI. Contributor: Malaya K. Nayak  
  T. Fleig, M. K. Nayak and M. G. Kozlov. TaN, a molecular system for
  probing P,T-violating hadron physics. Phys. Rev. A 93, 012505, 2016;
  <http://dx.doi.org/10.1103/PhysRevA.93.012505> doi:
  10.1103/PhysRevA.93.012505

Revised features:

- Maximum number of basis functions increased from 30k to 40k.

Error corrections:

- Fixed use of point charges (do not try to obtain e.g. mass of a point
  charge).
- Fixed spelling error in keyword .GAUNTSCALE (was .GAUTSCALE).
- Fixed T1EQN1 function call in src/relccsd/ccgrad.F (thanks to Martin
  Siegert).
- Fixed wrong memory allocation for spin-free or non-relativistic
  KRMCSCF.
- Fixed runtime error for XPROPMAT in src/krmc/krmccan.F property
  modules.
- Fixed runtime error for calling XPRIND in src/prp/pamprp.F property
  modules.

Basis set corrections:

- Fixed Be basis set in cc-pV5Z.
- Fixed Al basis set in aug-cc-pV(D+d)Z, aug-cc-pV(T+d)Z and
  aug-cc-pV(Q+d)Z.
- Fixed Si basis set in aug-cc-pV(D+d)Z and aug-cc-pV(Q+d)Z.

Performance improvements:

- Reduced static memory requirement from common blocks.

Documentation:

- Restore local build of Sphinx documentation.

------------------------------------------------------------------------

# 18.0 (2018-12-13)

New features:

- The visualization module has been extended to generate various radial
  distributions using the keyword .RADIAL. Contributor: Trond Saue.

Error corrections:

- Fix f correlating exponents for 3d in dyall.cv2z and dyall.ae2z basis
  sets. Contributor: Ken G. Dyall.
- Fix f polarizing exponents for 3d in dyall.cv3z and dyall.ae3z basis
  sets. Contributor: Ken G. Dyall.
- Bugfix in KRCI module (subroutine STRTYP_GAS_REL), reported by
  Shigeyoshi Yamamoto.
- Bugfix in polarizable embedding (the printed electronic part of the
  energy was not corrected with PE contributions).
- Fixed the formatting of the carbon potential in the file
  basis_ecp/ECPCE2SO.

Performance improvements:

- New much more efficient quaternion diagonalization routine using
  openMP for performance. Contributor: Hans Joergen Aa. Jensen.
- New much more efficient complex diagonalization routine using openMP
  for performance. Contributor: Hans Joergen Aa. Jensen.
- Use MPI_REDUCE for adding Fock matrix contributions from MPI worker
  nodes on MPI master node. Contributor: Hans Joergen Aa. Jensen.
- Save memory in direct Fock matrix constructions, now \>13000 basis
  functions can be run in 32GB RAM. Contributor: Hans Joergen Aa.
  Jensen.
- (With these performance improvements, a 4c HF test calculation with
  13000 basis functions, 3300 postive energy orbitals (6600 in total)
  has been run on Titan at ORNL with 800 nodes, each with 8 openMP
  threads, at 1h59m-2h06m per HF iteration.)

Framework changes:

- Allow to compile with OpenMP support.
- Update in CMake framework: pass compilers as cmake variables, not as
  environment variables.

------------------------------------------------------------------------

# 17.0 (2017-12-12)

New features:

- Kramers restricted Polarization Propagator in the ADC framework for electronic excitations, activated with ".POLPRP".  
  13. Pernpointner. The relativistic polarization propagator for the
      calculation of electronic excitations in heavy systems.
  14. Chem. Phys. 140, 084108 (2014);
      <http://dx.doi.org/10.1063/1.4865964>

  M. Pernpointner, L. Visscher and A. B. Trofimov. Four-component
  Polarization Propagator Calculations of Electron Excitations:
  Spectroscopic Implications of Spin-Orbit Coupling Effects. J. Chem.
  Theory Comput. (2017) submitted.

- Polarizable embedding using PElib
  (<https://gitlab.com/pe-software/pelib-public>). Activated with
  ".PEQM", additional options under \*PEQM.

> Reference: E. D. Hedegård, R. Bast, J. Kongsted, J. M. H. Olsen, and
> H. J. Aa. Jensen: Relativistic Polarizable Embedding., J. Chem. Theory
> Comput. 13, 2870-2880 (2017);
> <http://doi.org/10.1021/acs.jctc.7b00162>

- New and numerically stable procedure for elimination/freezing of
  orbitals at SCF level. Author: T. Saue.
- New ".MVOFAC" option in \*KRMC input section. Author: H. J. Aa.
  Jensen.
- New easier options for point charges in the .mol file: "LARGE
  POINTCHARGE" or "LARGE NOBASIS" (the two choices are equivalent).
- Added the RPF-4Z and aug-RPF-4Z basis sets for f-elements to the
  already existing files with sets for s, p and d elements. Deleted the
  aug-RPF-3Z set as that was not an official set.
- Write out effective Hamiltonian in Fock space coupled cluster to a
  file for post processing. Can be used with external code of Andrei
  Zaitsevskii (St. Petersburg).
- Provided memory counter for RelCC calculations, what is suitable for
  memory consuming large scale Coupled Cluster calculations. See the
  web-documentation.
- Updated basis_dalton/ with basis set updates in the Dalton
  distribution:
- fix of errors in Ahlrichs-pVDZ (several diffuse exponents were a
  factor 10 too big)
- fix of errors for 2. row atoms in aug-cc-pCV5Z
- added many atoms to aug-cc-PVTZ_J
- added many Frank Jensen "pc" type basis sets
- added Turbomole "def2" type basis sets

Error corrections:

- Fixed errors for quaternion symmetries in 2-electron MO integrals used
  in CI calculations with GASCIP. It is now possible to do CI
  calculations with GASCIP for C1 symmetry (i.e. no symmetry).
- Fixed the p exponents for Na in the dyall 4z basis sets to match the
  archive. The changes are small so should not significantly affect
  results.
- Fixed compilation of XCFun on Mac OS X High Sierra.
- Fixed error for parallel complex CI or MCSCF with GASCIP

New defaults:

- .SKIPEP is now default for KR-MCSCF, new keyword .WITHEP to include
  e-p rotations

Performance improvements:

- restored integral screening behavior from DIRAC15

------------------------------------------------------------------------

# 16.0 (2016-12-12)

New features:

- RELCCSD expectation values aka first-order unrelaxed analytic energy
  derivative. For more information, see J. Chem. Phys. 145 (2016) 184107
  as well as test/cc_gradient for an example.
- Calculation of approximate dipole intensities together with the
  excitation energies in Fock space CC. See test/fscc_intensities for an
  example.
- Improved start potential for SCF: sum of atomic LDA potentials,
  generated by GRASP.

New defaults:

- Negative denominators (e.g. appearing in core ionized systems)
  accepted in RELCCSD
- AOFOCK is now default if at least 25 MPI nodes (parallelizes better
  than SOFOCK). .AOFOCK documented.

Error corrections and updates in isotope properties for the following atoms:  
- Br isotope 2: quadrupole moment .2620 -\> .2615
- Ag isotope 2: magnetic moment .130563 -\> -.130691 (note sign change)
- In isotope 2: quadrupole moment .790 -\> .799
- Nd magnetic moments of isotopes 4 and 5 were interchanged: -0.065 -\>
  -1.065 and -1.065 -\> -0.065
- Gd: quadrupole meoments of isotopes 4 and 5 updated: 1.36 -\> 1.35 and
  1.30 -\> 1.27
- Ho isotope 1: quadrupole moment updated 3.49 -\> 3.58
- Lu isotope 2: quadrupole moment updtaed 4.92 -\> 4.97
- Hf isotope 1: mass was real\*4, not real\*8, thus 7 digits instead of
  179.9465457D0 (i.e. approx 179.9465)
- Ta isotope 1: quadrupole moment added 0.00 -\> 3,17
- Tl isotope 1: nuclear moment 1.63831461D0 -\> 1.63821461D0 (typo,
  error 1.d-4)
- Pb isotope 3: nuclear moment 0.582583D0 -\> 0.592583D0 (typo, error
  1.d-2)
- Po isotope 1: nuclear moment added: 0.000 -\> 0.688

Framework improvements:

- Faster Dirac compilation due to implemented OBJECT libraries (requires
  CMake of version at least 2.8.10).
- Run benchmark tests and only benchmark tests with "ctest -C benchmarks
  -L benchmark".
- Change any tab character in coordinate specifications in .mol file to
  a blank. (tab characters have been reported to cause errors in reading
  coordinates).
- Added OPENBLAS in search list for blas and lapack libraries.
- Made keyword .X2Cmmf case insensitive.
- Cache compiler flags.

Bug fixes:

- Bugfix: MCSCF and GASCIP was not correct for odd number of electrons.
- Bugfix: now .CASSCF works again for KRMSCF input.
- Bugfix: KRMSCF was not converging is some cases (problem reported by
  Miro Ilias for Os atom).
- Bugfix: "all" keyword for ".GAS SHELLS" was not recognized (problem
  reported by Miro Ilias).
- Bugfix for numerical gradient when automatic symmetry detection: do no
  move or rotate molecule for finding highest possible symmetry because
  then the numerical gradient will not be correct (problem reported by
  Miro Ilias).
- Bugfix for spinfree X2Cmmf and linear symmetry.
- Bugfix: restored "make install" target.
- Bugfix: Test reladc_fano does not report failure if Stieltjes is
  deactivated.

------------------------------------------------------------------------

# 15.0 (2015-12-16)

- FanoADC-Stieltjes: Calculation of decay widths of electronic decay
  processes.
- Better convergence for open-shell calculations, using .OPENFAC 0.5 as
  default. Final orbital energies are recalculated with .OPENFAC 1.0,
  for IP interpretation.
- Performance improvement for MP2 NO: only transform (G O[\|G O)
  integrals instead of (G
  G\|](##SUBST##|G O) integrals instead of (G G|)G G).
- Geometry optimization with xyz input.
- Performance improvements for determinant generation in GASCIP.
- Use Kramers conjugation on CI vectors (cuts time for CI in half for
  ESR doublets).
- ANO-RCC basis: Fixed Carbon basis set (wrong contraction coefficients,
  see \[MOLCAS ANO-RCC\](<http://www.molcas.org/ANO/>).
- ANO-RCC basis: Modified the 3 Th h-functions by replacing them with
  the 3 Ac h-functions to Th. (A mistake was made in the original work
  when the 3 Th h-functions were made, and this has never been redone.
  They are too diffuse, exponents (0.3140887600, 0.1256355100,
  0.0502542000) for Th, Z=90, compared to (0.7947153600, 0.3149038200,
  0.1259615200) for Ac, Z=89, and (0.8411791300, 0.3310795400,
  0.1324318200) for Pa, Z=91. We have selected to just replace the 3 Th
  h-functions with those from the Ac basis set, because the Ac
  g-functions are quite close tot he Th g-functions, closer than Ac
  g-functions, and therefore differences in results compared to
  optimized Th h-functions should be minimal.) Thanks to Kirk Peterson
  for pointing out the Th problem on <http://daltonforum.org>.
- Fixed reading of ANO-RCC and ANO-DK3 basis sets from the included
  basis set library.
- Update PCMSolver module to its \[1.0.4
  version\](<https://github.com/PCMSolver/pcmsolver/releases>).
- Do not write Hartree-Fock in output if it is a DFT calculation, write
  DFT.
- Fix a bunch of problems where the number of arguments of the call does
  not match the subroutine definition (submitted by Martin Siegert).
- Fixed plotting of the electrostatic potential (electronic part was
  factor 2 too large).
- Configuration framework uses \[Autocmake\](<http://autocmake.org>).

------------------------------------------------------------------------

# 14.1 (2015-07-20)

- Added OpenBLAS to default blas search list.
- Fixed misleading error message for .OPEN SHELL input.
- Workaround for CMake warning about nonexistent run_pamadm target.
- Intel compilers xHost flag automatic detection only upon request (-D
  ENABLE_XHOST_FLAG_DETECTION=ON).
- Fixed minor issues to allow compilation using IBM XL Fortran compiler
  (thanks to Bob Walkup, IBM).
- Update PCMSolver module to its \[1.0.3
  version\](<https://github.com/PCMSolver/pcmsolver/releases>).

------------------------------------------------------------------------

# 14.0 (2014-12-12)

- DIRAC14 release, see doc/release/release-statement.txt
