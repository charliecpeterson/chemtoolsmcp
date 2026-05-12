<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/dmrgscf.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.11. DMRGSCF

[previous](<dimerpert.html> "4.2.10. DIMERPERT ¤") | [next](<dynamix.html> "4.2.12. DYNAMIX") | [index](<../../genindex.html> "General Index")

# 4.2.11. DMRGSCF¶

  * DMRGSCF input section

    * DMRGSettings input section

    * OOptimizationSettings input section

  * Runtime options

  * Input files

  * Output files

  * Input example




The DMRGSCF program in Molcas performs multiconfigurational SCF calculations in which a density matrix renormalization group (DMRG) driver replaces a CI driver for the solution of the Complete Active Space (CAS) problem. In the latter a matrix-product state (MPS) wave function is obtained as an approximate solution to a full CI wave function in an active orbital space. In analogy to CASSCF, the DMRGSCF method is based on a partitioning of the occupied molecular orbitals into the following groups:

  * **Inactive orbitals:** Orbitals that are doubly occupied in all configurations.

  * **Active orbitals:** In these orbitals all possible occupations are allowed.

  * **Secondary orbitals:** Orbitals that are empty (unoccupied) in all configurations.




The DMRGSCF program currently supports only the **QCMaquis** DMRG program suite [[75](<../../references.html#id335> "S. Keller, M. Dolfi, M. Troyer, M. Reiher. J. Chem. Phys., 143 \(2015\) 244118."), [76](<../../references.html#id336> "S. Keller, M. Reiher. J. Chem. Phys., 144 \(2016\) 134101."), [77](<../../references.html#id337> "S. Knecht, E. D. Hedegård, S. Keller, A. Kovyrshin, Y. Ma, A. Muolo, C. J. Stein, M. Reiher. Chimia, 70 \(2016\) 244-251.")] as active space optimizer but in future it could be useful also for other DMRG programs interfaced to Molcas such as Block or CheMPS2. The latter two are currently activated through the RASSCF input.

For further information concerning input orbitals, convergence (acceleration) of the orbital optimization algorithm, dependencies, input orbitals, etc., we refer the reader to the introdcution of the RASSCF program (see [Section 4.2.47](<rasscf.html#ug-sec-rasscf>)) which is called by the DMRGSCF program in order to perform the actual orbital optimization. DMRGSCF calculations require to have only active orbitals in RAS2 (see the keyword list in the program RASSCF, [Section 4.2.47](<rasscf.html#ug-sec-rasscf>) for details).

**NOTE** : The DMRGSCF program does **NOT** support RASSCF/GASSCF calculations but it can be combined with MC-PDFT. For options concerning the latter, see the documentation of the MCPDFT program, [Section 4.2.34](<mcpdft.html#ug-sec-mcpdft>).

## 4.2.11.1. DMRGSCF input section¶

The DRMGSCF program is activated in general by
    
    
    &DMRGSCF &END
    ...
    End of Input
    

In the following we provide further input options for a DMRGSCF calculation.

ActiveSpaceOptimizer
    

**MANDATORY** : Sets the DMRG program to be used as active space optimizer. Currently the only choice is QCMaquis, i.e.
    
    
    ActiveSpaceOptimizer=QCMaquis
    

Fiedler
    

The Fiedler keyword, i.e.,
    
    
    Fiedler=on
    

enables a state-specific orbital ordering for the MPS optimization by exploiting concepts from graph theory. The ordering follows from the elements of the Fiedler vector which is the eigenvector corresponding to the second lowest eigenvalue of the so-called graph Laplacian.

CIDEAS
    

The CIDEAS keyword, i.e.,
    
    
    CIDEAS=on
    

enables a more advanced algorithm to construct a suitable initial MPS (see the keyword init_state in Table 7 of the [QCMaquis](<https://scine.ethz.ch/static/download/qcmaquis_manual.pdf>) manual for other options) provided by the configuration interaction dynamically extended active space (CI-DEAS) approach. The CI-DEAS protocol can be interpreted as an orbital entanglement entropy guided configuration selection and the quality of this initial guess depends on the quality of the initial CAS vector. The CI-DEAS functionality is currently restricted to calculations performed with C1 symmetry. Support for other point group symmetries will be available in due time. **Note** : The CIDEAS option requires to set the _HF occupation_ for each state in the OptimizationSettings input section below by means of the SOCC keyword.

### 4.2.11.1.1. DMRGSettings input section¶

DMRGSCF calculations require to set some DMRG-specific options, which will be passed on to the **QCMaquis** program. All mandatory keywords, which must be present in each calculation, are summarized below. In addition to those keywords, any **QCMaquis** keyword listed in Table 8 of the [QCmaquis](<https://scine.ethz.ch/static/download/qcmaquis_manual.pdf>) manual may be specified in this section. The start and end of the DMRGSettings input section is given by
    
    
    DMRGSettings
    ...
    EndDMRGSettings
    

max_bond_dimension
    

Maximum number of renormalized block states (commonly referred to as \\(m\\)-value or bond dimension) that will be kept during each microiteration step of a sweep.

nsweeps
    

Maximum number of DMRG sweeps. Please be aware that nsweeps sets the number of combined forward and backward sweeps. Thus, the actual number of sweeps is \\(2\times\\)nsweeps.

donotdelete
    

Set donotdelete=1 to restart DMRGSCF optimization from an existing QCMaquis MPS checkpoint. Useful e.g. to restart crashed calculations.

### 4.2.11.1.2. OOptimizationSettings input section¶

The Orbital OptimizationSettings block contains general, non DMRG-specific options required for the MPS wave function optimisation (such as number of the active electrons, active orbital specification etc.), i.e., a normal input for a CASSCF or a CASCI calculation with the RASSCF module. Most of the RASSCF keywords listed in the keyword section of [Section 4.2.47](<rasscf.html#ug-sec-rasscf>) are accepted, with the exception of keywords relating to explicit CI wave function quantities. Please consult the RASSCF module description for further details on the input. In addition to the standard RASSCF keywords, several optional keywords are available within DMRGSCF are listed below. The start and end of the OptimizationSettings input section is given by
    
    
    OOptimizationSettings
    ...
    EndOOptimizationSettings
    

FCIDUMP
    

Skip the wave function optimization and write out the transformed active MO integrals to a FCIDUMP file in $WorkDir which can be used in subsequent **QCMaquis** DMRG calculations.

SOCCupy
    

Initial electronic configuration for the calculated state(s). This keyword is equivalent to the hf_occ card in the **QCMaquis** input (see Table 8 of the [QCMaquis](<https://scine.ethz.ch/static/download/qcmaquis_manual.pdf>) manual), but allows input for multiple states. The occupation is inserted as a string (strings) of aliases of occupations of the active (RAS2) orbitals with the aliases `2` = full, `u` = up, `d` = down, `0` = empty. For several states, the occupation strings for each state are separated by newlines.

NEVPT2prep
    

Prepare for a subsequent DMRG-NEVPT2 or CASPT2 calculation. Then the four- and transition three-particle density matrices (4- and t-3RDMs), required for the MRPT2 calculations, will be evaluated and stored on disk in $WorkDir. **QCMaquis** input files for the 4- and t-3RDMs evaluation are prepared and the RDM evaluation may be performed externally or directly in the NEVPT2 program. More about external RDM evaluation in Section 6.3 of the [QCMaquis](<https://scine.ethz.ch/static/download/qcmaquis_manual.pdf>) manual. If this keyword is used with ITER=0,0 keyword, the DMRG-SCF/DMRG-CI calculation is skipped and only **QCMaquis** input files for the 4- and t-3RDMs evaluation are prepared. This is useful for a NEVPT2 calculation for an already converged DMRG-CI/DMRG-SCF calculation.

## 4.2.11.2. Runtime options¶

**QCMaquis** is built by default with a shared-memory OMP parallelization. To speed up calculations the user can thus set at runtime the environment variable QCMaquis_CPUS or OMP_NUM_THREADS to the number of shared-memory cores to be used. Example:
    
    
    >>> EXPORT QCMaquis_CPUS=16
    

The default is to use a single core.

## 4.2.11.3. Input files¶

DMRGSCF will use (in analogy to RASSCF) the following input files: ONEINT, ORDINT, RUNFILE, INPORB, JOBIPH (for more information see [Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)). We strongly recommend to use the HDF5 files ($Project.ProgramName.h5) produced by the wave function modules in Molcas as orbital input files, see the keyword FILEORB in the RASSCF input, [Section 4.2.47](<rasscf.html#ug-sec-rasscf>) for further details.

A number of additional files generated by SEWARD are also used by the DMRGSCF program. The availability of either of the files named INPORB and JOBOLD is optional and determined by the input options LUMORB and JOBIPH, respectively.

## 4.2.11.4. Output files¶

JOBIPH
    

This file is written in binary format and carries the results of the wave function optimization such as MO- and CI-coefficients. If several consecutive RASSCF calculations are made, the file names will be modified by appending “01”,”02”, etc.

RUNFILE
    

The RUNFILE is updated with information from the RASSCF calculation such as the first order density and the Fock matrix.

MD_CAS.x
    

Molden input file for molecular orbital analysis for MPS state x.

RASORB
    

This ASCII file contains molecular orbitals, occupation numbers, and orbital indices from a DMRGSCF calculation. The natural orbitals of individual states in an average-state calculation are also produced, and are named RASORB.1, RASORB.2, etc.

MCDENS
    

This ASCII file is generated for MC-PDFT calculations. It contains spin densities, total density and on-top pair density values on grid (coordinates in a.u.).

dmrgscf.h5
    

This .h5 file contains contains molecular orbitals, occupation numbers, and orbital indices from a DMRGSCF calculation. In addition, it stores the names of the **QCMaquis** output files.

checkpoint_state.x.h5
    

Directory containing the MPS for state x.

results_state.x.h5
    

File containing the MPS optimization information and property data calculated for state x.

## 4.2.11.5. Input example¶

The following example shows the input to the DMRGSCF program for a calculation on the nitrogen molecule. The calculation is performed in \\(D_{2h}\\) symmetry. The max_bond_dimension is set to 100, which is sufficient for a small CAS(6,6) problem.
    
    
    &GATEWAY
     coord
     2
    angstrom
     N       0.000000  0.000000  -0.54880
     N       0.000000  0.000000   0.54880
     basis=cc-pvdz
    &SEWARD
    &SCF
    &DMRGSCF
    ActiveSpaceOptimizer=QCMaquis
    DMRGSettings
      conv_thresh        = 1e-4
      truncation_final   = 1e-5
      ietl_jcd_tol       = 1e-6
      nsweeps            = 4
      max_bond_dimension = 100
    EndDMRGSettings
    OOptimizationSettings
      inactive = 2 0 0 0 2 0 0 0
      RAS2     = 1 1 1 0 1 1 1 0
      ITER     = 15,100
      SOCC     = 2,2,2,0,0,0
      LINEAR
    EndOOptimizationSettings
    

### Table of Contents

  * [1\. Introduction](<../../intro.html>)
  * [2\. Installation Guide](<../../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../../tutorials/tut.html>)
  * [4\. User’s Guide](<../ug.html>)
    * [4.1. The Molcas environment](<../env-main.html>)
    * [4.2. Programs](<../programs.html>)
      * [4.2.1. ALASKA](<alaska.html>)
      * [4.2.2. AVERD](<averd.html>)
      * [4.2.3. CASPT2](<caspt2.html>)
      * [4.2.4. CASVB](<casvb.html>)
      * [4.2.5. CCSDT](<ccsdt.html>)
      * [4.2.6. CHCC](<chcc.html>)
      * [4.2.7. CHT3](<cht3.html>)
      * [4.2.8. CMOCORR ¤](<cmocorr.html>)
      * [4.2.9. CPF](<cpf.html>)
      * [4.2.10. DIMERPERT ¤](<dimerpert.html>)
      * 4.2.11. DMRGSCF
      * [4.2.12. DYNAMIX](<dynamix.html>)
      * [4.2.13. EMBQ ¤](<embq.html>)
      * [4.2.14. ESPF (+ QM/MM interface)](<espf.html>)
      * [4.2.15. EXPBAS](<expbas.html>)
      * [4.2.16. EXTF](<extf.html>)
      * [4.2.17. FALCON ¤](<falcon.html>)
      * [4.2.18. FALSE](<false.html>)
      * [4.2.19. FFPT](<ffpt.html>)
      * [4.2.20. GATEWAY](<gateway.html>)
      * [4.2.21. GENANO](<genano.html>)
      * [4.2.22. GEO ¤](<geo.html>)
      * [4.2.23. GRID_IT](<grid_it.html>)
      * [4.2.24. GUESSORB](<guessorb.html>)
      * [4.2.25. GUGA](<guga.html>)
      * [4.2.26. GUGACI](<gugaci.html>)
      * [4.2.27. GUGADRT](<gugadrt.html>)
      * [4.2.28. LEVEL](<level.html>)
      * [4.2.29. LOCALISATION](<localisation.html>)
      * [4.2.30. LOPROP](<loprop.html>)
      * [4.2.31. MBPT2](<mbpt2.html>)
      * [4.2.32. MCKINLEY (a.k.a. DENALI)](<mckinley.html>)
      * [4.2.33. MCLR](<mclr.html>)
      * [4.2.34. MCPDFT](<mcpdft.html>)
      * [4.2.35. MKNEMO ¤](<mknemo.html>)
      * [4.2.36. MOTRA](<motra.html>)
      * [4.2.37. MPPROP](<mpprop.html>)
      * [4.2.38. MPSSI](<mpssi.html>)
      * [4.2.39. MRCI](<mrci.html>)
      * [4.2.40. MULA](<mula.html>)
      * [4.2.41. NEMO ¤](<nemo.html>)
      * [4.2.42. NEVPT2](<nevpt2.html>)
      * [4.2.43. NUMERICAL_GRADIENT](<numerical_gradient.html>)
      * [4.2.44. POLY_ANISO](<poly_aniso.html>)
      * [4.2.45. QMSTAT](<qmstat.html>)
      * [4.2.46. QUATER](<quater.html>)
      * [4.2.47. RASSCF](<rasscf.html>)
      * [4.2.48. RASSI](<rassi.html>)
      * [4.2.49. RHODYN](<rhodyn.html>)
      * [4.2.50. RPA](<rpa.html>)
      * [4.2.51. SCF](<scf.html>)
      * [4.2.52. SEWARD](<seward.html>)
      * [4.2.53. SINGLE_ANISO](<single_aniso.html>)
      * [4.2.54. SLAPAF](<slapaf.html>)
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<dimerpert.html> "4.2.10. DIMERPERT ¤") | [next](<dynamix.html> "4.2.12. DYNAMIX") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/dmrgscf.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
