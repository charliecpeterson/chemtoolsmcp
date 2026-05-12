<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/nevpt2.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.42. NEVPT2

[previous](<nemo.html> "4.2.41. NEMO ¤") | [next](<numerical_gradient.html> "4.2.43. NUMERICAL_GRADIENT") | [index](<../../genindex.html> "General Index")

# 4.2.42. NEVPT2¶

  * Dependencies

  * Input files

  * Output files

  * NEVPT2 input

  * Input example




NEVPT2 is a second-order perturbation theory with a CAS (or a CAS-like) reference wavefunction originally developed by Angeli et al. [[21](<../../references.html#id346> "C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu. J. Chem. Phys., 114 \(2001\) 10252."), [22](<../../references.html#id347> "C. Angeli, R. Cimiraglia, J.-P. Malrieu. Chem. Phys. Lett., 350 \(2001\) 297-305."), [23](<../../references.html#id348> "C. Angeli, R. Cimiraglia, J.-P. Malrieu. J. Chem. Phys., 117 \(2002\) 9138-9153."), [24](<../../references.html#id349> "C. Angeli, S. Borini, M. Cestari, R. Cimiraglia. J. Chem. Phys., 121 \(2004\) 4043-4049.")] In contrast to CASPT2, it uses a Dyall Hamiltonian [[25](<../../references.html#id350> "K. G. Dyall. J. Chem. Phys., 102 \(1995\) 4909-4918.")] as the zeroth-order Hamiltonian and is therefore inherently free of intruder states and parameters such as the IPEA shift. NEVPT2 exists in two formulations – the strongly- (SC-) and the partially-contracted NEVPT2 (PC-NEVPT2), which differ in the basis of the first-order wavefunction expansion.

The implementation in the NEVPT2 program is based on the original NEVPT2 implementation by Angeli et al. [[23](<../../references.html#id348> "C. Angeli, R. Cimiraglia, J.-P. Malrieu. J. Chem. Phys., 117 \(2002\) 9138-9153."), [24](<../../references.html#id349> "C. Angeli, S. Borini, M. Cestari, R. Cimiraglia. J. Chem. Phys., 121 \(2004\) 4043-4049.")], with the implementation of the QCMaquis DMRG reference wave function and Cholesky decomposition for the two-electron integrals [[26](<../../references.html#id351> "L. Freitag, S. Knecht, C. Angeli, M. Reiher. J. Chem. Theory Comput., 13 \(2017\) 451-459.")]. For excited states both single-state and multi-state calculations with the QD-NEVPT2 approach [[24](<../../references.html#id349> "C. Angeli, S. Borini, M. Cestari, R. Cimiraglia. J. Chem. Phys., 121 \(2004\) 4043-4049.")] are supported.

## 4.2.42.1. Dependencies¶

The NEVPT2 program needs the JOBIPH file (or its HDF5 equivalent) with a reference wavefunction a from a RASSCF/DMRGSCF calculation. Currently, **only DMRG reference wavefunctions calculated with QCMaquis** are supported. Additionally, transformed MO integrals or Cholesky vectors from MOTRA are required.

Optionally, four-particle reduced density matrices (and transition three-particle reduced density matrices for QD-NEVPT2 calculations) can be precalculated with QCMaquis in a massively parallel fashion and stored on disk. These QCMaquis calculations may be prepared and executed with the help of two scripts found in $MOLCAS/Tools/distributed-4rdm directory, namely jobmanager.py and prepare_rdm_template.sh. The distributed RDM evaluation is strongly recommended for active spaces larger than 10-11 orbitals and is described in detail in [Section 3.3.9.2](<../../tutorials/tut_nevpt2.html#tut-sec-nevpt2-distrdm>).

## 4.2.42.2. Input files¶

JobIph or dmrgscf.h5
    

File containing information about the reference wavefunction.

ijkl.h5
    

Transformed integrals or Cholesky vectors, calculated by the MOTRA program.

## 4.2.42.3. Output files¶

nevpt2.h5
    

File in HDF5 format, similar to RASSCF/DMRGSCF dmrgscf.h5 files, containing the effective Hamiltonian for QD-NEVPT2 calculations (both strongly- and partially-contracted).

## 4.2.42.4. NEVPT2 input¶

The NEVPT2 program is activated by
    
    
    &NEVPT2
    

The optional keywords supported by NEVPT2 are listed below.

STATES
    

Number of electronic states to calculate. Default: 1

NOMS
    

Omit the QD-NEVPT2 calculation and perform single-state NEVPT2 calculations instead.

MULT
    

Select specific states to perform QD-NEVPT2 calculation. Followed by a list of whitespace-separated state numbers, preceded by their total amount. Example: `MULT=3 1 2 4` for states 1, 2, 4 of a preceding DMRG-SCF calculation of 4 roots (or more). `MULT=ALL` includes all states and is the default.

FILE
    

Specify the path to a JobIph or .h5 file with the reference wavefunction. By default, the reference wavefunction is read from JOBIPH.

FROZEN
    

Specify the number of frozen orbitals. The number of frozen orbitals may be specified in two ways: if only one number \\(n\\) is specified, then all orbitals from 1 to \\(n\\) are frozen. Otherwise, it is possible to freeze specific orbitals with the SELECT keyword which follows the FROZEN keyword. In this case, the total number of frozen orbitals followed by the space-separated list of frozen orbitals must be entered. Note that if symmetry is used, the orbital numbering for all symmetries is still consecutive, e.g. the 1st orbital of symmetry 2 is has the number \\(m+1\\) if there are \\(m\\) orbitals in symmetry 1.

If frozen orbitals are specified in MOTRA input, they will be autodetected in NEVPT2 and there is no need to input them separately, so that this keyword is not needed.

NOPC
    

Disable the PC-NEVPT2 calculation. If the option is not present (default), both SC-NEVPT2 and PC-NEVPT2 calculations are performed.

SKIPK
    

Skip the calculation of Koopmans’ matrices. Requires a file named nevpt.h5 obtained from a previous calculation in the scratch directory. May be useful to restart a previous crashed calculation if it crashed past the calculation of Koopmans’ matrices, and may save some computational time, especially for large active spaces.

RDMRead
    

Do not calculate the 4-RDM, but rather read it from QCMaquis result files $Project.results_state.X.h5 for state `X`. Useful if the previous calculation crashed but the 4-RDM evaluation step has succeeded. Do NOT use it if you are using the distributed 4-RDM calculation.

DISTributedRDM
    

Enable reading of the RDMs calculated with the distributed RDM evaluation script. This keyword should be followed by another line, which specifies the path to the folder with the calculation results. The 4-RDM will then be read from QCMaquis HDF5 files found in <path>/4rdm-scratch.<state>/parts/part-*/$Project.results_state.<state>.h5. The distributed \\(n\\)-RDM evaluation is described in the NEVPT2 program-based tutorial. If the tutorial is followed, the path should be $WorkDir.

## 4.2.42.5. Input example¶

An input example for NEVPT2 may be found in [Section 3.3.9.1](<../../tutorials/tut_nevpt2.html#tut-sec-nevpt2-run>).

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
      * [4.2.11. DMRGSCF](<dmrgscf.html>)
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
      * 4.2.42. NEVPT2
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

[previous](<nemo.html> "4.2.41. NEMO ¤") | [next](<numerical_gradient.html> "4.2.43. NUMERICAL_GRADIENT") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/nevpt2.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
