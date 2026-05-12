<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/mpssi.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.38. MPSSI

[previous](<mpprop.html> "4.2.37. MPPROP") | [next](<mrci.html> "4.2.39. MRCI") | [index](<../../genindex.html> "General Index")

# 4.2.38. MPSSI¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example




The MPSSI (MPS State Interaction) program [[108](<../../references.html#id334> "S. Knecht, S. Keller, J. Autschbach, M. Reiher. J. Chem. Theory Comput., 12 \(2016\) 5881-5894.")] forms overlaps and other matrix elements of the Hamiltonian and other operators over a wave function basis, which consists of matrix-product state (MPS) wave functions, each with an individual set of orbitals. Following the philosophy of the RASSI program, it is a generalized state-interaction approach for both nonorthogonal and orthonormal spinfree MPS wave functions which enables the evaluation of arbitrary one- and two-particle transition matrix elements as well as, for example, matrix elements of the spin-orbit coupling operator. For instance, diagonalization of the spin-orbit Hamiltonian matrix yields spin-orbit coupled wave functions as linear combinations of the uncoupled, spin-pure MPS states. The latter can (but do not have to) be obtained as results from one or several DMRG-SCF orbital optimization calculations (see DMRGSCF).

Following the work of Malmqvist [[109](<../../references.html#id30> "P.-Å. Malmqvist, B. O. Roos. Chem. Phys. Lett., 155 \(1989\) 189-194.")], the central element of the MPS-SI algorithm is the transformation of the bra and ket MPS wave functions to a biorthonormal basis representation. It is important to note that the latter transformation is not needed if the MPS wave functions considered for state interaction share a common MO basis. In this particular case, the MPS-SI program directly proceeds with the calculation of the reduced (transition) one- and two-particle density matrices. We emphasize that our approach is applicable to the general case with MPS wave functions built from mutually nonorthogonal molecular orbital bases. It therefore provides the desired flexibility to find the best individual molecular orbital basis to represent wave functions of different spin and/or spatial symmetry. After solving a generalized eigenvalue equation of the form

(4.2.38.1)¶\\[Hc = ESc\\]

with the Hamiltonian matrix \\(H\\) expressed in the basis of the DMRG-SCF MPS wave functions and the wave function overlap matrix \\(S\\), a set of fully orthogonal and noninteracting states are obtained as linear combinations of the DMRG-SCF MPS wave functions with the expansion coefficients given by \\(c\\) in Eq. (4.2.38.1).

Apart from computing oscillator strengths, overlaps and Hamiltonian matrix elements can be used to compute electron transfer rates, or to form quasi-diabatic states and reexpress matrix elements over a basis of such states.

Moreover, it is possible to “dress” the diagonal elements of the Hamiltonian in Eq. (4.2.38.1) for MPS-SI by adding a correlation-correction term obtained, for example, from a preceding NEVPT2 calculation (see Section 6), by either using the HDIAG keyword within the RASSI module or provide the nevpt2.h5 wave function file as input

## 4.2.38.1. Dependencies¶

The MPSSI program needs one or more dmrgscf.h5 files produced by the DMRGSCF program (or if MPSSI is running subseuqently after a NEVPT2 calculation one or more nevpt2.h5 files). Also, it needs a ONEINT file from SEWARD, with overlap integrals and any one-electron property integrals for the requested matrix elements. If Hamiltonian matrix elements are used, also the ORDINT file is needed.

or further information see the description of the RASSI program ([Section 4.2.48](<rassi.html#ug-sec-rassi>)).

## 4.2.38.2. Files¶

### 4.2.38.2.1. Input files¶

ORDINT*
    

Ordered two-electron integral file produced by the SEWARD program. In reality, this is up to 10 files in a multi-file system, named ORDINT, ORDINT1,…,ORDINT9. This is necessary on some platforms in order to store large amounts of data.

ONEINT
    

The one-electron integral file from SEWARD

dmrgscf.h5
    

A number of dmrgscf.h5 files from different DMRGSCF jobs.

### 4.2.38.2.2. Output files¶

SIORBnn
    

A number of files containing natural orbitals, (numbered sequentially as SIORB01, SIORB02, etc.)

BRAORBnnmm, KETORBnnmm
    

A number of files containing binatural orbitals for the transition between states nn and mm.

TOFILE
    

This output is only created if TOFIle is given in the input. It will contain the transition density matrix computed by MPSSI. Currently, this file is only used as input to QMSTAT (NOT TESTED!).

EIGV
    

Like TOFILE this file is only created if TOFIle is given in the input. It contains auxiliary information that is picked up by QMSTAT (NOT TESTED!).

## 4.2.38.3. Input¶

This section describes the input to the MPSSI program in the Molcas program system, with the program name:
    
    
    &MPSSI
    

When a keyword is followed by additional mandatory lines of input, this sequence cannot be interrupted by a comment line. The first 4 characters of keywords are decoded. An unidentified keyword makes the program stop. Note that MPSSI shares **ALL** keywords with RASSI which do **NOT** request CI-type quantities. Below is just a list of additional keywords available for enabling the effective Hamiltonian from a preceding NEVPT2 calculation, in order to achieve a state-dressing.

### 4.2.38.3.1. Keywords¶

QDSC
    

Enable the effective Hamiltonian from a quasi-degenerate (QD) multi-state strongly-contracted i(SC) NEVPT2 calculation.

QDPC
    

Enable the effective Hamiltonian from a quasi-degenerate (QD) multi-state partially-contracted (PC) NEVPT2 calculation.

### 4.2.38.3.2. Input example¶

An example with single JobIph:
    
    
    &MPSSI
    NrofJobIphs
    1 2           --- 1 JobIph (actually an .h5 file) - 2 states to be read
    1 2           --- which roots from the .h5 file.
    FILE
    1
    n2+.dmrgscf.h5
    omega
    SPIN
    EPRG
    1.0
    MEES
    PROPerties
      3
    'AngMom' 1
    'AngMom' 2
    'AngMom' 3
    * This input will compute spinfree and spin-orbit eigenstates in the space
    * spanned by the 2 input functions
    

An example with two separate JobIphs (singlet and triplet calculation of methylene):
    
    
    * Triplet calculation
    &DMRGSCF
      ActiveSpaceOptimizer=QCMaquis
      DMRGSettings
        max_bond_dimension=1024
        nsweeps=10
      EndDMRGSettings
      OOptimizationSettings
        Spin=3
        Inactive=1
        Ras2=6
        NActEl=6,0,0
      EndOOptimizationSettings
    * Save JobIph, because it will be overwritten by the subsequent calculation
    >> COPY $Project.JobIph JOBOLD
    >> COPY $Project.dmrgscf.h5 $Project.trip.h5
    * Save QCMaquis checkpoint since it will also be overwritten.
    * COPY does not work on directories so we move it
    >> EXEC mv $CurrDir/$Project.checkpoint_state.0.h5 $CurrDir/$Project.trip.checkpoint_state.0.h5
    * The rasscf.h5 file contains the QCMaquis checkpoint file name.
    * Now that QCMaquis checkpoint has been renamed, the name needs to
    * be changed in the rasscf.h5 file. The script below accomplishes this
    >> EXEC $MOLCAS/Tools/qcmaquis/qcm_checkpoint_rename.py $Project.trip.h5 -q
    
    * Singlet calculation
    &DMRGSCF
      ActiveSpaceOptimizer=QCMaquis
      DMRGSettings
        max_bond_dimension=1024
        nsweeps=10
      EndDMRGSettings
      OOptimizationSettings
        Spin=1
        Inactive=1
        Ras2=6
        NActEl=6,0,0
        JobIph
      EndOOptimizationSettings
    
    * Perform checkpoint manipulations as with triplet
    >> COPY $Project.dmrgscf.h5 $Project.sing.h5
    >> EXEC mv $CurrDir/$Project.checkpoint_state.0.h5 $CurrDir/$Project.sing.checkpoint_state.0.h5
    >> EXEC $MOLCAS/Tools/qcmaquis/qcm_checkpoint_rename.py $Project.sing.h5 -q
    
    * Run MPSSI
    &MPSSI
    Nrof
    2 1 1
    1
    1
    FILE
    2
    $Project.trip.h5
    $Project.sing.h5
    EJOB
    SOCOupling
    0.0001
    

The input is similar to an analogous RASSI input, with a notable exception of manipulations of QCMaquis checkpoints and rasscf.h5 files. Since the MPS is stored in QCMaquis checkpoint folders, these have to be saved in addition to the rasscf.h5 file. In addition, rasscf.h5 saves the QCMaquis checkpoint file name, so when the latter is renamed, also the name saved in rasscf.h5 must be changed. This is accomplished with the command
    
    
    $MOLCAS/Tools/qcmaquis/qcm_checkpoint_rename.py <rasscf.h5> -q
    

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
      * 4.2.38. MPSSI
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

[previous](<mpprop.html> "4.2.37. MPPROP") | [next](<mrci.html> "4.2.39. MRCI") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/mpssi.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
