<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_ccsdt.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.16. CCSDT — A Set of Coupled-Cluster Programs

[previous](<tut_cpf.html> "3.3.15. CPF — A Coupled-Pair Functional Program") | [next](<tut_optimiza.html> "3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization") | [index](<../genindex.html> "General Index")

# 3.3.16. CCSDT — A Set of Coupled-Cluster Programs¶

The Molcas program CCSDT computes Coupled-Cluster Singles Doubles, CCSD, and Coupled-Cluster Singles Doubles and Non-iterative Triples Correction CCSD(T) wave functions for restricted single reference both closed- and open-shell systems.

In addition to the ONEINT and ORDINT integral files (in non-Cholesky calculations), the CCSDT code requires the JOBIPH file containing the reference wave function (remember that it is not possible to compute open-shell systems with the SCF program) and the transformed two-electron integrals produced by the MOTRA module and stored in the TRAINT file.

Previously to execute the CCSDT module, wave functions and integrals have to be prepared. First, a RASSCF calculation has to be run in such a way that the resulting wave function has one single reference. In closed-shell situations this means to include all the orbitals as inactive and set the number of active electrons to zero. Keyword OUTOrbitals followed by the specification CANOnical must be used in the RASSCF input to activate the construction of canonical orbitals and the calculation of the CI-vectors on the basis of the canonical orbitals. After that the MOTRA module has to be run to transform the two-electron integrals using the molecular orbitals provided by the RASSCF module. The files JOBIPH or RASORB from the RASSCF calculation can be used directly by MOTRA using the keywords JOBIph or LUMOrb in the MOTRA input. Frozen or deleted orbitals can be introduced in the transformation step by the proper options in the MOTRA input.

## 3.3.16.1. CCSDT Outputs¶

The section of the Molcas output corresponding to the CC program is self explanatory. The default output simply contains the wave function specifications from the previous RASSCF calculation, the orbital specifications, the diagonal Fock matrix elements and orbital energies, the technical description of the calculation, the iterations leading to the CCSD energy, and the five largest amplitudes of each type, which will help to evaluate the calculation. If triples excitations have been required the description of the employed method (from the three available) to compute perturbatively the triple excited contributions to the CC energy, the value of the correction, and the energy decomposition into spin parts will be available.

## 3.3.16.2. Example of a CCSD(T) calculation¶

Block 3.3.16.1 contains the input files required by the SEWARD, SCF, RASSCF, MOTRA and CCSDT programs to compute the ground state of the \\(\ce{HF^+}\\) cation. molecule, which is a doublet of \\(\Sigma^+\\) symmetry. A more detailed description of the different options included in the input of the programs can be found in the CCSDT section of the user’s guide. This example describes how to calculate CCSD(T) energy for \\(\ce{HF^+}\\) cation. This cation can be safely represented by the single determinant as a reference function, so one can assume that CCSD(T) method will be suitable for its description.

The calculation can be divided into few steps:

  1. Run SEWARD to generate AO integrals.

  2. Calculate the HF molecule at the one electron level using SCF to prepare an estimate of MO for the RASSCF run.

  3. Calculate \\(\ce{HF^+}\\) cation by subtracting one electron from the orbital with the first symmetry. There is only one electron in one active orbital so only one configuration is created. Hence, we obtain a simple single determinant ROHF reference.

  4. Perform MO transformation exploiting MOTRA using MO coefficients from the RASSCF run.

  5. Perform the Coupled Cluster calculation using CCSDT program. First, the data produced by the programs RASSCF and MOTRA need to be reorganized, then the CCSD calculation follows, with the chosen spin adaptation being T2 DDVV. Finally, the noniterative triple excitation contribution calculation is following, where the CCSD amplitudes are used.




This is an open shell case, so it is suitable to choose CCSD(T) method as it is defined by Watts _et al._ [[27](<../references.html#id132> "J. D. Watts, J. Gauss, R. J. Bartlett. J. Chem. Phys., 98 \(1993\) 8718-8733.")]. Since CCSD amplitudes produced by previous CCSD run are partly spin adapted and denominators are produced from the corresponding diagonal Fock matrix elements, final energy is sometimes referred as SA1 CCSD(T)\\(_d\\) (see [[28](<../references.html#id66> "P. Neogrády, M. Urban. Int. J. Quantum Chem., 55 \(1995\) 187-203.")]).

Block 3.3.16.1 Sample input containing the files required by the SEWARD, SCF, RASSCF, MOTRA, CCSORT, CCSD, and CCT3 programs to compute the ground state of the \\(\ce{HF^+}\\) cation.¶
    
    
    &SEWARD &END
    Title= HF molecule
    Symmetry
    X Y
    Basis set
    F.ANO-S-VDZ
    F      0.00000   0.00000   1.73300
    End of basis
    Basis set
    H.ANO-S-VDZ
    H      0.00000   0.00000   0.00000
    End of basis
    End of input
    &SCF
    &RASSCF
    Title= HF(+) cation
    OUTOrbitals= Canonical
    Symmetry= 1; Spin= 2
    nActEl= 1 0 0; Inactive= 2 1 1 0; Ras2= 1 0 0 0
    LumOrb; OUTOrbitals= Canonical
    &MOTRA; JobIph; Frozen= 1 0 0 0
    &CCSDT
    Iterations= 50; Shift= 0.2,0.2; Accuracy= 1.0d-7
    Denominators= 2; Extrapolation= 5,4
    Adaptation= 1; Triples= 3; T3Denominators= 0
    

RASSCF calculates the HF ionized state by removing one electron from the orbital in the first symmetry. Do not forget to use keyword CANONICAL. In the CCSDT run, the number of iterations is limited to 50. Denominators will be formed using orbital energies. (This corresponds to the chosen spin adaptation.) Orbitals will be shifted by 0.2 au, what will accelerate the convergence. However, final energy will not be affected by the chosen type of denominators and orbital shifts. Required accuracy is 10\\(^{-7}\\) au. for the energy. T2 DDVV class of CCSD amplitudes will be spin adapted. To accelerate the convergence, DIIS procedure is exploited. It will start after 5th iteration and the last four iterations will be taken into account in each extrapolation step.

In the triples step the CCSD(T) procedure as defined by Watts _et al._ [[27](<../references.html#id132> "J. D. Watts, J. Gauss, R. J. Bartlett. J. Chem. Phys., 98 \(1993\) 8718-8733.")] will be performed. Corresponding denominators will be produced using diagonal Fock matrix elements.

## 3.3.16.3. CCSDT — Basic and Most Common Keywords¶

CCSD
    

Coupled-cluster singles and doubles method

CCT
    

CCSD plus a non iterative triples (T) calculation

### Table of Contents

  * [1\. Introduction](<../intro.html>)
  * [2\. Installation Guide](<../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<tut.html>)
    * [3.1. Quickstart Guide for Molcas](<nutshell.html>)
    * [3.2. Problem Based Tutorials](<pbtutorials.html>)
    * [3.3. Program Based Tutorials](<tutorials.html>)
      * [3.3.1. Molcas Flowchart](<flowchart.html>)
      * [3.3.2. Environment and EMIL Commands](<tut_emil.html>)
      * [3.3.3. GATEWAY — Definition of geometry, basis sets, and symmetry](<tut_gateway.html>)
      * [3.3.4. SEWARD — An Integral Generation Program](<tut_seward.html>)
      * [3.3.5. SCF — A Self-Consistent Field program and Kohn–Sham DFT](<tut_scf.html>)
      * [3.3.6. MBPT2 — A Second-Order Many-Body PT RHF Program](<tut_mbpt2.html>)
      * [3.3.7. RASSCF — A Multi Configurational Self-Consistent Field Program](<tut_rasscf.html>)
      * [3.3.8. CASPT2 — A Many Body Perturbation Program](<tut_caspt2.html>)
      * [3.3.9. NEVPT2 — \\(n\\)-Electron Valence State Second-Order Perturbation Theory](<tut_nevpt2.html>)
      * [3.3.10. RASSI — A RAS State Interaction Program](<tut_rassi.html>)
      * [3.3.11. CASVB — A non-orthogonal MCSCF program](<tut_casvb.html>)
      * [3.3.12. MOTRA — An Integral Transformation Program](<tut_motra.html>)
      * [3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program](<tut_guga.html>)
      * [3.3.14. MRCI — A Configuration Interaction Program](<tut_mrci.html>)
      * [3.3.15. CPF — A Coupled-Pair Functional Program](<tut_cpf.html>)
      * 3.3.16. CCSDT — A Set of Coupled-Cluster Programs
      * [3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization](<tut_optimiza.html>)
      * [3.3.18. MCKINLEY — A Program for Integral Second Derivatives](<tut_mckinley.html>)
      * [3.3.19. MCLR — A Program for Linear Response Calculations](<tut_mclr.html>)
      * [3.3.20. GENANO — A Program to Generate ANO Basis Sets](<tut_genano.html>)
      * [3.3.21. FFPT — A Finite Field Perturbation Program](<tut_ffpt.html>)
      * [3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules](<tut_vibrot.html>)
      * [3.3.23. SINGLE_ANISO — A Magnetism of Complexes Program](<tut_single_aniso.html>)
      * [3.3.24. POLY_ANISO — Semi-_ab initio_ Electronic Structure and Magnetism of Polynuclear Complexes Program](<tut_poly_aniso.html>)
      * [3.3.25. GRID_IT — A Program for Orbital Visualization](<tut_grid_it.html>)
      * [3.3.26. Writing MOLDEN input](<tut_molden.html>)
      * [3.3.27. Tools for selection of the active space](<tut_expbas.html>)
      * [3.3.28. Most frequent error messages found in Molcas](<tut_errors.html>)
      * [3.3.29. Some practical hints](<tut_hints.html>)
  * [4\. User’s Guide](<../users.guide/ug.html>)
  * [5\. Advanced Examples and Annexes](<../advanced.examples/ae.html>)



### Search

[previous](<tut_cpf.html> "3.3.15. CPF — A Coupled-Pair Functional Program") | [next](<tut_optimiza.html> "3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_ccsdt.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
