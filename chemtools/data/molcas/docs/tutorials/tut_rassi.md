<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_rassi.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.10. RASSI — A RAS State Interaction Program

[previous](<tut_nevpt2.html> "3.3.9. NEVPT2 — \\\(n\\\)-Electron Valence State Second-Order Perturbation Theory") | [next](<tut_casvb.html> "3.3.11. CASVB — A non-orthogonal MCSCF program") | [index](<../genindex.html> "General Index")

# 3.3.10. RASSI — A RAS State Interaction Program¶

Program RASSI (RAS State Interaction) computes matrix elements of the Hamiltonian and other operators in a wave function basis, which consists of individually optimized CI expansions from the RASSCF program. Also, it solves the Schrödinger equation within the space of these wave functions. There are many possible applications for such type of calculations. The first important consideration to have into account is that RASSI computes the interaction among RASSCF states expanding the same set of configurations, that is, having the same active space size and number of electrons.

The RASSI program is routinely used to compute electronic transition moments, as it is shown in the Advanced Examples in the calculation of transition dipole moments for the excited states of the thiophene molecule using CASSCF-type wave functions. By default the program will compute the matrix elements and expectation values of all the operators for which SEWARD has computed the integrals and has stored them in the ONEINT file.

RASSCF (or CASSCF) individually optimized states are interacting and non-orthogonal. It is imperative when the states involved have different symmetry to transform the states to a common eigenstate basis in such a way that the wave function remains unchanged. The State Interaction calculation gives an unambiguous set of non-interacting and orthonormal eigenstates to the projected Schrödinger equation and also the overlaps between the original RASSCF wave functions and the eigenstates. The analysis of the original states in terms of RASSI eigenstates is very useful to identify spurious local minima and also to inspect the wave functions obtained in different single-root RASSCF calculations, which can be mixed and be of no help to compare the states.

Finally, the RASSI program can be applied in situations when there are two strongly interacting states and there are two very different MCSCF solutions. This is a typical situation in transition metal chemistry when there are many close states associated each one to a configuration of the transition metal atom. It is also the case when there are two close quasi-equivalent localized and delocalized solutions. RASSI can provide with a single set of orbitals able to represent, for instance, avoided crossings. RASSI will produce a number of files containing the natural orbitals for each one of the desired eigenstates to be used in subsequent calculations.

RASSI requires as input files the ONEINT and ORDINT integral files and the JOBIPH files from the RASSCF program containing the states which are going to be computed. The JOBIPH files have to be named consecutively as JOB001, JOB002, etc. The input for the RASSI module has to contain at least the definition of the number of states available in each of the input JOBIPH files. Block 3.3.10.1 lists the input file for the RASSI program in a calculation including two JOBIPH files (2 in the first line), the first one including three roots (3 in the first line) and the second five roots (5 in the first line). Each one of the following lines lists the number of these states within each JOBIPH file. Also in the input, keyword NATOrb indicates that three files (named sequentially NAT001, NAT002, and NAT003) will be created for the three lowest eigenstates.

Block 3.3.10.1 Sample input requesting the RASSI module to calculate the matrix elements and expectation values for eight interacting RASSCF states¶
    
    
    &RASSI
    NROFjobiph= 2 3 5; 1 2 3; 1 2 3 4 5
    NATOrb= 3
    

## 3.3.10.1. RASSI Output¶

The RASSI section of the Molcas output is basically divided in three parts. Initially, the program prints the information about the JOBIPH files and input file, optionally prints the wave functions, and checks that all the configuration spaces are the same in all the input states. In second place RASSI prints the expectation values of the one-electron operators, the Hamiltonian matrix, the overlap matrix, and the matrix elements of the one-electron operators, all for the basis of input RASSCF states. The third part starts with the eigenvectors and eigenvalues for the states computed in the new eigenbasis, as well as the overlap of the computed eigenstates with the input RASSCF states. After that, the expectation values and matrix elements of the one-electron operators are repeated on the basis of the new energy eigenstates. A final section informs about the occupation numbers of the natural orbitals computed by RASSI, if any.

In the Advanced Examples a detailed example of how to interpret the matrix elements output section for the thiophene molecule is displayed. The rest of the output is self-explanatory. It has to be remembered that to change the default origins for the one electron operators (the dipole moment operator uses the nuclear charge centroid and the higher order operators the center of the nuclear mass) keyword CENTer in GATEWAY must be used. Also, if multipoles higher than order two are required, the option MULTipole has to be used in GATEWAY.

The program RASSI can also be used to compute a spin–orbit Hamiltonian for the input CASSCF wave functions as defined above. The keyword AMFI has to be used in SEWARD to ensure that the corresponding integrals are available.

Block 3.3.10.2 Sample input requesting the RASSI module to calculate and diagonalize the spin–orbit Hamiltonian the ground and triplet excited state in water.¶
    
    
    &RASSI
    NROFjobiph= 2 1 1; 1; 1
    Spinorbit
    Ejob
    

The first JOBMIX file contains the wave function for the ground state and the second file the \\(^3B_2\\) state discussed above. The keyword Ejob makes the RASSI program use the CASPT2 energies which have been written on the JOBMIX files in the diagonal of the spin–orbit Hamiltonian. The output of this calculation will give four spin–orbit states and the corresponding transition properties, which can for example be used to compute the radiative lifetime of the triplet state.

## 3.3.10.2. RASSI — Basic and Most Common Keywords¶

NROFjob
    

Number of input files, number of roots, and roots for each file

EJOB/HDIAG
    

Read energies from input file / inline

SPIN
    

Compute spin–orbit matrix elements for spin properties

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
      * 3.3.10. RASSI — A RAS State Interaction Program
      * [3.3.11. CASVB — A non-orthogonal MCSCF program](<tut_casvb.html>)
      * [3.3.12. MOTRA — An Integral Transformation Program](<tut_motra.html>)
      * [3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program](<tut_guga.html>)
      * [3.3.14. MRCI — A Configuration Interaction Program](<tut_mrci.html>)
      * [3.3.15. CPF — A Coupled-Pair Functional Program](<tut_cpf.html>)
      * [3.3.16. CCSDT — A Set of Coupled-Cluster Programs](<tut_ccsdt.html>)
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

[previous](<tut_nevpt2.html> "3.3.9. NEVPT2 — \\\(n\\\)-Electron Valence State Second-Order Perturbation Theory") | [next](<tut_casvb.html> "3.3.11. CASVB — A non-orthogonal MCSCF program") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_rassi.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
