<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_seward.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.4. SEWARD — An Integral Generation Program

[previous](<tut_gateway.html> "3.3.3. GATEWAY — Definition of geometry, basis sets, and symmetry") | [next](<tut_scf.html> "3.3.5. SCF — A Self-Consistent Field program and Kohn–Sham DFT") | [index](<../genindex.html> "General Index")

# 3.3.4. SEWARD — An Integral Generation Program¶

An _ab initio_ calculation always requires integrals. In the Molcas suite of programs, this function is supplied by the SEWARD module. SEWARD computes the one- and two-electron integrals for the molecule and basis set specified in the input to the program GATEWAY, which should be run before SEWARD. SEWARD can also be used to perform some property expectation calculations on the isolated molecule. The module is also used as an input parser for the reaction field and numerical quadrature parameters.

We commence our tutorial by calculating the integrals for a water molecule. The input is given in Block 3.3.4.1. Each Molcas module identifies input from a file by the name of the module. In the case of SEWARD, the program starts with the label `&SEWARD`, which is the first statement in the file shown below.

In normal cases no input is required for SEWARD, so the following input is optional. The first keyword used is TITLe. Only the first line of the title is printed in the output. The first title line is also saved in the integral file and appears in any subsequent programs that use the integrals calculated by SEWARD.

Block 3.3.4.1 Sample input requesting the SEWARD module to calculate the integrals for water in \\(C_{2v}\\) symmetry.¶
    
    
    &SEWARD
    Title
    Water - A Tutorial. The integrals of water are calculated using C2v symmetry
    

In more complicated cases more input may be needed, to specify certain types of integrals, that use of Cholesky decomposition techniques (CHOLesky keyword), etc. We refer to the specific sections of the User’s Guide for more information. The output from a SEWARD calculation is small and contains in principle only a list of the different types of integrals that are computed.

The integrals produced by the SEWARD module are stored in two files in the working directory. They are ascribed the FORTRAN names ONEINT and ORDINT which are automatically symbolically linked by the Molcas script to the file names $Project.OneInt and $Project.OrdInt, respectively or more specifically, in our case, water.OneInt and water.OrdInt, respectively. The default name for each symbolical name is contained in the corresponding program files of the directory $MOLCAS/shell. The ONEINT file contains the one-electron integrals. The ORDINT contains the ordered and packed two-electron integrals. Both files are used by later Molcas program modules.

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
      * 3.3.4. SEWARD — An Integral Generation Program
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

[previous](<tut_gateway.html> "3.3.3. GATEWAY — Definition of geometry, basis sets, and symmetry") | [next](<tut_scf.html> "3.3.5. SCF — A Self-Consistent Field program and Kohn–Sham DFT") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_seward.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
