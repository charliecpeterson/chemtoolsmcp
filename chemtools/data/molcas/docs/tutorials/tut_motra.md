<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_motra.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.12. MOTRA — An Integral Transformation Program

[previous](<tut_casvb.html> "3.3.11. CASVB — A non-orthogonal MCSCF program") | [next](<tut_guga.html> "3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program") | [index](<../genindex.html> "General Index")

# 3.3.12. MOTRA — An Integral Transformation Program¶

Integrals saved by the SEWARD module are stored in the Atomic Orbital (AO) basis. Some programs have their own procedures to transform the integrals into the Molecular Orbital (MO) basis. The Molcas MOTRA module performs this task for Configuration Interaction (CI), Coupled- and Modified Coupled-Pair (CPF and MCPF, respectively) and Coupled-Cluster (CC) calculations.

The sample input below contains the MOTRA input information for our continuing water calculation. We firstly specify that the RASSCF module interface file will be the source of the orbitals using the keyword JOBIph. The keyword FROZen is used to specify the number of orbitals in each symmetry which will not be correlated in subsequent calculations. This can also be performed in the corresponding MRCI, CPF or CC programs but is more efficient to freeze them here. Virtual orbitals can be deleted using the DELEte keyword.
    
    
    &MOTRA
    JobIph
    Frozen= 1 0 0 0
    

## 3.3.12.1. MOTRA Output¶

The MOTRA section of the output is short and self explanatory. The integral files produced by SEWARD, ONEINT and ORDINT, are used as input by the MOTRA module which produces the transformed symbolic files TRAONE and TRAINT, respectively. In our case, the files are called water.TraOne and water.TraInt, respectively.

The MOTRA module also requires input orbitals. If the LUMOrb keyword is specified the orbitals are taken from the INPORB file which can be any formated orbital file such as water.ScfOrb or water.RasOrb. The JOBIph keyword causes the MOTRA module to read the required orbitals from the JOBIPH file.

## 3.3.12.2. MOTRA — Basic and Most Common Keywords¶

FROZEN
    

By symmetry: non-correlated orbitals (default: core)

RFPErt
    

Previous reaction field introduced as a perturbation

LUMORB
    

Input orbital file as ASCII (INPORB)

JOBIPH
    

Input orbital file as binary (JOBOLD)

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
      * 3.3.12. MOTRA — An Integral Transformation Program
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

[previous](<tut_casvb.html> "3.3.11. CASVB — A non-orthogonal MCSCF program") | [next](<tut_guga.html> "3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_motra.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
