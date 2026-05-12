<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_mclr.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.19. MCLR — A Program for Linear Response Calculations

[previous](<tut_mckinley.html> "3.3.18. MCKINLEY — A Program for Integral Second Derivatives") | [next](<tut_genano.html> "3.3.20. GENANO — A Program to Generate ANO Basis Sets") | [index](<../genindex.html> "General Index")

# 3.3.19. MCLR — A Program for Linear Response Calculations¶

MCLR computes response calculations on single and multiconfigurational SCF wave functions. One of the basic uses of MCKINLEY and MCLR is to compute analytical Hessians (vibrational frequencies, IR intensities, etc). MCLR can also calculate the Lagrangian multipliers for a MCSCF state included in a state average optimization and construct the effective densities required for analytical gradients of such a state. The use of keyword RLXRoot in the RASSCF program is required. In both cases the explicit request of executing the MCLR module is not required and will be automatic. We postpone further discussion about MCLR to [Section 3.3.17](<tut_optimiza.html#tut-sec-structure>).

It follows an example of how to optimize an excited state from a previous State-Average (SA) CASSCF calculation.
    
    
    &GATEWAY
    Title= acrolein minimum optimization in excited state 2
    Coord=$MOLCAS/Coord/Acrolein.xyz
    Basis= sto-3g
    Group=NoSym
    >>> Do while
    &SEWARD
    &RASSCF
    Title= acrolein
    Spin= 1; nActEl= 6 0 0; Inactive= 12; Ras2= 5
    CiRoot= 3 3 1
    Rlxroot= 2
    &SLAPAF
    >>> EndDo
    

The root selected for optimization has been selected here with the keyword Rlxroot in RASSCF, but it is also possible to select it with keyword SALA in MCLR.

Now if follows an example as how to compute the analytical hessian for the lowest state of each symmetry in a CASSCF calculation (SCF, DFT, and RASSCF analytical Hessians are also available).
    
    
    &GATEWAY
    Title=p-benzoquinone anion. Casscf optimized geometry.
    Coord = $MOLCAS/Coord/benzoquinone.xyz
    Basis= sto-3g
    Group= X Y Z
    &SEWARD
    &RASSCF
    TITLE=p-benzoquinone anion. 2B3u state.
    SYMMETRY=2; SPIN=2; NACTEL=9 0 0
    INACTIVE=8  0  5  0  7  0  4  0
    RAS2    =0  3  0  1  0  3  0  1
    
    &MCKINLEY; Perturbation=Hessian
    

The MCLR is automatically called after MCKINLEY and it is not needed in the input.

## 3.3.19.1. MCLR program — Basic and Most Common Keywords¶

SALA
    

Root to relax in geometry optimizations

ITER
    

Number of iterations

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
      * [3.3.16. CCSDT — A Set of Coupled-Cluster Programs](<tut_ccsdt.html>)
      * [3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization](<tut_optimiza.html>)
      * [3.3.18. MCKINLEY — A Program for Integral Second Derivatives](<tut_mckinley.html>)
      * 3.3.19. MCLR — A Program for Linear Response Calculations
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

[previous](<tut_mckinley.html> "3.3.18. MCKINLEY — A Program for Integral Second Derivatives") | [next](<tut_genano.html> "3.3.20. GENANO — A Program to Generate ANO Basis Sets") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_mclr.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
