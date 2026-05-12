<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_optimiza.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization

[previous](<tut_ccsdt.html> "3.3.16. CCSDT — A Set of Coupled-Cluster Programs") | [next](<tut_mckinley.html> "3.3.18. MCKINLEY — A Program for Integral Second Derivatives") | [index](<../genindex.html> "General Index")

# 3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization¶

One of the most powerful functions of _ab initio_ calculations is geometry predictions. The minimum energy structure of a molecule for a given method and basis set is instructive especially when experiment is unable to determine the actual geometry. Molcas performs a geometry optimization with analytical gradients at the SCF, RASSCF and RASPT2 levels of calculation, and with numerical gradients for other methods.

In order to perform geometry optimization an input file must contain a loop, which includes several calls: calculation of integrals (SEWARD), calculation of energy (SCF, RASSCF, CASPT2), calculation of gradients (ALASKA), and calculation of the new geometry (SLAPAF).

This is an example of such input
    
    
    &GATEWAY
     coord= file.xyz
     basis= ANO-S-MB
    >> EXPORT MOLCAS_MAXITER=25
    >> Do While <<
    &SEWARD
    &SCF
    &SLAPAF
    >> EndDo <<
    

The initial coordinates will be taken from xyz file file.xyz, and the geometry will be optimized at the SCF level in this case. After the wave function calculation, calculation of gradients is required, although code ALASKA is automatically called by Molcas. SLAPAF in this case required the calculation of an energy minimum (no input). Other options are transition states (TS), minimum energy paths (MEP-search), etc The loop will be terminated if the geometry converges, or maximum number of iterations (MOLCAS_MAXITER) will be reached (the default value is 50).

There are several EMIL commands (see [Section 4.1.3.2](<../users.guide/emil.html#ug-sec-emil-commands>)) which can be used to control geometry optimization. For example, it is possible to execute some Molcas modules only once:
    
    
    >> IF ( ITER = 1 )
    * this part of the input will be executed only during the first iteration
    >> ENDIF
    

Program SLAPAF is tailored to use analytical or numerical gradients produced by ALASKA to relax the geometry of a molecule towards an energy minimum (default, no input required then) or a transition state. The program is also used for finding inter state crossings (ISC), conical intersections (CI), to compute reaction paths, intrinsic reaction coordinate (IRC) paths, etc.

## 3.3.17.1. SLAPAF — Basic and Most Common Keywords¶

TS
    

Computing a transition state

FindTS
    

Computing a transition state with a constraint

MEP-search
    

Computing a steepest-descent minimum reaction path

ITER
    

Number of iterations

INTErnal
    

Definition of the internal coordinates

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
      * 3.3.17. ALASKA and SLAPAF — A Molecular Structure Optimization
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

[previous](<tut_ccsdt.html> "3.3.16. CCSDT — A Set of Coupled-Cluster Programs") | [next](<tut_mckinley.html> "3.3.18. MCKINLEY — A Program for Integral Second Derivatives") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_optimiza.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
