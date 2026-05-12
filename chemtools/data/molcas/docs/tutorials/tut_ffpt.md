<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_ffpt.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.21. FFPT — A Finite Field Perturbation Program

[previous](<tut_genano.html> "3.3.20. GENANO — A Program to Generate ANO Basis Sets") | [next](<tut_vibrot.html> "3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules") | [index](<../genindex.html> "General Index")

# 3.3.21. FFPT — A Finite Field Perturbation Program¶

Many molecular properties of wave functions can be computed using the FFPT program module in Molcas. It adds the requested operator to the integrals computed by the SEWARD module. This must be done before the Molcas module calculating the required wave function is requested so the FFPT module is best run directly after the SEWARD module.

The TITLe keyword behaviors in a similar fashion to other Molcas modules. The sample input below contains the FFPT input requesting that the dipole moment operator be added to the integrals using the DIPOle keyword. The size and direction is specified using the COMP keyword which accepts free format input. We can compute the dipole of the molecule by numerical determination of the gradient of the energy curve determined for several values of the dipole operator. From the second derivative we can obtain the polarizability component.

Sample input requesting the FFPT module to include a dipole moment operator in the integral file:
    
    
    &FFPT
    Title= Finite Perturbation with a dipole in the x negative of strength 0.1 au
    FFPT
    Dipole
     Comp
     X -0.1
    

## 3.3.21.1. FFPT Output¶

The FFPT section of the output is short and self explanatory. The ONEINT file is updated with the requested operator.

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
      * [3.3.19. MCLR — A Program for Linear Response Calculations](<tut_mclr.html>)
      * [3.3.20. GENANO — A Program to Generate ANO Basis Sets](<tut_genano.html>)
      * 3.3.21. FFPT — A Finite Field Perturbation Program
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

[previous](<tut_genano.html> "3.3.20. GENANO — A Program to Generate ANO Basis Sets") | [next](<tut_vibrot.html> "3.3.22. VIBROT — A Program for Vibration–Rotation on Diatomic Molecules") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_ffpt.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
