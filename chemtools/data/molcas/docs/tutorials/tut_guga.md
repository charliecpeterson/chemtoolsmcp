<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_guga.html -->

[ ](<../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../index.html>)

3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program

[previous](<tut_motra.html> "3.3.12. MOTRA — An Integral Transformation Program") | [next](<tut_mrci.html> "3.3.14. MRCI — A Configuration Interaction Program") | [index](<../genindex.html> "General Index")

# 3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program¶

Several of the Configuration Interaction (CI) modules in Molcas use the GUGA module to compute the CI coupling coefficients. We continue our water calculations using the input file shown in the input below. The TITLe keyword behaves in a similar fashion as described in previous modules. There are several compulsory keywords of the GUGA module. The number of electrons to be correlated is specified using the ELECtrons keyword. We are correlating the valence electrons. The spin state is specified using the SPIN keyword.

Sample input requesting the the GUGA module to calculate the coupling coefficients for neutral triplet water in \\(C_{2v}\\) symmetry with six electrons in the active space:
    
    
    &GUGA
    Title= GUGA for C2v Water
    Electrons= 8; Spin= 3
    Inactive= 1 0 0 0; Active= 2 2 0 1
    CIAll= 1
    

The keywords CIALl and REFErence are mutually exclusive. We specify CIALl which will calculate the energy using all possible references functions that can be constructed using the input set of occupation numbers of the active orbitals regardless of the spin coupling (all configurations used to build the corresponding CASSCF wave function). Specific selected references can be chosen using the REFErence keyword. Either the ACTIve or INACtive keyword should be used for a meaningful calculation. The default for both keywords is zero for all symmetries. These keywords function in a similar fashion to these in the RASSCF program module. The INACtive keyword specifies the orbitals that are fully occupied in each symmetry in all the reference functions and the ACTIve keyword specifies the orbitals that may have varying occupations in all references. The selection of INACtive orbitals in the input above is forcing the bonding sp hybrid orbital to remain fully occupied in all reference states.

## 3.3.13.1. GUGA Output¶

The GUGA section of the output lists the possible configurations in the active space. There are nine possible triplet configurations of six electrons in five orbitals. Apart from the various types of orbital in each symmetry the GUGA section of the output also gives the number of states that will coupled with various states. There are no input files for the GUGA module but the calculated coupling coefficients are stored in CIGUGA.

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
      * 3.3.13. GUGA — A Configuration Interaction Coupling Coefficients Program
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

[previous](<tut_motra.html> "3.3.12. MOTRA — An Integral Transformation Program") | [next](<tut_mrci.html> "3.3.14. MRCI — A Configuration Interaction Program") | [index](<../genindex.html> "General Index")

[Get PDF](<../../Manual.pdf>) | [Show Source](<../_sources/tutorials/tut_guga.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
